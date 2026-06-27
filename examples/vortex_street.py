"""Flow past a circular cylinder: the von Karman vortex street.

Runs a D2Q9 BGK simulation of uniform flow past a cylinder at a Reynolds number
in the vortex-shedding regime and produces:

* ``figures/vortex_street.png`` -- a vorticity snapshot of the wake,
* ``figures/vortex_street.gif`` -- an animation of the shedding vorticity,
* ``figures/lift_drag.png``      -- lift/drag coefficient time series,

and prints the measured Strouhal number ``St = f D / U``.

Usage::

    python examples/vortex_street.py            # full showcase run
    python examples/vortex_street.py --quick    # smaller/faster preview
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from lbm import (
    LatticeBoltzmann,
    cylinder_mask,
    vorticity,
    force_coefficients,
    strouhal_number,
)

HERE = os.path.dirname(__file__)
FIG_DIR = os.path.join(HERE, "..", "figures")


def build_simulation(nx, ny, diameter, U, Re):
    """Set up a cylinder-in-channel simulation."""
    nu = U * diameter / Re
    tau = 3.0 * nu + 0.5
    cx, cy = nx // 4, ny // 2 + 2          # small offset breaks the symmetry
    solid = cylinder_mask(nx, ny, cx, cy, diameter / 2.0)

    # Uniform inlet with a tiny transverse perturbation to seed the instability.
    yv = np.arange(ny)
    inlet = np.zeros((2, ny))
    inlet[0] = U * (1.0 + 1e-2 * np.sin(2.0 * np.pi * yv / ny))

    sim = LatticeBoltzmann(nx, ny, tau, solid=solid, inlet_velocity=inlet)
    sim.cx, sim.cy = cx, cy                 # remember the cylinder centre
    return sim, solid, tau, nu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="smaller grid and fewer steps for a fast preview")
    args = parser.parse_args()

    if args.quick:
        nx, ny, D = 220, 90, 16.0
        warmup, record = 6000, 9000
    else:
        nx, ny, D = 400, 150, 20.0
        warmup, record = 10000, 16000

    U = 0.06
    Re = 90.0
    sample_every = 8
    frame_every = 60 if args.quick else 120

    sim, solid, tau, nu = build_simulation(nx, ny, D, U, Re)
    print(f"grid={nx}x{ny}  D={D}  U={U}  Re={Re}  nu={nu:.4f}  tau={tau:.4f}")

    probe_x = int(sim.cx + 3 * D)
    probe_y = sim.cy

    # ---- warm-up: let the wake develop -----------------------------------
    print(f"warming up for {warmup} steps ...")
    sim.run(warmup)

    # ---- recording: probe signal, forces, animation frames ---------------
    print(f"recording for {record} steps ...")
    times, uy_probe = [], []
    cd_series, cl_series = [], []
    frames = []

    for n in range(record):
        sim.step()
        if n % sample_every == 0:
            _, u = sim.macroscopic()
            times.append(sim.time)
            uy_probe.append(u[1, probe_x, probe_y])
            cd, cl = force_coefficients(sim.force_on_solid, U, D)
            cd_series.append(cd)
            cl_series.append(cl)
        if n % frame_every == 0:
            _, u = sim.macroscopic()
            frames.append(vorticity(u, solid=solid).T)

    times = np.array(times)
    uy_probe = np.array(uy_probe)
    cd_series = np.array(cd_series)
    cl_series = np.array(cl_series)

    # ---- Strouhal number -------------------------------------------------
    dt = sample_every                      # lattice time units between samples
    St, f_peak = strouhal_number(uy_probe, D, U, dt=dt)
    print(f"shedding frequency f = {f_peak:.3e} / step")
    print(f"Strouhal number  St = f D / U = {St:.3f}")
    in_range = 0.15 <= St <= 0.18
    print(f"St in expected 0.15-0.18 range: {in_range}")
    print(f"mean Cd = {cd_series.mean():.3f},  Cl amplitude = "
          f"{0.5 * (cl_series.max() - cl_series.min()):.3f}")

    os.makedirs(FIG_DIR, exist_ok=True)

    # ---- vorticity snapshot ----------------------------------------------
    vlim = np.nanpercentile(np.abs(frames[-1]), 99.5)
    fig, ax = plt.subplots(figsize=(10, 3.4))
    im = ax.imshow(frames[-1], origin="lower", cmap="RdBu_r",
                   vmin=-vlim, vmax=vlim, interpolation="bilinear")
    ax.set_title(f"Vorticity of the wake  (Re={Re:.0f},  St={St:.3f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label=r"$\omega_z$")
    fig.tight_layout()
    snap = os.path.join(FIG_DIR, "vortex_street.png")
    fig.savefig(snap, dpi=130)
    print(f"saved {snap}")
    plt.close(fig)

    # ---- animation -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 3.4))
    im = ax.imshow(frames[0], origin="lower", cmap="RdBu_r",
                   vmin=-vlim, vmax=vlim, interpolation="bilinear")
    ax.set_title(f"von Karman vortex street  (Re={Re:.0f})")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    def update(k):
        im.set_data(frames[k])
        return (im,)

    anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                   interval=80, blit=True)
    gif = os.path.join(FIG_DIR, "vortex_street.gif")
    anim.save(gif, writer=animation.PillowWriter(fps=12))
    print(f"saved {gif}  ({len(frames)} frames)")
    plt.close(fig)

    # ---- lift / drag time series -----------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(times, cd_series, label="drag $C_D$", color="C3")
    ax.plot(times, cl_series, label="lift $C_L$", color="C0")
    ax.set_xlabel("time step")
    ax.set_ylabel("force coefficient")
    ax.set_title("Lift and drag on the cylinder")
    ax.legend()
    fig.tight_layout()
    ld = os.path.join(FIG_DIR, "lift_drag.png")
    fig.savefig(ld, dpi=130)
    print(f"saved {ld}")
    plt.close(fig)


if __name__ == "__main__":
    main()
