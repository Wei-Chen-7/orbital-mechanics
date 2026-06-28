"""Plane Poiseuille flow: numerical profile vs the analytic parabola.

Drives a no-slip channel with a constant body force, runs to steady state and
plots the simulated streamwise velocity against the exact solution.  Saves
``figures/poiseuille.png`` and prints the maximum relative error.

Usage::

    python examples/poiseuille.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lbm import LatticeBoltzmann, channel_walls, poiseuille_profile

HERE = os.path.dirname(__file__)
FIG_DIR = os.path.join(HERE, "..", "figures")


def main():
    nx, ny = 16, 42
    tau = 0.8                       # nu = 0.1
    force = 2e-5

    solid = channel_walls(nx, ny)
    sim = LatticeBoltzmann(nx, ny, tau, solid=solid, force=(force, 0.0))

    # Run to steady state (monitor the change in the mid-channel profile).
    prev = None
    for n in range(40000):
        _, u = sim.step()
        if n % 500 == 0 and n > 0:
            cur = u[0, nx // 2, :].copy()
            if prev is not None and np.nanmax(np.abs(cur - prev)) / np.nanmax(np.abs(cur)) < 1e-10:
                print(f"steady state reached after {n} steps")
                break
            prev = cur

    _, u = sim.macroscopic()
    ux = u[0, nx // 2, :]
    y, ua = poiseuille_profile(ny, force, sim.nu)

    fluid = np.isfinite(ua)
    max_rel_err = np.max(np.abs(ux[fluid] - ua[fluid])) / np.max(ua[fluid])
    print(f"nu = {sim.nu:.4f},  u_max = {np.nanmax(ux):.6f}")
    print(f"max relative error vs analytic = {max_rel_err:.3%}")

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(ua[fluid], y[fluid], "-", lw=2, label="analytic parabola", color="0.4")
    ax.plot(ux[fluid], y[fluid], "o", ms=5, label="LBM (D2Q9)", color="C0")
    ax.set_xlabel("streamwise velocity $u_x$")
    ax.set_ylabel("channel coordinate $y$")
    ax.set_title(f"Plane Poiseuille flow\nmax rel. error = {max_rel_err:.2%}")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "poiseuille.png")
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
