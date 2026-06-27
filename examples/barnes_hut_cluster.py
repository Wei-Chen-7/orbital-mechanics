"""Barnes-Hut showcase: a star cluster animation and an O(N log N) timing plot.

* ``figures/cluster.gif`` -- a few hundred bodies drawn from a Plummer model,
  evolved with the Barnes-Hut tree code.
* ``figures/barnes_hut_scaling.png`` -- wall-clock time per force evaluation for
  the direct O(N^2) sum vs. Barnes-Hut, showing the expected crossover.
"""

import time

import numpy as np

from _common import fig_path, use_style
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from nbody import initial_conditions as ic, simulate, forces
from nbody.barnes_hut import barnes_hut_accelerations, accelerations_bh


def make_cluster_gif():
    sys = ic.plummer_sphere(n=400, total_mass=1.0, radius=1.0, seed=7)
    n_frames, steps_per_frame = 160, 5
    dt = 4e-3
    traj = simulate(sys, dt, n_frames * steps_per_frame,
                    record_every=steps_per_frame,
                    acc_func=barnes_hut_accelerations(theta=0.6))
    print(f"cluster energy error (max): {traj.energy_error().max():.2e}")

    pos = traj.positions
    speed = np.linalg.norm(traj.velocities, axis=2)  # (frames, N) for colour

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.set_aspect("equal")
    ax.set_facecolor("black")
    lim = 2.5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Plummer cluster, Barnes-Hut ($\\theta=0.6$)")

    scat = ax.scatter(pos[0, :, 0], pos[0, :, 1], s=6, c=speed[0],
                      cmap="plasma", vmin=0, vmax=speed.max())

    def update(frame):
        scat.set_offsets(pos[frame])
        scat.set_array(speed[frame])
        return (scat,)

    anim = FuncAnimation(fig, update, frames=len(pos), interval=50, blit=True)
    anim.save(fig_path("cluster.gif"), writer=PillowWriter(fps=20), dpi=80)
    print("wrote figures/cluster.gif")


def make_scaling_plot():
    sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    direct_t, bh_t = [], []
    rng = np.random.default_rng(0)
    for n in sizes:
        pos = rng.normal(size=(n, 2))
        m = rng.uniform(0.5, 1.5, n)
        t0 = time.perf_counter()
        forces.accelerations(pos, m, softening=0.05)
        direct_t.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        accelerations_bh(pos, m, softening=0.05, theta=0.5)
        bh_t.append(time.perf_counter() - t0)

    sizes = np.array(sizes, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(sizes, direct_t, "o-", color="#cc4444", label="direct  $O(N^2)$")
    ax.loglog(sizes, bh_t, "o-", color="#22aa66", label=r"Barnes-Hut  $O(N\log N)$")
    # Reference slopes anchored at the first point.
    ax.loglog(sizes, direct_t[0] * (sizes / sizes[0]) ** 2, "k--", lw=0.7,
              alpha=0.6, label=r"$\propto N^2$")
    ax.loglog(sizes, bh_t[0] * (sizes / sizes[0]) * np.log2(sizes) / np.log2(sizes[0]),
              "k:", lw=0.9, alpha=0.6, label=r"$\propto N\log N$")
    ax.set_xlabel("number of bodies  $N$")
    ax.set_ylabel("time per force evaluation  [s]")
    ax.set_title("Direct sum vs. Barnes-Hut")
    ax.legend(frameon=False)
    fig.savefig(fig_path("barnes_hut_scaling.png"))
    print("wrote figures/barnes_hut_scaling.png")


def main():
    use_style()
    make_scaling_plot()
    make_cluster_gif()


if __name__ == "__main__":
    main()
