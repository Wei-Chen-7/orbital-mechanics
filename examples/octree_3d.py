"""3D Barnes-Hut octree showcase: a star cluster evolved with the tree code.

Confirms the octree matches the 3D direct sum, then evolves a 3D Plummer-sphere
cluster entirely with Barnes-Hut and renders a slowly rotating view to
``figures/cluster_3d.gif``.
"""

import numpy as np

from _common import fig_path, use_style
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

from nbody import initial_conditions as ic, simulate, forces
from nbody.barnes_hut import barnes_hut_accelerations, accelerations_bh


def report_accuracy():
    """Print the octree force error against the exact 3D direct sum."""
    cluster = ic.plummer_sphere(n=400, seed=7, dim=3)
    direct = forces.accelerations(cluster.positions, cluster.masses,
                                  G=cluster.G, softening=cluster.softening)
    for theta in (0.0, 0.3, 0.6):
        bh = accelerations_bh(cluster.positions, cluster.masses, G=cluster.G,
                              softening=cluster.softening, theta=theta)
        rel = (np.linalg.norm(bh - direct, axis=1)
               / np.linalg.norm(direct, axis=1))
        print(f"theta={theta}: mean force error {rel.mean():.2e}, "
              f"max {rel.max():.2e}")


def make_gif():
    sys = ic.plummer_sphere(n=300, total_mass=1.0, radius=1.0, seed=7, dim=3)
    n_frames, steps_per_frame = 110, 3
    dt = 4e-3
    traj = simulate(sys, dt, n_frames * steps_per_frame,
                    record_every=steps_per_frame,
                    acc_func=barnes_hut_accelerations(theta=0.6))
    print(f"3D cluster energy error (max): {traj.energy_error().max():.2e}")

    pos = traj.positions
    speed = np.linalg.norm(traj.velocities, axis=2)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    lim = 2.0
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_axis_off()
    ax.set_title("3D Plummer cluster, Barnes-Hut octree ($\\theta=0.6$)",
                 color="white")

    scat = ax.scatter(pos[0, :, 0], pos[0, :, 1], pos[0, :, 2],
                      s=6, c=speed[0], cmap="plasma",
                      vmin=0, vmax=speed.max(), depthshade=False)

    def update(frame):
        scat._offsets3d = (pos[frame, :, 0], pos[frame, :, 1], pos[frame, :, 2])
        scat.set_array(speed[frame])
        ax.view_init(elev=18, azim=frame)  # slow rotation
        return (scat,)

    anim = FuncAnimation(fig, update, frames=len(pos), interval=50, blit=False)
    anim.save(fig_path("cluster_3d.gif"), writer=PillowWriter(fps=20), dpi=80)
    print("wrote figures/cluster_3d.gif")


def main():
    use_style()
    report_accuracy()
    make_gif()


if __name__ == "__main__":
    main()
