"""The figure-eight three-body choreography: static plot + animated GIF.

Three equal masses chase each other around a single figure-eight curve.
Produces ``figures/figure_eight.png`` and ``figures/figure_eight.gif``.
"""

import numpy as np

from _common import fig_path, use_style
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from nbody import initial_conditions as ic, simulate


def main():
    use_style()
    sys = ic.figure_eight()
    period = ic.FIGURE_EIGHT_PERIOD

    n_frames = 150
    steps_per_frame = 40
    dt = period / (n_frames * steps_per_frame)
    traj = simulate(sys, dt, n_frames * steps_per_frame,
                    record_every=steps_per_frame)
    pos = traj.positions  # (n_frames+1, 3, 2)
    colors = ["#3b7dd8", "#cc4444", "#22aa66"]
    # Over one period each body traverses the whole curve, so a single body's
    # path is the complete figure-eight.
    curve = pos[:, 0, :]

    # --- static figure: the shared orbit -------------------------------
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(curve[:, 0], curve[:, 1], color="0.6", lw=1.0)
    for k, c in enumerate(colors):
        ax.scatter(pos[-1, k, 0], pos[-1, k, 1], color=c, s=60,
                   edgecolor="k", lw=0.5, zorder=5)
    ax.set_aspect("equal")
    ax.set_title("Figure-eight choreography (equal masses)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.savefig(fig_path("figure_eight.png"))
    print("wrote figures/figure_eight.png")

    # --- animation -----------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    ax2.set_aspect("equal")
    ax2.plot(curve[:, 0], curve[:, 1], color="0.85", lw=0.8)
    ax2.set_xlim(curve[:, 0].min() - 0.2, curve[:, 0].max() + 0.2)
    ax2.set_ylim(curve[:, 1].min() - 0.2, curve[:, 1].max() + 0.2)
    ax2.set_title("Figure-eight three-body choreography")
    ax2.set_xticks([])
    ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_visible(False)

    trails = [ax2.plot([], [], color=c, lw=1.2, alpha=0.7)[0] for c in colors]
    dots = [ax2.plot([], [], "o", color=c, ms=9, mec="k", mew=0.5)[0]
            for c in colors]
    tail = 25

    def update(frame):
        lo = max(0, frame - tail)
        for k in range(3):
            trails[k].set_data(pos[lo:frame + 1, k, 0], pos[lo:frame + 1, k, 1])
            dots[k].set_data([pos[frame, k, 0]], [pos[frame, k, 1]])
        return trails + dots

    anim = FuncAnimation(fig2, update, frames=len(pos), interval=40, blit=True)
    anim.save(fig_path("figure_eight.gif"), writer=PillowWriter(fps=25))
    print("wrote figures/figure_eight.gif")


if __name__ == "__main__":
    main()
