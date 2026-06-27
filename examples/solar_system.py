"""Integrate the inner solar system + Jupiter and plot the orbits.

Units are AU / years / solar masses (so G = 4 pi^2).  Produces
``figures/solar_system.png`` and prints the energy drift over the run.
"""

import numpy as np

from _common import fig_path, use_style
import matplotlib.pyplot as plt

from nbody import initial_conditions as ic, simulate

PLANETS = ["Mercury", "Venus", "Earth", "Mars", "Jupiter"]
COLORS = ["#9c6b4f", "#d9a441", "#3b7dd8", "#c1440e", "#c98a3a"]


def main():
    use_style()
    sys = ic.solar_system(planets=PLANETS, seed=3)

    years = 12.0
    dt = 2e-3  # ~0.7 days; well inside Mercury's 88-day orbit
    n_steps = int(years / dt)
    traj = simulate(sys, dt, n_steps, record_every=20)

    err = traj.energy_error()
    print(f"Integrated {years:g} yr in {n_steps} steps.")
    print(f"Max relative energy error: {err.max():.2e}")

    fig, ax = plt.subplots(figsize=(7, 7))
    pos = traj.positions
    ax.scatter([0], [0], c="gold", s=180, edgecolor="k", lw=0.5, zorder=5,
               label="Sun")
    for k, (name, color) in enumerate(zip(PLANETS, COLORS), start=1):
        ax.plot(pos[:, k, 0], pos[:, k, 1], color=color, lw=1.0, alpha=0.9)
        ax.scatter(pos[-1, k, 0], pos[-1, k, 1], color=color, s=40,
                   edgecolor="k", lw=0.4, zorder=6, label=name)

    ax.set_aspect("equal")
    ax.set_xlabel("x  [AU]")
    ax.set_ylabel("y  [AU]")
    ax.set_title(f"Inner solar system + Jupiter, {years:g} years (velocity-Verlet)")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.savefig(fig_path("solar_system.png"))
    print("wrote figures/solar_system.png")


if __name__ == "__main__":
    main()
