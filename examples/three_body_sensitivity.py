"""Sensitivity to initial conditions in the chaotic three-body problem.

Two copies of the Pythagorean (Burrau) problem are started a tiny distance
``delta`` apart.  Their separation in phase space grows roughly exponentially
until the trajectories are completely uncorrelated -- the hallmark of
deterministic chaos.  Produces ``figures/three_body_sensitivity.png``.
"""

import numpy as np

from _common import fig_path, use_style
import matplotlib.pyplot as plt

from nbody import initial_conditions as ic, simulate


def main():
    use_style()
    base = ic.pythagorean()
    base.softening = 1e-3  # tame the violent close encounters slightly

    delta = 1e-9
    perturbed = base.copy()
    perturbed.positions[0, 0] += delta  # nudge one coordinate

    dt = 2e-4
    n_steps = 350_000
    record_every = 200
    t1 = simulate(base, dt, n_steps, record_every=record_every)
    t2 = simulate(perturbed, dt, n_steps, record_every=record_every)

    sep = np.linalg.norm(
        (t1.positions - t2.positions).reshape(t1.n_frames, -1), axis=1
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.semilogy(t1.times, sep, color="#cc4444", lw=1.2)
    ax1.axhline(delta, color="k", ls="--", lw=0.8,
                label=f"initial separation = {delta:g}")
    ax1.set_xlabel("time")
    ax1.set_ylabel("phase-space separation  $\\|\\Delta r\\|$")
    ax1.set_title("Exponential divergence (note log scale)")
    ax1.legend(frameon=False, loc="lower right")

    for k, c in enumerate(["#3b7dd8", "#cc4444", "#22aa66"]):
        ax2.plot(t1.positions[:, k, 0], t1.positions[:, k, 1], color=c, lw=0.7)
        ax2.scatter(*base.positions[k], color=c, s=50, edgecolor="k", lw=0.5,
                    zorder=5)
    # Zoom on the interaction region: one body is eventually ejected and would
    # otherwise flatten the whole tangle into a couple of straight lines.
    ax2.set_xlim(-6, 6)
    ax2.set_ylim(-6, 6)
    ax2.set_aspect("equal")
    ax2.set_title("Pythagorean 3-body dance (zoomed)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")

    fig.savefig(fig_path("three_body_sensitivity.png"))
    print(f"final separation: {sep[-1]:.3e}  (started at {delta:g})")
    print(f"amplification factor: {sep[-1] / delta:.2e}")
    print("wrote figures/three_body_sensitivity.png")


if __name__ == "__main__":
    main()
