"""Symplectic vs. non-symplectic: bounded energy error vs. secular drift.

Runs the same eccentric two-body orbit with velocity-Verlet and with RK4 at a
deliberately coarse timestep for many orbits.  Verlet's energy error stays in a
bounded band; RK4's grows steadily even though RK4 is the higher-order method.
Produces ``figures/energy_comparison.png``.
"""

import numpy as np

from _common import fig_path, use_style
import matplotlib.pyplot as plt

from nbody import initial_conditions as ic, simulate, diagnostics as diag


def main():
    use_style()
    sys = ic.two_body(m1=1.0, m2=1.0, a=1.0, e=0.3, G=1.0)
    mu = sys.G * sys.masses.sum()
    period = diag.kepler_period(1.0, mu)

    steps_per_orbit, n_orbits = 50, 3000
    dt = period / steps_per_orbit
    n_steps = int(n_orbits * period / dt)

    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {
        "verlet": ("#22aa66", "velocity-Verlet (symplectic)"),
        "rk4": ("#cc4444", "RK4 (non-symplectic)"),
    }
    results = {}
    for method, (color, label) in styles.items():
        traj = simulate(sys, dt, n_steps, method=method,
                        record_every=steps_per_orbit)
        e = traj.total_energy()
        rel = (e - e[0]) / abs(e[0])
        ax.plot(traj.times / period, rel, color=color, lw=1.2, label=label)
        results[method] = np.abs(rel)

    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel("time  [orbital periods]")
    ax.set_ylabel(r"relative energy error  $(E(t)-E_0)/|E_0|$")
    ax.set_title(f"Energy drift over {n_orbits} orbits  (dt = T/{steps_per_orbit})")
    ax.legend(loc="lower left", frameon=False)
    fig.savefig(fig_path("energy_comparison.png"))

    print(f"Verlet max |dE/E|: {results['verlet'].max():.2e} (bounded)")
    print(f"RK4    max |dE/E|: {results['rk4'].max():.2e} (drifts)")
    print("wrote figures/energy_comparison.png")


if __name__ == "__main__":
    main()
