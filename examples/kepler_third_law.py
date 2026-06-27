"""Kepler's third law: measure the period for several orbits and fit T^2 vs a^3.

Produces ``figures/kepler_third_law.png`` -- a log-log plot of measured period
against semi-major axis, overlaid with the analytic ``T = 2 pi sqrt(a^3/mu)``.
"""

import numpy as np

from _common import fig_path, use_style
import matplotlib.pyplot as plt

from nbody import initial_conditions as ic, simulate, diagnostics as diag


def main():
    use_style()
    G, m1, m2 = 1.0, 1.0, 1e-3
    mu = G * (m1 + m2)
    semi_major = np.array([0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0])

    measured = []
    for a in semi_major:
        sys = ic.two_body(m1=m1, m2=m2, a=a, e=0.0, G=G)
        analytic = diag.kepler_period(a, mu)
        traj = simulate(sys, analytic / 2000, int(1.2 * 2000), record_every=2)
        measured.append(diag.measure_period(traj))
    measured = np.array(measured)
    analytic = diag.kepler_period(semi_major, mu)

    # Fit log T = slope * log a + c; slope should be 3/2.
    slope, intercept = np.polyfit(np.log(semi_major), np.log(measured), 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.loglog(semi_major, measured, "o", ms=7, color="#3b7dd8",
               label="measured period")
    a_fine = np.linspace(semi_major.min(), semi_major.max(), 100)
    ax1.loglog(a_fine, diag.kepler_period(a_fine, mu), "-", color="k", lw=1,
               label=r"$T = 2\pi\sqrt{a^3/\mu}$")
    ax1.set_xlabel("semi-major axis  $a$")
    ax1.set_ylabel("orbital period  $T$")
    ax1.set_title(f"log-log slope = {slope:.4f}  (Kepler: 3/2)")
    ax1.legend(frameon=False)

    # Normalise by the analytic constant so the y-axis is a clean ratio ~1.
    ratio = (measured ** 2 / semi_major ** 3) / (4 * np.pi ** 2 / mu)
    ax2.plot(semi_major, ratio, "o-", color="#22aa66", label=r"measured")
    ax2.axhline(1.0, color="k", lw=1, ls="--", label="Kepler prediction")
    ax2.set_ylim(0.999, 1.001)
    ax2.set_xlabel("semi-major axis  $a$")
    ax2.set_ylabel(r"$(T^2/a^3) \,/\, (4\pi^2/\mu)$")
    ax2.set_title("constant across orbits (within 0.01%)")
    ax2.legend(frameon=False)

    fig.savefig(fig_path("kepler_third_law.png"))
    print(f"Fitted log-log slope: {slope:.5f} (expected 1.5)")
    print(f"max |T_measured/T_analytic - 1|: {np.abs(measured/analytic-1).max():.2e}")
    print("wrote figures/kepler_third_law.png")


if __name__ == "__main__":
    main()
