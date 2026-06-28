"""Integrator properties: symplectic boundedness vs RK4 drift, order, reversibility."""

import numpy as np
import pytest

from nbody import initial_conditions as ic, simulate, diagnostics as diag


def test_verlet_bounded_rk4_drifts():
    """The key claim: Verlet energy error is bounded (no secular growth) while
    RK4 drifts.  We run the same eccentric orbit with a deliberately coarse
    timestep for many orbits and compare the error in the last fifth of the run
    to the error in the first fifth."""
    sys = ic.two_body(m1=1.0, m2=1.0, a=1.0, e=0.3, G=1.0)
    mu = sys.G * sys.masses.sum()
    period = diag.kepler_period(1.0, mu)

    steps_per_orbit, n_orbits = 40, 800
    dt = period / steps_per_orbit
    n_steps = int(n_orbits * period / dt)

    def growth(method):
        traj = simulate(sys, dt, n_steps, method=method,
                        record_every=steps_per_orbit)
        err = traj.energy_error()
        n = len(err)
        first = err[: n // 5].max()
        last = err[4 * n // 5:].max()
        return err, last / max(first, 1e-30)

    verlet_err, verlet_growth = growth("verlet")
    rk4_err, rk4_growth = growth("rk4")

    # Verlet: bounded and small, essentially no growth between halves.
    assert verlet_err.max() < 0.05
    assert verlet_growth < 1.5

    # RK4: error in the late run is many times the early-run error (drift).
    assert rk4_growth > 3.0
    # And by the end RK4 is far worse than Verlet despite being higher order.
    assert rk4_err[-1] > 5 * verlet_err[-1]


def test_verlet_second_order_convergence():
    """Halving dt should cut the one-period position error by ~4x (2nd order)."""
    sys = ic.two_body(m1=1.0, m2=1e-3, a=1.0, e=0.2, G=1.0)
    mu = sys.G * sys.masses.sum()
    period = diag.kepler_period(1.0, mu)

    def end_error(n_steps):
        traj = simulate(sys, period / n_steps, n_steps, record_every=n_steps)
        return np.abs(traj.positions[-1] - sys.positions).max()

    e_coarse = end_error(400)
    e_fine = end_error(800)
    ratio = e_coarse / e_fine
    assert 3.5 < ratio < 4.5  # ~4 for a 2nd-order method


def test_verlet_time_reversible():
    """Integrate forward then backward (negate velocities) -> return to start."""
    sys = ic.two_body(m1=1.0, m2=1.0, a=1.0, e=0.4, G=1.0)
    mu = sys.G * sys.masses.sum()
    period = diag.kepler_period(1.0, mu)
    dt, n = period / 2000, 2000

    fwd = simulate(sys, dt, n)
    reversed_state = fwd.state_at(-1)
    reversed_state.velocities = -reversed_state.velocities
    back = simulate(reversed_state, dt, n)

    assert np.abs(back.positions[-1] - sys.positions).max() < 1e-8


def test_momentum_conserved_by_integrator():
    sys = ic.figure_eight()
    traj = simulate(sys, 1e-3, 5000)
    p = traj.linear_momentum()
    assert np.abs(p).max() < 1e-10
