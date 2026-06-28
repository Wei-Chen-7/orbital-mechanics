"""Two-body / Kepler physics: closed ellipse, conservation, third law.

These are the headline correctness checks.  They assert *physics* -- a closed
non-precessing ellipse, conserved energy and angular momentum, and
T^2 proportional to a^3 -- not merely that the code runs.
"""

import numpy as np
import pytest

from nbody import initial_conditions as ic, simulate, diagnostics as diag


def _eccentric_orbit():
    sys = ic.two_body(m1=1.0, m2=1.0, a=1.0, e=0.5, G=1.0)
    mu = sys.G * sys.masses.sum()
    period = diag.kepler_period(1.0, mu)
    return sys, mu, period


def test_energy_and_angular_momentum_conserved():
    sys, mu, period = _eccentric_orbit()
    dt = period / 2000
    traj = simulate(sys, dt, int(40 * period / dt), record_every=10)

    # Energy bounded to a tight tolerance over 40 orbits.
    assert traj.energy_error().max() < 1e-4

    # Angular momentum essentially exact (Verlet conserves it to round-off).
    L = traj.angular_momentum()[:, 0]
    assert np.max(np.abs((L - L[0]) / L[0])) < 1e-10


def test_closed_ellipse_no_precession():
    """The Laplace-Runge-Lenz vector is conserved -> the ellipse does not
    precess and therefore closes on itself."""
    sys, mu, period = _eccentric_orbit()
    dt = period / 4000
    traj = simulate(sys, dt, int(40 * period / dt), record_every=20)

    r, v = diag.relative_orbit(traj)
    evec = diag.eccentricity_vector(r, v, mu)
    emag = np.linalg.norm(evec, axis=1)

    # Eccentricity magnitude stays at its analytic value 0.5.
    assert emag.mean() == pytest.approx(0.5, abs=1e-3)
    assert emag.std() < 1e-4

    # Perihelion direction drifts < 0.1 deg over 40 orbits (no secular precession).
    direction = evec / emag[:, None]
    cos_drift = np.clip(direction @ direction[0], -1.0, 1.0)
    max_drift_deg = np.degrees(np.arccos(cos_drift).max())
    assert max_drift_deg < 0.1


def test_orbit_returns_to_start_after_one_period():
    sys, mu, period = _eccentric_orbit()
    dt = period / 8000
    traj = simulate(sys, dt, 8000, record_every=1)
    # After exactly one period the bodies come back to their starting state.
    assert np.abs(traj.positions[-1] - traj.positions[0]).max() < 1e-3
    assert np.abs(traj.velocities[-1] - traj.velocities[0]).max() < 1e-3


def test_orbital_elements_recovered():
    # Recover (a, e) from the instantaneous state via vis-viva + LRL vector.
    sys = ic.two_body(m1=1.0, m2=0.5, a=2.0, e=0.4, G=1.0)
    mu = sys.G * sys.masses.sum()
    r = sys.positions[1] - sys.positions[0]
    v = sys.velocities[1] - sys.velocities[0]
    a, e = diag.orbital_elements(r, v, mu)
    assert a == pytest.approx(2.0, rel=1e-10)
    assert e == pytest.approx(0.4, rel=1e-10)


def test_keplers_third_law():
    """T^2 / a^3 is the same constant (= 4 pi^2 / mu) for every orbit."""
    G, m1, m2 = 1.0, 1.0, 1e-3
    mu = G * (m1 + m2)
    semi_major = [0.5, 1.0, 1.5, 2.0, 3.0]
    ratios = []
    for a in semi_major:
        sys = ic.two_body(m1=m1, m2=m2, a=a, e=0.0, G=G)
        analytic = diag.kepler_period(a, mu)
        traj = simulate(sys, analytic / 1500, int(1.2 * 1500), record_every=3)
        measured = diag.measure_period(traj)
        assert measured == pytest.approx(analytic, rel=1e-3)
        ratios.append(measured ** 2 / a ** 3)

    ratios = np.array(ratios)
    # All ratios equal to within 0.1%, and equal to 4 pi^2 / mu.
    assert ratios.std() / ratios.mean() < 1e-3
    assert ratios.mean() == pytest.approx(4 * np.pi ** 2 / mu, rel=1e-3)
