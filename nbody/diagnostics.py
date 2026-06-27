"""Orbital diagnostics for the two-body / Kepler tests.

These functions turn a relative orbit (one body about another) into the
classical orbital elements so the tests can assert real physics:

* :func:`orbital_elements` -- semi-major axis ``a`` and eccentricity ``e`` from
  the vis-viva and angular-momentum relations.
* :func:`eccentricity_vector` -- the Laplace-Runge-Lenz vector, normalised.  It
  points from the focus to perihelion; its *constancy* is exactly the
  statement that the orbit is a closed, non-precessing ellipse.  This is a far
  sharper test of "closed Kepler ellipse" than eyeballing a plot.
* :func:`kepler_period` -- the analytic period ``2 pi sqrt(a^3 / mu)``.
* :func:`measure_period` -- the period measured from a trajectory by detecting
  when the relative orbit completes one full revolution.
"""

from __future__ import annotations

import numpy as np


def relative_orbit(trajectory, i=0, j=1):
    """Relative position/velocity of body ``j`` with respect to body ``i``.

    Returns ``(r_rel, v_rel)`` each of shape ``(T, D)``.
    """
    r = trajectory.positions[:, j, :] - trajectory.positions[:, i, :]
    v = trajectory.velocities[:, j, :] - trajectory.velocities[:, i, :]
    return r, v


def eccentricity_vector(r, v, mu):
    """Laplace-Runge-Lenz (eccentricity) vector for relative motion.

        e_vec = (v x L) / mu - r_hat ,   L = r x v

    Works on a single state ``(D,)`` or a stack of states ``(T, D)`` and for
    both 2D and 3D inputs (2D vectors are promoted to 3D internally).
    """
    r = np.atleast_2d(r).astype(float)
    v = np.atleast_2d(v).astype(float)
    if r.shape[1] == 2:  # embed 2D in 3D so cross products are vectors
        r = np.column_stack([r, np.zeros(len(r))])
        v = np.column_stack([v, np.zeros(len(v))])

    L = np.cross(r, v)  # specific angular momentum (per reduced mass)
    rnorm = np.linalg.norm(r, axis=1, keepdims=True)
    evec = np.cross(v, L) / mu - r / rnorm
    return evec


def orbital_elements(r, v, mu):
    """Semi-major axis and scalar eccentricity from a single relative state.

    Parameters
    ----------
    r, v : (D,) arrays
        Relative position and velocity.
    mu : float
        Standard gravitational parameter ``G * (m_i + m_j)``.

    Returns
    -------
    (a, e) : semi-major axis and eccentricity.
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    rnorm = np.linalg.norm(r)
    speed2 = np.dot(v, v)
    energy = 0.5 * speed2 - mu / rnorm  # specific orbital energy
    a = -mu / (2.0 * energy)
    e = np.linalg.norm(eccentricity_vector(r, v, mu)[0])
    return a, e


def kepler_period(a, mu):
    """Analytic orbital period from Kepler's third law: ``2 pi sqrt(a^3/mu)``."""
    return 2.0 * np.pi * np.sqrt(a ** 3 / mu)


def measure_period(trajectory, i=0, j=1):
    """Measure the orbital period of body ``j`` about body ``i`` from data.

    The relative angle is unwrapped and the first time it advances by ``2 pi``
    (relative to the start) is found by linear interpolation.  Returns ``nan``
    if the trajectory does not cover a full revolution.
    """
    r, _ = relative_orbit(trajectory, i, j)
    angle = np.unwrap(np.arctan2(r[:, 1], r[:, 0]))
    angle = angle - angle[0]
    target = np.sign(angle[-1]) * 2.0 * np.pi  # respect orbit direction
    t = trajectory.times

    advanced = np.abs(angle) >= 2.0 * np.pi
    if not np.any(advanced):
        return float("nan")
    k = np.argmax(advanced)  # first index past one full turn
    if k == 0:
        return float("nan")
    # Linear interpolation for the exact crossing time.
    a0, a1 = angle[k - 1], angle[k]
    t0, t1 = t[k - 1], t[k]
    frac = (target - a0) / (a1 - a0)
    return t0 + frac * (t1 - t0)
