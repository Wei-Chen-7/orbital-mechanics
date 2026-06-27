"""Newtonian gravitational forces with Plummer softening.

The fundamental quantity is the acceleration felt by each body,

    a_i = G * sum_{j != i} m_j (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)

The ``eps`` term is *Plummer softening*: it caps the force at small
separations so that close encounters do not blow up the integrator.  With
``eps = 0`` this is the exact Newtonian point-mass acceleration, which is what
the two-body / Kepler tests use.

Everything here is a pure function of NumPy arrays so the integrators and the
Barnes-Hut tree can share the same definitions of "energy" and "acceleration".
"""

from __future__ import annotations

import numpy as np


def accelerations(positions, masses, G=1.0, softening=0.0):
    """Direct (O(N^2)) gravitational acceleration of every body.

    Parameters
    ----------
    positions : (N, D) array
        Cartesian positions; ``D`` may be 2 or 3.
    masses : (N,) array
        Body masses.
    G : float
        Gravitational constant (sets the unit system).
    softening : float
        Plummer softening length ``eps``.

    Returns
    -------
    (N, D) array of accelerations.
    """
    positions = np.asarray(positions, dtype=float)
    masses = np.asarray(masses, dtype=float)

    # diff[i, j] = r_j - r_i, shape (N, N, D)
    diff = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
    dist2 = np.sum(diff * diff, axis=-1)
    dist2 += softening * softening

    # The diagonal is the self-interaction.  Put a harmless 1.0 there before
    # taking the (-3/2) power to avoid 0**-1.5 warnings, then zero it out so a
    # body exerts no force on itself.
    np.fill_diagonal(dist2, 1.0)
    inv_dist3 = dist2 ** -1.5
    np.fill_diagonal(inv_dist3, 0.0)

    # a_i = G * sum_j m_j * diff[i, j] * inv_dist3[i, j]
    factor = inv_dist3 * masses[np.newaxis, :]  # (N, N)
    acc = G * np.einsum("ij,ijk->ik", factor, diff)
    return acc


def potential_energy(positions, masses, G=1.0, softening=0.0):
    """Total gravitational potential energy.

        U = -G * sum_{i<j} m_i m_j / sqrt(|r_i - r_j|^2 + eps^2)

    Uses the same softening as :func:`accelerations` so that energy is a true
    invariant of the softened dynamics (and is exactly conserved by the
    symplectic integrator up to integration error).
    """
    positions = np.asarray(positions, dtype=float)
    masses = np.asarray(masses, dtype=float)
    n = len(masses)
    if n < 2:
        return 0.0

    diff = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
    dist2 = np.sum(diff * diff, axis=-1) + softening * softening
    iu = np.triu_indices(n, k=1)
    inv_dist = dist2[iu] ** -0.5
    mass_pairs = masses[iu[0]] * masses[iu[1]]
    return -G * np.sum(mass_pairs * inv_dist)
