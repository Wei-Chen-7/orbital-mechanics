"""Tests for the D2Q9 lattice constants and the equilibrium distribution.

These check the moment relations that the stencil must satisfy for the scheme
to recover the Navier-Stokes equations.
"""

import numpy as np

from lbm.lattice import C, W, OPP, CS2, equilibrium


def test_weights_sum_to_one():
    assert np.isclose(W.sum(), 1.0)


def test_first_moment_of_weights_vanishes():
    # sum_i w_i c_i = 0  (no net momentum at rest)
    assert np.allclose((W[:, None] * C).sum(axis=0), 0.0)


def test_second_moment_of_weights_is_isotropic():
    # sum_i w_i c_ia c_ib = cs^2 delta_ab
    second = np.einsum("q,qa,qb->ab", W, C, C)
    assert np.allclose(second, CS2 * np.eye(2))


def test_opposite_directions():
    # The OPP table must point each velocity exactly backwards.
    assert np.array_equal(C[OPP], -C)


def test_equilibrium_at_rest_equals_weights():
    rho = np.ones((3, 4))
    u = np.zeros((2, 3, 4))
    feq = equilibrium(rho, u)
    for i in range(9):
        assert np.allclose(feq[i], W[i])


def test_equilibrium_recovers_density_and_momentum():
    rng = np.random.default_rng(0)
    rho = 1.0 + 0.05 * rng.standard_normal((5, 6))
    u = 0.02 * rng.standard_normal((2, 5, 6))
    feq = equilibrium(rho, u)

    # Zeroth moment -> density.
    assert np.allclose(feq.sum(axis=0), rho)
    # First moment -> momentum rho * u.
    momentum = np.einsum("qd,qxy->dxy", C, feq)
    assert np.allclose(momentum, rho[None] * u)


def test_equilibrium_is_non_negative_for_small_mach():
    rho = np.ones((4, 4))
    u = np.zeros((2, 4, 4))
    u[0] = 0.1
    feq = equilibrium(rho, u)
    assert np.all(feq > 0)
