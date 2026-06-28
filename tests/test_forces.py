"""Physics of the force law itself: direction, magnitude, softening, energy."""

import numpy as np
import pytest

from nbody import forces


def test_two_body_force_direction_and_magnitude():
    # Unit masses at (-1, 0) and (+1, 0); separation 2, so |a| = G m / d^2 = 1/4.
    pos = np.array([[-1.0, 0.0], [1.0, 0.0]])
    m = np.array([1.0, 1.0])
    a = forces.accelerations(pos, m, G=1.0, softening=0.0)

    assert a[0] == pytest.approx([0.25, 0.0])  # body 0 pulled toward +x
    assert a[1] == pytest.approx([-0.25, 0.0])  # body 1 pulled toward -x


def test_newtons_third_law():
    # Sum of m_i * a_i must vanish: internal forces cancel exactly.
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(7, 3))
    m = rng.uniform(0.5, 2.0, 7)
    a = forces.accelerations(pos, m, G=1.3, softening=0.1)
    net = np.sum(m[:, None] * a, axis=0)
    assert np.allclose(net, 0.0, atol=1e-12)


def test_softening_reduces_force():
    pos = np.array([[0.0, 0.0], [1.0, 0.0]])
    m = np.array([1.0, 1.0])
    hard = np.linalg.norm(forces.accelerations(pos, m, softening=0.0)[0])
    soft = np.linalg.norm(forces.accelerations(pos, m, softening=0.5)[0])
    assert soft < hard


def test_force_scales_linearly_with_G():
    rng = np.random.default_rng(2)
    pos = rng.normal(size=(5, 2))
    m = rng.uniform(0.5, 2.0, 5)
    a1 = forces.accelerations(pos, m, G=1.0)
    a3 = forces.accelerations(pos, m, G=3.0)
    assert np.allclose(a3, 3.0 * a1)


def test_inverse_square_falloff():
    # Acceleration from a fixed mass should scale as 1/r^2.
    m = np.array([1.0, 1e-30])  # second body is a massless probe
    for r in [2.0, 4.0, 8.0]:
        pos = np.array([[0.0, 0.0], [r, 0.0]])
        a = np.linalg.norm(forces.accelerations(pos, m)[1])
        assert a == pytest.approx(1.0 / r ** 2, rel=1e-12)


def test_potential_energy_value_and_sign():
    pos = np.array([[0.0, 0.0], [2.0, 0.0]])
    m = np.array([1.0, 3.0])
    # U = -G m0 m1 / d = -1 * 1 * 3 / 2
    assert forces.potential_energy(pos, m, G=1.0) == pytest.approx(-1.5)


def test_potential_energy_pairwise_additive():
    # Three far-apart pairs: total U is the sum of the three pair potentials.
    pos = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    m = np.array([1.0, 1.0, 1.0])
    d01, d02, d12 = 10.0, 10.0, np.sqrt(200.0)
    expected = -(1 / d01 + 1 / d02 + 1 / d12)
    assert forces.potential_energy(pos, m) == pytest.approx(expected)
