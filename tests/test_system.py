"""The System container computes its conserved quantities correctly."""

import numpy as np
import pytest

from nbody import System


def test_kinetic_energy():
    pos = np.zeros((2, 2))
    vel = np.array([[3.0, 0.0], [0.0, 4.0]])
    m = np.array([2.0, 0.5])
    sys = System(pos, vel, m)
    # 0.5*(2*9 + 0.5*16) = 0.5*(18 + 8) = 13
    assert sys.kinetic_energy() == pytest.approx(13.0)


def test_angular_momentum_single_circular():
    # One unit mass at radius 2 moving tangentially at speed 3: L_z = m r v = 6.
    pos = np.array([[2.0, 0.0]])
    vel = np.array([[0.0, 3.0]])
    m = np.array([1.0])
    sys = System(pos, vel, m)
    assert sys.angular_momentum()[0] == pytest.approx(6.0)


def test_linear_momentum_and_com_velocity():
    pos = np.zeros((2, 2))
    vel = np.array([[1.0, 0.0], [-1.0, 2.0]])
    m = np.array([1.0, 3.0])
    sys = System(pos, vel, m)
    assert sys.linear_momentum() == pytest.approx([-2.0, 6.0])
    assert sys.com_velocity() == pytest.approx([-0.5, 1.5])


def test_com_frame_zeroes_momentum():
    rng = np.random.default_rng(3)
    sys = System(rng.normal(size=(6, 3)), rng.normal(size=(6, 3)),
                 rng.uniform(0.5, 2, 6))
    com = sys.to_com_frame()
    assert np.allclose(com.linear_momentum(), 0.0, atol=1e-12)
    assert np.allclose(com.center_of_mass(), 0.0, atol=1e-12)
    # Total energy is frame-dependent, but internal structure is unchanged:
    # potential energy (a function of separations only) must be identical.
    assert com.potential_energy() == pytest.approx(sys.potential_energy())


def test_shape_validation():
    with pytest.raises(ValueError):
        System(np.zeros((3, 2)), np.zeros((2, 2)), np.ones(3))
    with pytest.raises(ValueError):
        System(np.zeros((3, 2)), np.zeros((3, 2)), np.ones(2))
