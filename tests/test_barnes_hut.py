"""Barnes-Hut reproduces the direct sum within its theta-controlled tolerance."""

import numpy as np
import pytest

from nbody import forces, simulate, initial_conditions as ic
from nbody.barnes_hut import accelerations_bh, barnes_hut_accelerations


def _random_cloud(n, seed=0, dim=2):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, dim)), rng.uniform(0.5, 1.5, n)


@pytest.mark.parametrize("dim", [2, 3])
def test_theta_zero_is_exact(dim):
    """With theta = 0 every cell is opened, so the tree == direct sum
    (quadtree in 2D, octree in 3D)."""
    pos, m = _random_cloud(120, seed=1, dim=dim)
    direct = forces.accelerations(pos, m, G=1.0, softening=0.05)
    bh = accelerations_bh(pos, m, G=1.0, softening=0.05, theta=0.0)
    assert np.allclose(bh, direct, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("dim", [2, 3])
def test_force_error_bounded_and_grows_with_theta(dim):
    pos, m = _random_cloud(300, seed=2, dim=dim)
    direct = forces.accelerations(pos, m, G=1.0, softening=0.05)

    def rel_err(theta):
        bh = accelerations_bh(pos, m, G=1.0, softening=0.05, theta=theta)
        per_body = (np.linalg.norm(bh - direct, axis=1)
                    / np.linalg.norm(direct, axis=1))
        return per_body.mean(), per_body.max()

    mean_small, max_small = rel_err(0.3)
    mean_large, max_large = rel_err(0.8)
    # A typical opening angle keeps the *typical* force error at the
    # percent level (a few bodies near cell boundaries can be worse)...
    assert mean_small < 0.02
    # ...and a larger angle is a coarser (worse) approximation overall.
    assert mean_large > mean_small
    assert max_large > max_small


def test_octree_rejects_other_dimensions():
    pos = np.zeros((5, 4))
    m = np.ones(5)
    with pytest.raises(ValueError):
        accelerations_bh(pos, m)


def test_barnes_hut_integration_matches_direct():
    """A short orbit integrated with theta=0 Barnes-Hut matches the direct
    integrator trajectory closely."""
    sys = ic.two_body(m1=1.0, m2=1.0, a=1.0, e=0.3, G=1.0)
    sys.softening = 1e-3  # BH needs softening to be well defined at r->0
    n = 2000
    dt = 5.0 / n
    direct = simulate(sys, dt, n, record_every=20)
    bh = simulate(sys, dt, n, record_every=20,
                  acc_func=barnes_hut_accelerations(theta=0.0))
    assert np.abs(direct.positions - bh.positions).max() < 1e-9


@pytest.mark.parametrize("dim", [2, 3])
def test_barnes_hut_conserves_energy_on_cluster(dim):
    """A softened cluster integrated with Barnes-Hut keeps energy bounded,
    in both 2D (quadtree) and 3D (octree)."""
    sys = ic.plummer_sphere(n=150, seed=5, dim=dim)
    assert sys.n_dim == dim
    traj = simulate(sys, 2e-3, 400, record_every=20,
                    acc_func=barnes_hut_accelerations(theta=0.5))
    assert traj.energy_error().max() < 0.05
