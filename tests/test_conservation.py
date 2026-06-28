"""Conservation tests on a fully periodic domain.

With no body force, no obstacle and periodic boundaries, the BGK collision and
streaming conserve total mass and total momentum exactly (up to round-off).
"""

import numpy as np

from lbm import LatticeBoltzmann
from lbm.lattice import C, equilibrium


def _taylor_green_init(sim, u0=0.04):
    """Seed the solver with a divergence-free Taylor-Green-like velocity."""
    nx, ny = sim.nx, sim.ny
    x = np.arange(nx)[:, None]
    y = np.arange(ny)[None, :]
    kx, ky = 2 * np.pi / nx, 2 * np.pi / ny
    u = np.zeros((2, nx, ny))
    u[0] = u0 * np.sin(kx * x) * np.cos(ky * y)
    u[1] = -u0 * np.cos(kx * x) * np.sin(ky * y)
    rho = np.ones((nx, ny))
    sim.f = equilibrium(rho, u)


def test_mass_is_conserved():
    sim = LatticeBoltzmann(32, 32, tau=0.7)
    _taylor_green_init(sim)
    mass0 = sim.f.sum()

    masses = []
    for _ in range(300):
        sim.step()
        masses.append(sim.f.sum())
    masses = np.array(masses)

    drift = np.abs(masses - mass0).max() / mass0
    assert drift < 1e-12


def test_momentum_is_conserved():
    sim = LatticeBoltzmann(32, 32, tau=0.7)
    _taylor_green_init(sim)

    def total_momentum():
        return np.einsum("qd,qxy->d", C, sim.f)

    p0 = total_momentum()
    for _ in range(300):
        sim.step()
    p1 = total_momentum()

    # Momentum starts essentially zero, so compare against the velocity scale.
    assert np.allclose(p1, p0, atol=1e-10)


def test_simulation_stays_finite():
    sim = LatticeBoltzmann(32, 32, tau=0.7)
    _taylor_green_init(sim)
    for _ in range(300):
        sim.step()
    assert np.all(np.isfinite(sim.f))
    rho, _ = sim.macroscopic()
    assert np.all(rho > 0)
