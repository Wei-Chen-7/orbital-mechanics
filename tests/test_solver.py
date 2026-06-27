"""Solver-level sanity checks: API behaviour and the obstacle flow."""

import numpy as np
import pytest

from lbm import LatticeBoltzmann, cylinder_mask
from lbm.lattice import equilibrium


def test_tau_must_be_above_half():
    with pytest.raises(ValueError):
        LatticeBoltzmann(4, 4, tau=0.5)


def test_viscosity_matches_relaxation_time():
    sim = LatticeBoltzmann(4, 4, tau=0.9)
    assert np.isclose(sim.nu, (0.9 - 0.5) / 3.0)


def test_macroscopic_inverts_equilibrium():
    # Building f from equilibrium(rho, u) and reading the moments back must
    # return the same rho and u (no body force here).
    sim = LatticeBoltzmann(6, 5, tau=0.7)
    rng = np.random.default_rng(1)
    rho = 1.0 + 0.05 * rng.standard_normal((6, 5))
    u = 0.02 * rng.standard_normal((2, 6, 5))
    sim.f = equilibrium(rho, u)
    rho_out, u_out = sim.macroscopic()
    assert np.allclose(rho_out, rho)
    assert np.allclose(u_out, u)


def test_cylinder_flow_runs_and_stays_finite():
    nx, ny = 80, 40
    U = 0.05
    D = 12.0
    Re = 80.0
    nu = U * D / Re
    tau = 3 * nu + 0.5

    solid = cylinder_mask(nx, ny, nx // 4, ny // 2, D / 2)
    sim = LatticeBoltzmann(nx, ny, tau, solid=solid, inlet_velocity=(U, 0.0))

    for _ in range(200):
        sim.step()

    rho, u = sim.macroscopic()
    assert np.all(np.isfinite(sim.f))
    assert np.all(rho > 0)
    # Flow should remain subsonic on the lattice (low Mach number).
    assert np.max(np.abs(u)) < 0.3
    # The bulk flow moves downstream.
    assert u[0].mean() > 0
    # A momentum deficit (wake) sits directly behind the obstacle: the
    # centreline is slower than the free stream above/below it.
    x_wake = nx // 4 + int(D)
    centreline = u[0, x_wake, ny // 2]
    free_stream = u[0, x_wake, 2]
    assert centreline < free_stream


def test_inlet_velocity_shape_validation():
    with pytest.raises(ValueError):
        LatticeBoltzmann(8, 6, tau=0.7, inlet_velocity=np.zeros((2, 5)))


def test_drag_coefficient_is_physically_reasonable():
    # Steady (sub-critical) flow past a cylinder at Re = 20.  The drag
    # coefficient should land near the textbook value (~2, raised somewhat by
    # channel confinement) and the lift should be ~0 by symmetry.
    from lbm import force_coefficients

    nx, ny = 160, 60
    U, D, Re = 0.05, 10.0, 20.0
    nu = U * D / Re
    tau = 3 * nu + 0.5
    solid = cylinder_mask(nx, ny, nx // 4, ny // 2, D / 2)
    sim = LatticeBoltzmann(nx, ny, tau, solid=solid, inlet_velocity=(U, 0.0))

    sim.run(6000)
    cd, cl = force_coefficients(sim.force_on_solid, U, D)

    assert 1.5 < cd < 4.0          # textbook ~2, plus confinement
    assert abs(cl) < 0.05          # symmetric flow -> negligible lift
    assert sim.force_on_solid[0] > 0   # drag points downstream
