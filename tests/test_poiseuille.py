"""Plane Poiseuille flow: the headline physics validation.

A constant body force drives flow through a no-slip channel.  In steady state
the streamwise velocity must match the analytic parabolic profile

    u(y) = F / (2 rho nu) * y * (H - y)

where ``H`` is the channel height.  We assert that the maximum relative error is
small.
"""

import numpy as np

from lbm import LatticeBoltzmann, channel_walls, poiseuille_profile


def _run_to_steady_state(sim, probe_col, max_steps=20000, tol=1e-9):
    """Step until the mid-channel profile stops changing."""
    prev = None
    for n in range(max_steps):
        _, u = sim.step()
        if n % 500 == 0 and n > 0:
            cur = u[0, probe_col, :].copy()
            if prev is not None:
                denom = np.nanmax(np.abs(cur))
                change = np.nanmax(np.abs(cur - prev)) / denom
                if change < tol:
                    return n
            prev = cur
    return max_steps


def test_poiseuille_matches_analytic_profile():
    nx, ny = 8, 22
    tau = 0.8                 # nu = 0.1
    force = 1e-5

    solid = channel_walls(nx, ny)
    sim = LatticeBoltzmann(nx, ny, tau, solid=solid, force=(force, 0.0))
    _run_to_steady_state(sim, probe_col=nx // 2)

    _, u = sim.macroscopic()
    ux = u[0, nx // 2, :]
    _, ua = poiseuille_profile(ny, force, sim.nu)

    fluid = np.isfinite(ua)
    max_rel_err = np.max(np.abs(ux[fluid] - ua[fluid])) / np.max(ua[fluid])
    assert max_rel_err < 0.01


def test_poiseuille_profile_is_symmetric():
    nx, ny = 8, 22
    sim = LatticeBoltzmann(nx, ny, tau=0.8,
                           solid=channel_walls(nx, ny), force=(1e-5, 0.0))
    _run_to_steady_state(sim, probe_col=nx // 2)
    _, u = sim.macroscopic()
    ux = u[0, nx // 2, 1:-1]          # drop the wall rows
    assert np.allclose(ux, ux[::-1], atol=1e-6)


def test_poiseuille_transverse_velocity_is_negligible():
    nx, ny = 8, 22
    sim = LatticeBoltzmann(nx, ny, tau=0.8,
                           solid=channel_walls(nx, ny), force=(1e-5, 0.0))
    _run_to_steady_state(sim, probe_col=nx // 2)
    _, u = sim.macroscopic()
    assert np.max(np.abs(u[1])) < 1e-9


def test_poiseuille_conserves_mass():
    nx, ny = 8, 22
    sim = LatticeBoltzmann(nx, ny, tau=0.8,
                           solid=channel_walls(nx, ny), force=(1e-5, 0.0))
    mass0 = sim.f.sum()
    sim.run(2000)
    drift = abs(sim.f.sum() - mass0) / mass0
    assert drift < 1e-9
