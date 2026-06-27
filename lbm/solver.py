"""A vectorised D2Q9 BGK Lattice Boltzmann solver.

The solver advances a single relaxation time (BGK) lattice Boltzmann equation::

    f_i(x + c_i, t + 1) = f_i(x, t) - omega * (f_i - f_i^eq) + S_i

where ``omega = 1 / tau`` is the relaxation rate, ``f_i^eq`` is the discrete
Maxwell-Boltzmann equilibrium and ``S_i`` is an optional Guo body-force term.

The implementation is fully vectorised: collision is an array expression and
streaming is nine ``np.roll`` shifts (one per discrete velocity).  There are no
per-cell Python loops in the hot path.

Boundary conditions
--------------------
* ``solid`` mask     -- halfway bounce-back, used for the channel walls and the
                        immersed obstacle (no-slip).
* ``inlet_outlet``   -- a Zou/He velocity inlet on the left wall and a
                        zero-gradient (Neumann) outflow on the right wall.  When
                        disabled the x-direction is periodic.
* the y-direction is always periodic (walls, if any, are part of ``solid``).
"""

import numpy as np

from .lattice import (
    C, CX, CY, W, OPP, LEFT, RIGHT, CENTER_X, CS2, NQ, equilibrium,
)


class LatticeBoltzmann:
    """D2Q9 BGK solver on a regular ``nx`` by ``ny`` grid.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions (x is the streamwise direction).
    tau : float
        BGK relaxation time.  The kinematic viscosity is ``nu = cs^2 (tau-0.5)``.
        Must be > 0.5 for a positive viscosity.
    solid : ndarray of bool, shape (nx, ny), optional
        Bounce-back (no-slip) nodes: channel walls and/or an obstacle.
    force : (2,) array-like, optional
        Constant body force per unit volume (Fx, Fy), applied with the Guo
        forcing scheme.  Used to drive Poiseuille flow.
    inlet_velocity : ndarray, shape (2, ny), or (2,) array-like, optional
        Prescribed inlet velocity (ux, uy) on the left wall.  Supplying this
        switches on the inlet/outlet boundary conditions; otherwise x is
        periodic.
    rho0 : float, optional
        Initial (and reference) density.
    """

    def __init__(self, nx, ny, tau, solid=None, force=(0.0, 0.0),
                 inlet_velocity=None, rho0=1.0):
        if tau <= 0.5:
            raise ValueError("tau must be > 0.5 for a positive viscosity")
        self.nx = int(nx)
        self.ny = int(ny)
        self.tau = float(tau)
        self.omega = 1.0 / self.tau
        self.nu = CS2 * (self.tau - 0.5)
        self.rho0 = float(rho0)
        self.time = 0

        self.force = np.asarray(force, dtype=float)
        self.has_force = bool(np.any(self.force != 0.0))

        if solid is None:
            self.solid = np.zeros((self.nx, self.ny), dtype=bool)
        else:
            self.solid = np.asarray(solid, dtype=bool)
            if self.solid.shape != (self.nx, self.ny):
                raise ValueError("solid mask must have shape (nx, ny)")
        self.has_solid = bool(self.solid.any())

        # Inlet / outlet configuration.
        self.inlet_outlet = inlet_velocity is not None
        if self.inlet_outlet:
            inlet_velocity = np.asarray(inlet_velocity, dtype=float)
            if inlet_velocity.shape == (2,):
                inlet_velocity = np.repeat(inlet_velocity[:, None], self.ny, axis=1)
            if inlet_velocity.shape != (2, self.ny):
                raise ValueError("inlet_velocity must have shape (2,) or (2, ny)")
            self.inlet_u = inlet_velocity
        else:
            self.inlet_u = None

        # Initialise populations at equilibrium.
        rho = np.full((self.nx, self.ny), self.rho0)
        u0 = np.zeros((2, self.nx, self.ny))
        if self.inlet_outlet:
            u0[:] = self.inlet_u[:, None, :]
        self.f = equilibrium(rho, u0)

    # ------------------------------------------------------------------ #
    # Macroscopic quantities
    # ------------------------------------------------------------------ #
    def macroscopic(self):
        """Return the density ``rho`` (nx, ny) and velocity ``u`` (2, nx, ny).

        The half-force correction of the Guo scheme is included so that ``u`` is
        the physical fluid velocity.
        """
        rho = self.f.sum(axis=0)
        u = np.einsum("qd,qxy->dxy", C, self.f) / rho
        if self.has_force:
            u += 0.5 * self.force[:, None, None] / rho[None]
        return rho, u

    # ------------------------------------------------------------------ #
    # Guo forcing source term
    # ------------------------------------------------------------------ #
    def _guo_source(self, u):
        """Guo (2002) body-force source term S_i, shape (9, nx, ny)."""
        F = self.force
        cu = np.einsum("qd,dxy->qxy", C, u)            # c_i . u
        # term_d = (c_i - u)/cs^2 + (c_i.u) c_i / cs^4
        term = (
            (C[:, :, None, None] - u[None]) / CS2
            + (cu[:, None] * C[:, :, None, None]) / CS2 ** 2
        )                                              # (9, 2, nx, ny)
        s_dot_f = np.einsum("qdxy,d->qxy", term, F)
        prefactor = 1.0 - 0.5 * self.omega             # (1 - 1/(2 tau))
        return prefactor * W[:, None, None] * s_dot_f

    # ------------------------------------------------------------------ #
    # One time step
    # ------------------------------------------------------------------ #
    def step(self):
        """Advance the simulation by one lattice time step.

        Returns the (rho, u) used for the collision, so callers can probe the
        flow without recomputing the moments.
        """
        f = self.f

        # 1. Outflow: zero-gradient on the left-going populations at the right.
        if self.inlet_outlet:
            f[LEFT, -1, :] = f[LEFT, -2, :]

        # 2. Macroscopic moments (with half-force correction).
        rho = f.sum(axis=0)
        u = np.einsum("qd,qxy->dxy", C, f) / rho
        if self.has_force:
            u += 0.5 * self.force[:, None, None] / rho[None]

        # 3. Inlet: prescribe velocity, recover density from Zou/He.
        if self.inlet_outlet:
            u[:, 0, :] = self.inlet_u
            rho[0, :] = (1.0 / (1.0 - u[0, 0, :])) * (
                f[CENTER_X, 0, :].sum(axis=0) + 2.0 * f[LEFT, 0, :].sum(axis=0)
            )

        # 4. Equilibrium.
        feq = equilibrium(rho, u)

        # 5. Inlet: fill the unknown (right-going) populations.  Zou/He as a
        #    non-equilibrium bounce-back keeps it simple and vectorised.
        if self.inlet_outlet:
            f[RIGHT, 0, :] = (
                feq[RIGHT, 0, :] + f[OPP[RIGHT], 0, :] - feq[OPP[RIGHT], 0, :]
            )

        # 6. BGK collision (+ Guo forcing).
        fout = f - self.omega * (f - feq)
        if self.has_force:
            fout += self._guo_source(u)

        # 7. Bounce-back on the solid nodes (no-slip walls / obstacle).
        if self.has_solid:
            for i in range(NQ):
                fout[i, self.solid] = f[OPP[i], self.solid]

        # 8. Streaming: shift each population along its velocity.
        for i in range(NQ):
            fout[i] = np.roll(fout[i], (CX[i], CY[i]), axis=(0, 1))

        self.f = fout
        self.time += 1
        return rho, u

    def run(self, steps):
        """Run ``steps`` time steps and return the final (rho, u)."""
        rho = u = None
        for _ in range(steps):
            rho, u = self.step()
        if rho is None:
            rho, u = self.macroscopic()
        return rho, u
