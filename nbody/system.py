"""The :class:`System` container: positions, velocities, masses + invariants.

A ``System`` is a snapshot of the dynamical state at one instant.  It knows how
to compute its own conserved quantities (energy, linear and angular momentum),
which is what the physics tests lean on.  Integrators take a ``System`` and
return a :class:`~nbody.integrators.Trajectory`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import forces


@dataclass
class System:
    """A set of point masses under mutual gravity.

    Attributes
    ----------
    positions : (N, D) array
    velocities : (N, D) array
    masses : (N,) array
    G : float
        Gravitational constant.  ``G = 1`` is convenient for dimensionless
        tests; ``G = 4*pi**2`` gives units of AU, years and solar masses.
    softening : float
        Plummer softening length (0 for exact point masses).
    """

    positions: np.ndarray
    velocities: np.ndarray
    masses: np.ndarray
    G: float = 1.0
    softening: float = 0.0

    def __post_init__(self):
        self.positions = np.asarray(self.positions, dtype=float)
        self.velocities = np.asarray(self.velocities, dtype=float)
        self.masses = np.asarray(self.masses, dtype=float)
        if self.positions.shape != self.velocities.shape:
            raise ValueError("positions and velocities must have the same shape")
        if self.positions.ndim != 2:
            raise ValueError("positions must be a (N, D) array")
        if self.masses.shape != (self.positions.shape[0],):
            raise ValueError("masses must be a (N,) array matching the bodies")

    # -- structure -------------------------------------------------------
    @property
    def n_bodies(self) -> int:
        return self.positions.shape[0]

    @property
    def n_dim(self) -> int:
        return self.positions.shape[1]

    def copy(self) -> "System":
        return System(
            self.positions.copy(),
            self.velocities.copy(),
            self.masses.copy(),
            self.G,
            self.softening,
        )

    # -- dynamics --------------------------------------------------------
    def accelerations(self) -> np.ndarray:
        return forces.accelerations(self.positions, self.masses, self.G, self.softening)

    # -- conserved quantities -------------------------------------------
    def kinetic_energy(self) -> float:
        return 0.5 * np.sum(self.masses * np.sum(self.velocities ** 2, axis=1))

    def potential_energy(self) -> float:
        return forces.potential_energy(
            self.positions, self.masses, self.G, self.softening
        )

    def total_energy(self) -> float:
        return self.kinetic_energy() + self.potential_energy()

    def linear_momentum(self) -> np.ndarray:
        return np.sum(self.masses[:, None] * self.velocities, axis=0)

    def angular_momentum(self) -> np.ndarray:
        """Total angular momentum about the origin.

        Returns a 3-vector in 3D, or a length-1 array holding the scalar
        ``L_z = sum_i m_i (x_i v_yi - y_i v_xi)`` in 2D.
        """
        p = self.masses[:, None] * self.velocities
        if self.n_dim == 2:
            lz = self.positions[:, 0] * p[:, 1] - self.positions[:, 1] * p[:, 0]
            return np.array([np.sum(lz)])
        return np.sum(np.cross(self.positions, p), axis=0)

    def center_of_mass(self) -> np.ndarray:
        return np.sum(self.masses[:, None] * self.positions, axis=0) / self.masses.sum()

    def com_velocity(self) -> np.ndarray:
        return self.linear_momentum() / self.masses.sum()

    def to_com_frame(self) -> "System":
        """Return a copy shifted into the center-of-mass rest frame."""
        out = self.copy()
        out.positions = out.positions - self.center_of_mass()
        out.velocities = out.velocities - self.com_velocity()
        return out
