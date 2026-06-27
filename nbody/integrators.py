"""Time integration: a symplectic velocity-Verlet and a reference RK4.

Velocity-Verlet (a.k.a. leapfrog in kick-drift-kick form) is the work-horse.
It is *symplectic* and time-reversible, so for a bound orbit the energy error
stays bounded and oscillatory forever instead of drifting.  RK4 is included
only as a foil: it is more accurate per step but, being non-symplectic, it
slowly bleeds energy on long runs.  ``examples/energy_comparison.py`` and
``tests/test_energy_conservation.py`` make that contrast quantitative.

``simulate`` returns a :class:`Trajectory` that recomputes invariants from the
stored states on demand, so the diagnostics always reflect the actual sampled
configurations rather than something cached during stepping.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import forces
from .system import System


@dataclass
class Trajectory:
    """Recorded history of a simulation.

    Attributes
    ----------
    times : (T,) array
    positions : (T, N, D) array
    velocities : (T, N, D) array
    masses : (N,) array
    G, softening : float
    method : str
        Name of the integrator that produced it.
    """

    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    masses: np.ndarray
    G: float = 1.0
    softening: float = 0.0
    method: str = "verlet"

    @property
    def n_frames(self) -> int:
        return self.positions.shape[0]

    def state_at(self, index: int) -> System:
        """Reconstruct the :class:`System` at frame ``index``."""
        return System(
            self.positions[index],
            self.velocities[index],
            self.masses,
            self.G,
            self.softening,
        )

    # -- diagnostics computed across all frames -------------------------
    def kinetic_energy(self) -> np.ndarray:
        v2 = np.sum(self.velocities ** 2, axis=2)  # (T, N)
        return 0.5 * np.sum(self.masses[None, :] * v2, axis=1)

    def potential_energy(self) -> np.ndarray:
        return np.array(
            [
                forces.potential_energy(p, self.masses, self.G, self.softening)
                for p in self.positions
            ]
        )

    def total_energy(self) -> np.ndarray:
        return self.kinetic_energy() + self.potential_energy()

    def energy_error(self) -> np.ndarray:
        """Relative energy error ``|E(t) - E(0)| / |E(0)|`` at each frame."""
        e = self.total_energy()
        return np.abs((e - e[0]) / e[0])

    def linear_momentum(self) -> np.ndarray:
        return np.einsum("tnd,n->td", self.velocities, self.masses)

    def angular_momentum(self) -> np.ndarray:
        """Total angular momentum at each frame, shape ``(T, 3)`` or ``(T, 1)``."""
        p = self.velocities * self.masses[None, :, None]
        x = self.positions
        if x.shape[2] == 2:  # 2D -> scalar L_z per frame
            lz = x[:, :, 0] * p[:, :, 1] - x[:, :, 1] * p[:, :, 0]
            return np.sum(lz, axis=1)[:, None]
        return np.sum(np.cross(x, p), axis=1)


# ---------------------------------------------------------------------------
# Integrators
# ---------------------------------------------------------------------------
def _default_acc(acc_func):
    if acc_func is None:
        return forces.accelerations
    return acc_func


def simulate(
    system: System,
    dt: float,
    n_steps: int,
    method: str = "verlet",
    record_every: int = 1,
    acc_func=None,
) -> Trajectory:
    """Integrate ``system`` for ``n_steps`` steps of size ``dt``.

    Parameters
    ----------
    method : {"verlet", "leapfrog", "rk4"}
        ``"verlet"`` and ``"leapfrog"`` are the same symplectic scheme.
    record_every : int
        Store every ``record_every``-th step (plus the initial state) to keep
        long runs memory-light.
    acc_func : callable, optional
        ``acc_func(positions, masses, G, softening) -> (N, D)``.  Defaults to
        the direct O(N^2) sum; pass a Barnes-Hut evaluator here for large N.
    """
    method = method.lower()
    if method in ("verlet", "leapfrog", "velocity_verlet"):
        stepper = _step_velocity_verlet
    elif method == "rk4":
        stepper = _step_rk4
    else:
        raise ValueError(f"unknown method {method!r}")

    acc = _default_acc(acc_func)
    G, soft = system.G, system.softening

    x = system.positions.copy()
    v = system.velocities.copy()
    m = system.masses
    a = acc(x, m, G, soft)

    n_records = n_steps // record_every + 1
    times = np.empty(n_records)
    xs = np.empty((n_records, *x.shape))
    vs = np.empty((n_records, *v.shape))

    times[0] = 0.0
    xs[0] = x
    vs[0] = v
    rec = 1

    for step in range(1, n_steps + 1):
        x, v, a = stepper(x, v, a, m, dt, G, soft, acc)
        if step % record_every == 0:
            times[rec] = step * dt
            xs[rec] = x
            vs[rec] = v
            rec += 1

    return Trajectory(
        times[:rec], xs[:rec], vs[:rec], m.copy(), G, soft, method
    )


def _step_velocity_verlet(x, v, a, m, dt, G, soft, acc):
    """One kick-drift-kick velocity-Verlet step (symplectic, 2nd order)."""
    v_half = v + 0.5 * dt * a
    x_new = x + dt * v_half
    a_new = acc(x_new, m, G, soft)
    v_new = v_half + 0.5 * dt * a_new
    return x_new, v_new, a_new


def _step_rk4(x, v, a, m, dt, G, soft, acc):
    """One classical RK4 step on (x, v).  Non-symplectic, 4th order."""
    # State derivative: dx/dt = v, dv/dt = a(x).
    k1x = v
    k1v = a

    k2x = v + 0.5 * dt * k1v
    k2v = acc(x + 0.5 * dt * k1x, m, G, soft)

    k3x = v + 0.5 * dt * k2v
    k3v = acc(x + 0.5 * dt * k2x, m, G, soft)

    k4x = v + dt * k3v
    k4v = acc(x + dt * k3x, m, G, soft)

    x_new = x + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
    v_new = v + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
    a_new = acc(x_new, m, G, soft)
    return x_new, v_new, a_new
