"""nbody -- a small, well-tested N-body gravitational simulator.

Point masses under Newtonian gravity (with optional Plummer softening),
advanced by a symplectic velocity-Verlet integrator.  The package is
deliberely small; the interesting part is that every physics claim is backed by
a test (energy/angular-momentum conservation, closed Kepler ellipses, Kepler's
third law, and the figure-eight choreography).

Quick start
-----------
>>> from nbody import initial_conditions as ic, simulate
>>> sys = ic.two_body(a=1.0, e=0.5)
>>> traj = simulate(sys, dt=1e-3, n_steps=10000)
>>> traj.energy_error().max() < 1e-4
True
"""

from .system import System
from .integrators import simulate, Trajectory
from . import forces, diagnostics, initial_conditions, barnes_hut

__all__ = [
    "System",
    "simulate",
    "Trajectory",
    "forces",
    "diagnostics",
    "initial_conditions",
    "barnes_hut",
]

__version__ = "0.1.0"
