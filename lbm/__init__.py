"""A small, vectorised 2-D Lattice Boltzmann (D2Q9, BGK) fluid solver.

Public API
----------
``LatticeBoltzmann``
    The D2Q9 BGK solver.
``equilibrium``
    Discrete Maxwell-Boltzmann equilibrium populations.
Geometry / post-processing helpers in :mod:`lbm.utils`.
"""

from .lattice import C, W, OPP, CS2, NQ, equilibrium
from .solver import LatticeBoltzmann
from .utils import (
    cylinder_mask,
    ellipse_mask,
    channel_walls,
    vorticity,
    speed,
    poiseuille_profile,
    obstacle_force,
    force_coefficients,
    dominant_frequency,
    strouhal_number,
)

__version__ = "0.1.0"

__all__ = [
    "LatticeBoltzmann",
    "equilibrium",
    "C",
    "W",
    "OPP",
    "CS2",
    "NQ",
    "cylinder_mask",
    "ellipse_mask",
    "channel_walls",
    "vorticity",
    "speed",
    "poiseuille_profile",
    "obstacle_force",
    "force_coefficients",
    "dominant_frequency",
    "strouhal_number",
]
