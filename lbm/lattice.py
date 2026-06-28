"""D2Q9 lattice definitions for the Lattice Boltzmann method.

The D2Q9 stencil uses nine discrete velocities on a square lattice: one rest
velocity plus eight pointing to the nearest and next-nearest neighbours.  The
weights ``W`` and the lattice speed of sound ``CS2`` are chosen so that the
moments of the discrete equilibrium reproduce the isothermal Navier-Stokes
equations in the low-Mach limit.

Velocity ordering (used everywhere in this package)::

        6   2   5          NW   N   NE
          \\ | /
        3 - 0 - 1          W  rest  E
          / | \\
        7   4   8          SW   S   SE
"""

import numpy as np

# Discrete velocities (cx, cy).  Index 0 is the rest population.
C = np.array(
    [
        [0, 0],    # 0  rest
        [1, 0],    # 1  E
        [0, 1],    # 2  N
        [-1, 0],   # 3  W
        [0, -1],   # 4  S
        [1, 1],    # 5  NE
        [-1, 1],   # 6  NW
        [-1, -1],  # 7  SW
        [1, -1],   # 8  SE
    ],
    dtype=np.int64,
)

CX = C[:, 0]
CY = C[:, 1]

# Lattice weights.  They sum to 1 and make the stencil isotropic to the order
# required by the Navier-Stokes equations.
W = np.array(
    [
        4 / 9,
        1 / 9, 1 / 9, 1 / 9, 1 / 9,
        1 / 36, 1 / 36, 1 / 36, 1 / 36,
    ]
)

# Index of the velocity pointing the opposite way (used for bounce-back walls).
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

# Direction groups by sign of the x-component, used by the inlet / outlet BCs.
RIGHT = np.array([1, 5, 8])     # cx > 0  (point into the domain at the left wall)
LEFT = np.array([3, 6, 7])      # cx < 0  (point into the domain at the right wall)
CENTER_X = np.array([0, 2, 4])  # cx == 0

# Speed of sound squared on the lattice.
CS2 = 1.0 / 3.0

# Number of discrete velocities.
NQ = 9


def equilibrium(rho, u):
    """Second-order discrete Maxwell-Boltzmann equilibrium.

    Parameters
    ----------
    rho : ndarray, shape (nx, ny)
        Density field.
    u : ndarray, shape (2, nx, ny)
        Velocity field (ux, uy).

    Returns
    -------
    feq : ndarray, shape (9, nx, ny)
        Equilibrium populations.
    """
    cu = np.einsum("qd,dxy->qxy", C, u)        # c_i . u
    usqr = u[0] ** 2 + u[1] ** 2               # |u|^2
    feq = W[:, None, None] * rho[None] * (
        1.0 + 3.0 * cu + 4.5 * cu ** 2 - 1.5 * usqr[None]
    )
    return feq
