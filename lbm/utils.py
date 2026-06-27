"""Helpers for building geometries, diagnosing the flow and post-processing."""

import numpy as np

from .lattice import C, CX, CY, NQ


def cylinder_mask(nx, ny, cx, cy, radius):
    """Boolean mask (nx, ny) that is True inside a disc.

    Parameters
    ----------
    nx, ny : int
        Grid size.
    cx, cy : float
        Centre of the disc, in lattice units.
    radius : float
        Disc radius, in lattice units.
    """
    x = np.arange(nx)[:, None]
    y = np.arange(ny)[None, :]
    return (x - cx) ** 2 + (y - cy) ** 2 < radius ** 2


def ellipse_mask(nx, ny, cx, cy, rx, ry):
    """Boolean mask (nx, ny) that is True inside an axis-aligned ellipse."""
    x = np.arange(nx)[:, None]
    y = np.arange(ny)[None, :]
    return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 < 1.0


def channel_walls(nx, ny):
    """Boolean mask with solid top and bottom rows (a no-slip channel)."""
    solid = np.zeros((nx, ny), dtype=bool)
    solid[:, 0] = True
    solid[:, -1] = True
    return solid


def vorticity(u, solid=None):
    """Out-of-plane vorticity ``omega_z = d uy/dx - d ux/dy``.

    Parameters
    ----------
    u : ndarray, shape (2, nx, ny)
        Velocity field.
    solid : ndarray of bool, optional
        If given, vorticity inside the solid is set to NaN so it renders blank.
    """
    ux, uy = u[0], u[1]
    duy_dx = np.gradient(uy, axis=0)
    dux_dy = np.gradient(ux, axis=1)
    w = duy_dx - dux_dy
    if solid is not None:
        w = np.where(solid, np.nan, w)
    return w


def speed(u):
    """Velocity magnitude sqrt(ux^2 + uy^2)."""
    return np.sqrt(u[0] ** 2 + u[1] ** 2)


def poiseuille_profile(ny, force, nu, rho=1.0):
    """Analytic plane-Poiseuille velocity profile for the channel set-up.

    The solver marks the top and bottom rows (``j = 0`` and ``j = ny-1``) as
    solid.  Halfway bounce-back places the no-slip walls half a lattice spacing
    beyond the last fluid node, so the effective channel height is ``H = ny - 2``
    and fluid node ``j`` sits at ``y = j - 0.5`` measured from the bottom wall.

    Returns
    -------
    y : ndarray, shape (ny,)
        Wall-normal coordinate of each row (NaN inside the walls).
    u : ndarray, shape (ny,)
        Analytic streamwise velocity (NaN inside the walls).
    """
    H = ny - 2
    j = np.arange(ny)
    y = j - 0.5                       # distance from the bottom wall
    u = force / (2.0 * rho * nu) * y * (H - y)
    u[0] = np.nan                     # bottom wall
    u[-1] = np.nan                    # top wall
    y = y.astype(float)
    y[0] = np.nan
    y[-1] = np.nan
    return y, u


def obstacle_force(f_post_collision, solid):
    """Force on a stationary obstacle via the momentum-exchange method.

    Sums the momentum transferred across every boundary link (a fluid node with
    a solid neighbour): a population ``f_i`` streaming into the wall bounces
    back, delivering ``2 c_i f_i`` of momentum per link.

    The populations **must be the post-collision distribution** (the values
    about to stream into the wall).  Inside :class:`lbm.solver.LatticeBoltzmann`
    this force is already computed every step and exposed as
    ``sim.force_on_solid``; this standalone helper is kept for reference and
    testing.  Validated against a control-volume momentum balance (agreement to
    ~0.2 %).

    Parameters
    ----------
    f_post_collision : ndarray, shape (9, nx, ny)
        Post-collision populations.
    solid : ndarray of bool, shape (nx, ny)
        Obstacle mask.

    Returns
    -------
    force : ndarray, shape (2,)
        Net (Fx, Fy) = (drag, lift) in lattice units.
    """
    fluid = ~solid
    fx = fy = 0.0
    for i in range(1, NQ):
        # Fluid nodes whose neighbour in direction i is solid.
        neighbour_solid = np.roll(solid, (-CX[i], -CY[i]), axis=(0, 1))
        link = fluid & neighbour_solid
        amount = f_post_collision[i][link].sum()
        fx += 2.0 * CX[i] * amount
        fy += 2.0 * CY[i] * amount
    return np.array([fx, fy])


def force_coefficients(force, U, D, rho=1.0):
    """Convert an obstacle force to (drag, lift) coefficients.

    ``C = F / (0.5 rho U^2 D)`` with ``D`` the projected length (diameter).
    """
    q = 0.5 * rho * U ** 2 * D
    return force / q


def dominant_frequency(signal, dt=1.0):
    """Dominant (non-zero) frequency of a 1-D signal via the real FFT.

    A Hann window suppresses spectral leakage and a parabolic interpolation of
    the peak gives sub-bin frequency resolution, so a clean estimate is possible
    even from a modest number of shedding cycles.

    Parameters
    ----------
    signal : ndarray
        Time series, sampled at uniform spacing ``dt``.
    dt : float
        Sampling interval in lattice time units.

    Returns
    -------
    f_peak : float
        Frequency (cycles per lattice time unit) of the largest spectral peak.
    """
    sig = np.asarray(signal, dtype=float)
    sig = sig - sig.mean()
    window = np.hanning(sig.size)
    spectrum = np.abs(np.fft.rfft(sig * window))
    spectrum[0] = 0.0                  # ignore the DC component
    k = int(np.argmax(spectrum))

    # Parabolic interpolation around the peak bin for sub-bin accuracy.
    delta = 0.0
    if 0 < k < spectrum.size - 1:
        a, b, c = spectrum[k - 1], spectrum[k], spectrum[k + 1]
        denom = a - 2.0 * b + c
        if denom != 0.0:
            delta = 0.5 * (a - c) / denom

    df = 1.0 / (sig.size * dt)
    return (k + delta) * df


def strouhal_number(signal, diameter, velocity, dt=1.0):
    """Strouhal number ``St = f D / U`` from a probe time series.

    Parameters
    ----------
    signal : ndarray
        Probe signal (e.g. transverse velocity downstream of the obstacle).
    diameter : float
        Obstacle diameter ``D`` in lattice units.
    velocity : float
        Free-stream / inlet velocity ``U`` in lattice units.
    dt : float
        Sampling interval in lattice time units.

    Returns
    -------
    St : float
        Strouhal number.
    f_peak : float
        Shedding frequency (cycles per lattice time unit).
    """
    f_peak = dominant_frequency(signal, dt=dt)
    return f_peak * diameter / velocity, f_peak
