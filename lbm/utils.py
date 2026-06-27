"""Helpers for building geometries, diagnosing the flow and post-processing."""

import numpy as np


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


def dominant_frequency(signal, dt=1.0):
    """Dominant (non-zero) frequency of a 1-D signal via the real FFT.

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
    freqs = np.fft.rfftfreq(sig.size, d=dt)
    amp = np.abs(np.fft.rfft(sig))
    amp[0] = 0.0                       # ignore the DC component
    return freqs[np.argmax(amp)]


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
