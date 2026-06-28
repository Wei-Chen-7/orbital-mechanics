"""Ready-made initial conditions used by the tests and the example scripts.

All builders return a :class:`~nbody.system.System` in (or close to) the
center-of-mass rest frame so that linear momentum is zero and plots stay
centered.  Units are stated per-function.
"""

from __future__ import annotations

import numpy as np

from .system import System


def two_body(m1=1.0, m2=1.0, a=1.0, e=0.0, G=1.0):
    """Two bodies on a Kepler orbit, started at perihelion.

    The relative orbit has semi-major axis ``a`` and eccentricity ``e``; the
    bodies are split about a stationary center of mass at the origin, so the
    exact analytic period is ``2*pi*sqrt(a**3 / (G*(m1+m2)))``.

    Returns a 2D :class:`System`.
    """
    M = m1 + m2
    mu = G * M
    r_peri = a * (1.0 - e)
    # vis-viva at perihelion
    v_peri = np.sqrt(mu * (1.0 + e) / (a * (1.0 - e)))

    # Relative state: separation along +x, velocity along +y.
    r_rel = np.array([r_peri, 0.0])
    v_rel = np.array([0.0, v_peri])

    # Split into the COM frame: r1 = -(m2/M) r_rel, r2 = (m1/M) r_rel.
    pos = np.array([-(m2 / M) * r_rel, (m1 / M) * r_rel])
    vel = np.array([-(m2 / M) * v_rel, (m1 / M) * v_rel])
    masses = np.array([m1, m2])
    return System(pos, vel, masses, G=G)


def circular_orbit(central_mass=1.0, orbit_mass=1e-3, radius=1.0, G=1.0):
    """A light body on a circular orbit around a heavy central mass (2D)."""
    return two_body(m1=central_mass, m2=orbit_mass, a=radius, e=0.0, G=G)


def figure_eight():
    """The Chenciner-Montgomery figure-eight choreography (equal masses, 2D).

    Three unit masses with ``G = 1`` chase each other around a single
    figure-eight curve.  The classic initial data (Chenciner & Montgomery
    2000) has period ``T ~ 6.3259``.  Bodies 1 and 2 are mirror images; body 3
    sits at the origin moving so that the total momentum vanishes.
    """
    x, y = 0.97000436, 0.24308753
    vx, vy = 0.93240737, 0.86473146

    pos = np.array([[x, -y], [-x, y], [0.0, 0.0]])
    vel = np.array([[vx / 2, vy / 2], [vx / 2, vy / 2], [-vx, -vy]])
    masses = np.ones(3)
    return System(pos, vel, masses, G=1.0)


# Period of the figure-eight choreography for the initial data above.
FIGURE_EIGHT_PERIOD = 6.3259


def pythagorean():
    """The Pythagorean (Burrau) three-body problem -- a chaotic showcase.

    Masses 3, 4, 5 sit at the vertices of a 3-4-5 right triangle and are
    released from rest with ``G = 1``.  The motion is famously chaotic, which
    makes it ideal for the sensitivity-to-initial-conditions demo.
    """
    pos = np.array([[1.0, 3.0], [-2.0, -1.0], [1.0, -1.0]])
    vel = np.zeros((3, 2))
    masses = np.array([3.0, 4.0, 5.0])
    return System(pos, vel, masses, G=1.0)


# Approximate semi-major axes (AU), eccentricities and masses (solar masses).
# Source: standard planetary fact sheets, rounded.  Good enough for a demo and
# honest about being approximate.
_PLANETS = {
    #          a (AU)   e       mass (Msun)
    "Mercury": (0.387, 0.2056, 1.651e-7),
    "Venus":   (0.723, 0.0068, 2.447e-6),
    "Earth":   (1.000, 0.0167, 3.003e-6),
    "Mars":    (1.524, 0.0934, 3.213e-7),
    "Jupiter": (5.203, 0.0484, 9.543e-4),
}


def solar_system(planets=("Mercury", "Venus", "Earth", "Mars", "Jupiter"), seed=0):
    """Sun plus selected planets in AU / year / solar-mass units (2D).

    With these units ``G = 4*pi**2`` (so a 1 AU circular orbit has period 1
    year and speed ``2*pi``).  Each planet starts at perihelion with the
    correct perihelion speed, placed at a random orbital phase so the orbits do
    not overlap on the plot.  The Sun is given the recoil velocity that zeroes
    the total momentum.
    """
    G = 4.0 * np.pi ** 2
    rng = np.random.default_rng(seed)

    M_sun = 1.0
    positions = [np.zeros(2)]
    velocities = [np.zeros(2)]
    masses = [M_sun]

    for name in planets:
        a, e, m = _PLANETS[name]
        phase = rng.uniform(0, 2 * np.pi)
        r_peri = a * (1.0 - e)
        v_peri = np.sqrt(G * (M_sun + m) * (1.0 + e) / (a * (1.0 - e)))
        rot = np.array([[np.cos(phase), -np.sin(phase)],
                        [np.sin(phase), np.cos(phase)]])
        positions.append(rot @ np.array([r_peri, 0.0]))
        velocities.append(rot @ np.array([0.0, v_peri]))
        masses.append(m)

    positions = np.array(positions)
    velocities = np.array(velocities)
    masses = np.array(masses)

    sys = System(positions, velocities, masses, G=G)
    # Remove net momentum by adjusting the Sun (the heavy body absorbs recoil).
    velocities[0] = -np.sum(masses[1:, None] * velocities[1:], axis=0) / masses[0]
    return System(positions, velocities, masses, G=G)


def _random_directions(rng, n, dim):
    """``n`` isotropic unit vectors in ``dim`` dimensions."""
    v = rng.standard_normal((n, dim))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def plummer_sphere(n=500, total_mass=1.0, radius=1.0, G=1.0, seed=0, dim=2):
    """A Plummer-model star cluster: ``n`` bodies in virial equilibrium.

    Radii are drawn from the Plummer cumulative mass profile and speeds from
    the Plummer velocity distribution, each pointed in an isotropic random
    direction (``dim`` may be 2 or 3); positions and velocities are then
    recentred to the COM frame.  This is the input for the Barnes-Hut cluster
    demos.  Softening of order ``radius / sqrt(n)`` is appropriate when
    integrating it.
    """
    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")
    rng = np.random.default_rng(seed)
    m = total_mass / n

    # Radii from the Plummer cumulative mass profile via inverse transform.
    f = rng.uniform(0, 1, n)
    r = radius / np.sqrt(f ** (-2.0 / 3.0) - 1.0)
    pos = r[:, None] * _random_directions(rng, n, dim)

    # Speeds: sample the Plummer velocity distribution g(q) = q^2 (1-q^2)^3.5.
    q = np.empty(n)
    for k in range(n):
        while True:
            x = rng.uniform(0, 1)
            y = rng.uniform(0, 0.1)
            if y <= x ** 2 * (1.0 - x ** 2) ** 3.5:
                q[k] = x
                break
    v_esc = np.sqrt(2.0) * (1.0 + r ** 2 / radius ** 2) ** -0.25 * np.sqrt(G * total_mass / radius)
    speed = q * v_esc
    vel = speed[:, None] * _random_directions(rng, n, dim)

    masses = np.full(n, m)
    pos -= np.average(pos, axis=0, weights=masses)
    vel -= np.average(vel, axis=0, weights=masses)
    softening = radius / np.sqrt(n)
    return System(pos, vel, masses, G=G, softening=softening)
