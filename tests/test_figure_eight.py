"""The figure-eight three-body choreography: periodicity, conservation, shape."""

import numpy as np

from nbody import initial_conditions as ic, simulate


def test_figure_eight_is_periodic():
    """After one period the three bodies return to their initial state -- the
    defining property of the choreography, and a strong check that we have the
    real figure-eight and not some nearby orbit."""
    sys = ic.figure_eight()
    n = 9000
    dt = ic.FIGURE_EIGHT_PERIOD / n
    traj = simulate(sys, dt, n, record_every=1)

    assert np.abs(traj.positions[-1] - traj.positions[0]).max() < 1e-3
    assert np.abs(traj.velocities[-1] - traj.velocities[0]).max() < 1e-3


def test_figure_eight_conserves_energy_and_momentum():
    sys = ic.figure_eight()
    traj = simulate(sys, ic.FIGURE_EIGHT_PERIOD / 9000, 9000, record_every=10)
    assert traj.energy_error().max() < 1e-5
    # COM stays put: total linear momentum is zero throughout.
    assert np.abs(traj.linear_momentum()).max() < 1e-10


def test_figure_eight_choreography_single_curve():
    """All three bodies follow one curve, phase-shifted by T/3:
    body0(t) == body1(t + T/3) == body2(t + 2 T/3)."""
    sys = ic.figure_eight()
    n = 9000  # divisible by 3
    traj = simulate(sys, ic.FIGURE_EIGHT_PERIOD / n, n, record_every=1)
    pos = traj.positions
    shift = n // 3
    idx = np.arange(n + 1 - 2 * shift)

    assert np.abs(pos[idx, 0] - pos[idx + shift, 1]).max() < 1e-2
    assert np.abs(pos[idx, 0] - pos[idx + 2 * shift, 2]).max() < 1e-2
