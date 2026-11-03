"""Downsample uniformly sampled trajectories by exact sample selection."""

import numpy as np

from .._utils.validation import _validate_finite, _validate_integer, _validate_positive, _validate_real
from .simulate import Trajectory


def downsample_trajectory(trajectory: Trajectory, *, stride: int) -> Trajectory:
    """Keep every stride-th sample while preserving both endpoints.

    trajectory contains finite real states of shape (D, N), with D > 0,
    and finite, strictly increasing, uniformly spaced times of shape (N,).
    At least two observations are required. Uniform spacing is checked
    within a relative tolerance of 1e-7.

    stride must be a positive integer dividing N - 1. The returned arrays
    have (N - 1) // stride + 1 samples and preserve every input channel.
    Arrays are copied so the result can be modified independently and does
    not retain the full source arrays. No filtering or interpolation occurs.
    """
    _validate_integer([("stride", stride)])
    _validate_positive([("stride", stride)])

    # Reject complex inputs before float conversion can discard imaginary components
    _validate_real([("trajectory.time", trajectory.time), ("trajectory.state", trajectory.state)])
    time = np.asarray(trajectory.time, dtype=np.float64)
    state = np.asarray(trajectory.state, dtype=np.float64)
    if time.ndim != 1 or time.size < 2:
        raise ValueError("trajectory.time must have shape (N,) with N >= 2")
    if state.ndim != 2 or state.shape[0] == 0 or state.shape[1] != time.size:
        raise ValueError("trajectory.state must have shape (D, N) with D > 0 matching the timestamps")

    # Both endpoints lie on the selected grid only when the stride divides the interval count
    if (time.size - 1) % stride != 0:
        raise ValueError("stride must divide the number of intervals (N - 1) to preserve both endpoints")

    _validate_finite([("trajectory.time", time), ("trajectory.state", state)])
    intervals = np.diff(time)
    if np.any(intervals <= 0):
        raise ValueError("trajectory.time must be strictly increasing")

    sample_dt = float((time[-1] - time[0]) / (time.size - 1))
    _validate_finite([("sampling interval", sample_dt)])  # Finite timestamps can still overflow when computing their span
    _validate_positive([("sampling interval", sample_dt)])
    if not np.allclose(intervals, sample_dt, rtol=1e-7, atol=0):
        raise ValueError("Downsampling requires uniformly spaced timestamps")

    return Trajectory(time=time[::stride].copy(), state=state[:, ::stride].copy())
