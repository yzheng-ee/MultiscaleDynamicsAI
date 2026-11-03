"""Derive sampling intervals and align slow and fast sampling grids."""

from math import frexp, ldexp

import numpy as np

from .._utils.validation import _validate_finite, _validate_integer, _validate_positive, _validate_real


def sampling_interval_from_period(period: float, samples_per_period: int) -> float:
    """Return period / samples_per_period for a finite positive period.

    samples_per_period must be a positive integer. It counts sampling
    intervals within a period; including both endpoints of one period gives
    samples_per_period + 1 observations.
    """
    _validate_real([("period", period)])
    if np.ndim(period) != 0:
        raise ValueError("period must be a scalar")
    _validate_finite([("period", period)])
    _validate_positive([("period", period)])
    _validate_integer([("samples_per_period", samples_per_period)])
    _validate_positive([("samples_per_period", samples_per_period)])

    sampling_interval = float(period) / int(samples_per_period)
    _validate_positive([("derived sampling interval", sampling_interval)])  # Division can underflow to zero

    return sampling_interval


def align_multiscale_sampling(
    slow_sampling_interval: float,
    fast_sampling_interval: float,
) -> tuple[float, int, float]:
    """Choose a common fine interval that subdivides the slow interval.

    Both supplied intervals must be finite positive scalars in the same
    time units. The inputs are preserved; the returned fine interval uses
    a power-of-two subdivision and is no larger than either input interval,
    within floating-point precision. An exact power-of-two ratio retains
    that factor, while ratios below one use a factor of one.
    Unrepresentable ratios and intervals are rejected.

    Returns (interval_ratio, subdivision_factor, aligned_sampling_interval).
    The factor is the smallest power of two at least the interval ratio,
    with a minimum of one. Every subdivision_factor-th fine-grid sample
    lies on the original slow grid.
    """
    parameters = [("slow_sampling_interval", slow_sampling_interval), ("fast_sampling_interval", fast_sampling_interval)]
    _validate_real(parameters)
    for name, value in parameters:
        if np.ndim(value) != 0:
            raise ValueError(f"{name} must be a scalar")
    _validate_finite(parameters)
    _validate_positive(parameters)

    slow_interval = float(slow_sampling_interval)
    fast_interval = float(fast_sampling_interval)
    interval_ratio = slow_interval / fast_interval
    _validate_finite([("interval ratio", interval_ratio)])  # Dividing finite positive intervals can overflow to infinity
    _validate_positive([("interval ratio", interval_ratio)])  # Dividing positive intervals can underflow to zero

    # Binary decomposition avoids log2 rounding down just above a power of two
    fraction, exponent = frexp(interval_ratio)
    if fraction == 0.5:
        exponent -= 1
    exponent = max(0, exponent)
    subdivision_factor = 2 ** exponent

    # Binary scaling also handles factors too large to represent as a float
    aligned_interval = ldexp(slow_interval, -exponent)
    _validate_positive([("aligned sampling interval", aligned_interval)])  # Binary scaling can underflow to zero

    return interval_ratio, subdivision_factor, aligned_interval
