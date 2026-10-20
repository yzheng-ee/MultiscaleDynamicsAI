"""Characteristic-period estimation for uniformly sampled trajectories.

These utilities use the characteristic-timescale implementation provided by
Dysts 0.96. See https://github.com/GilpinLab/dysts.

The ``panda_pilot`` environment pins Dysts 0.95 to match the version resolved
by the Panda revision used in the pilot. This does not conflict with this
module: the ``find_characteristic_timescale`` API and implementation used here
are unchanged between Dysts 0.95 and 0.96. See
https://github.com/abao1999/panda.
"""

from collections.abc import Callable
from typing import Literal

from dysts.utils import find_characteristic_timescale
import numpy as np
from numpy.typing import ArrayLike, NDArray


PeriodReduction = Literal["median", "mean", "min", "max"]


def estimate_channel_periods(
    trajectory: ArrayLike,
    sampling_interval: float,
    *,
    time_axis: int = -1,
) -> NDArray[np.float64]:
    """Estimate the dominant Fourier period of every trajectory channel.

    The calculation uses the characteristic-timescale estimator from Dysts
    0.96. A one-dimensional input is treated as a single channel. For a
    multidimensional input, every non-time dimension is flattened into the
    channel dimension.

    Args:
        trajectory: Uniformly sampled trajectory data.
        sampling_interval: Physical time separating consecutive observations.
        time_axis: Axis containing time. Defaults to the last axis.

    Returns:
        One physical-time period estimate per flattened channel.

    Raises:
        ValueError: If the inputs or estimated periods are invalid.
    """
    values = np.asarray(trajectory, dtype=np.float64)
    if values.ndim == 0:
        raise ValueError("trajectory must contain a time dimension")
    if not np.isfinite(sampling_interval) or sampling_interval <= 0:
        raise ValueError("sampling_interval must be finite and positive")

    try:
        values = np.moveaxis(values, time_axis, -1)
    except np.AxisError as error:
        raise ValueError(f"invalid time_axis {time_axis} for shape {values.shape}") from error

    if values.shape[-1] < 2:
        raise ValueError("trajectory must contain at least two time points")
    if not np.isfinite(values).all():
        raise ValueError("trajectory must contain only finite values")

    channels = values.reshape(-1, values.shape[-1])
    periods = np.asarray(
        [
            float(find_characteristic_timescale(channel)) * sampling_interval
            for channel in channels
        ],
        dtype=np.float64,
    )
    if not np.isfinite(periods).all() or np.any(periods <= 0):
        raise ValueError("Dysts returned a non-finite or non-positive period")
    return periods


def estimate_characteristic_period(
    trajectory: ArrayLike,
    sampling_interval: float,
    *,
    time_axis: int = -1,
    reduction: PeriodReduction | Callable[[NDArray[np.float64]], float] = "median",
) -> float:
    """Estimate one characteristic Fourier period for a trajectory.

    System-specific channel selection should happen before calling this
    function. For example, callers studying a slow-fast system can pass only
    its slow variables rather than allowing the more numerous fast variables
    to determine the aggregate period.

    Args:
        trajectory: Uniformly sampled trajectory data.
        sampling_interval: Physical time separating consecutive observations.
        time_axis: Axis containing time. Defaults to the last axis.
        reduction: Method used to combine channel-level periods. Supported
            names are ``"median"``, ``"mean"``, ``"min"``, and ``"max"``;
            a callable accepting the period array may also be supplied.

    Returns:
        The aggregated characteristic period in physical-time units.
    """
    periods = estimate_channel_periods(
        trajectory,
        sampling_interval,
        time_axis=time_axis,
    )

    return _reduce_periods(periods, reduction)


def _reduce_periods(
    periods: NDArray[np.float64],
    reduction: PeriodReduction | Callable[[NDArray[np.float64]], float],
) -> float:
    """Reduce validated channel periods to one positive finite period."""
    reductions: dict[PeriodReduction, Callable[[NDArray[np.float64]], float]] = {
        "median": np.median,
        "mean": np.mean,
        "min": np.min,
        "max": np.max,
    }
    if callable(reduction):
        period = float(reduction(periods))
    else:
        try:
            period = float(reductions[reduction](periods))
        except KeyError as error:
            choices = ", ".join(repr(name) for name in reductions)
            raise ValueError(f"reduction must be one of {choices} or a callable") from error

    if not np.isfinite(period) or period <= 0:
        raise ValueError("the aggregated period must be finite and positive")
    return period
