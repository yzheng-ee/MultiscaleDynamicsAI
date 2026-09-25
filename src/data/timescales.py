"""Characteristic-timescale estimation and period-normalized sampling.

These utilities use the characteristic-timescale implementation provided by
Dysts 0.96. See https://github.com/GilpinLab/dysts.
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


def make_period_normalized_times(
    period: float,
    *,
    num_periods: float = 40,  # Panda effectively uses 4096 / (4096 // 40).
    num_points: int = 4096,
) -> NDArray[np.float64]:
    """Return uniformly spaced observation times over a number of periods.

    The returned grid includes both endpoints, so ``num_points`` observations
    contain ``num_points - 1`` sampling intervals.

    Args:
        period: Characteristic period in physical-time units.
        num_periods: Number of characteristic periods spanned by the grid.
        num_points: Number of uniformly spaced observations, including both
            endpoints.

    Returns:
        A one-dimensional time grid from zero through
        ``num_periods * period``.

    Raises:
        TypeError: If ``num_points`` is not an integer.
        ValueError: If ``period`` or ``num_periods`` is not finite and
            positive, or if ``num_points`` is less than two.
    """
    if not np.isfinite(period) or period <= 0:
        raise ValueError("period must be finite and positive")
    if not np.isfinite(num_periods) or num_periods <= 0:
        raise ValueError("num_periods must be finite and positive")
    if isinstance(num_points, bool) or not isinstance(num_points, (int, np.integer)):
        raise TypeError("num_points must be an integer")
    if num_points < 2:
        raise ValueError("num_points must be at least two")

    return np.linspace(
        0.0,
        float(num_periods) * period,
        int(num_points),
        dtype=np.float64,
    )


def estimate_sampling_interval(
    trajectory: ArrayLike,
    sampling_interval: float,
    *,
    time_axis: int = -1,
    reduction: PeriodReduction | Callable[[NDArray[np.float64]], float] = "median",
    num_periods: float = 40,  # Panda effectively uses 4096 / (4096 // 40).
    num_points: int = 4096,
) -> float:
    """Estimate a period-normalized target sampling interval.

    Callers should select the channels that define the timescale of interest
    before calling this function. For a multiscale slow-fast system, pass only
    the slow-variable channels when the target sampling interval should follow
    the slow dynamics.

    Args:
        trajectory: Densely and uniformly sampled data containing only the
            channels relevant to the target timescale.
        sampling_interval: Physical time between observations in the dense
            input trajectory.
        time_axis: Axis containing time. Defaults to the last axis.
        reduction: Method used to combine channel-level period estimates.
        num_periods: Number of estimated characteristic periods that the
            target trajectory should span.
        num_points: Number of uniformly spaced target observations.

    Returns:
        The physical time between consecutive target observations.
    """
    period = estimate_characteristic_period(
        trajectory,
        sampling_interval,
        time_axis=time_axis,
        reduction=reduction,
    )
    times = make_period_normalized_times(
        period,
        num_periods=num_periods,
        num_points=num_points,
    )
    return float(times[1] - times[0])
