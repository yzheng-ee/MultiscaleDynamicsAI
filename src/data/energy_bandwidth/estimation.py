"""Estimate channel AC bandwidths and sampling intervals from trajectories."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..lorenz96.simulate import Trajectory
from .bandwidth import bandwidth_to_period, estimate_channel_bandwidths, reduce_bandwidths
from .sampling import sampling_interval_from_period
from .spectral import compute_spectral_energy


@dataclass(frozen=True)
class BandwidthEstimate:
    """Channel bandwidths with optional reduction, period, and sampling interval.

    channel_bandwidths has shape (D,) and preserves trajectory channel order.
    Both channel_bandwidths and reduced_bandwidth are frequencies in cycles
    per trajectory time unit. bandwidth_period is the reciprocal of the
    reduced bandwidth; sampling_interval divides that period by the requested
    number of intervals per period. Both use the trajectory's time units.
    Fields for stages that were not requested remain None.
    """

    channel_bandwidths: NDArray[np.float64]
    reduced_bandwidth: float | None = None
    bandwidth_period: float | None = None
    sampling_interval: float | None = None


def estimate(
    trajectory: Trajectory,
    *,
    cumulative_energy_cutoff: float = 0.99,
    reduction: Literal["max", "min", "mean", "median"] | None = None,
    samples_per_period: int | None = None,
) -> BandwidthEstimate:
    """Estimate channel bandwidths and optionally a period and sampling interval.

    trajectory contains finite real states of shape (D, N) and finite,
    strictly increasing, uniformly spaced times of shape (N,), with at least
    two observations. The input trajectory is not modified.

    Computes the FFT and one-sided energy, then selects the lowest frequency
    retaining cumulative_energy_cutoff of each channel's AC energy. The cutoff
    must lie strictly between zero and one; the default 0.99 retains 99%.
    DC is excluded, so explicit mean subtraction is unnecessary.

    Always returns channel frequencies of shape (D,) in cycles per trajectory
    time unit, preserving channel order. Channels with exactly zero AC energy
    return zero. Without reduction, all remaining result fields are None.

    Supply reduction as "max", "min", "mean", or "median" to also obtain
    a reduced frequency and its reciprocal period. Frequencies are reduced
    before conversion to a period; this can differ from reducing channel
    periods directly. Zero reduced bandwidth gives an infinite period.

    samples_per_period must be a positive integer and counts intervals per
    period. Including both endpoints gives samples_per_period + 1 observations.
    Providing samples_per_period requires an explicit reduction and also
    computes the sampling interval. Infinite periods cannot define a finite
    sampling interval and are rejected by sampling_interval_from_period.
    """
    if samples_per_period is not None and reduction is None:
        raise ValueError("reduction is required when samples_per_period is supplied")

    energy_spectrum = compute_spectral_energy(trajectory)
    bandwidths = estimate_channel_bandwidths(
        energy_spectrum,
        cumulative_energy_cutoff=cumulative_energy_cutoff,
    )

    bandwidth = None
    period = None
    sampling_interval = None
    if reduction is not None:
        bandwidth = reduce_bandwidths(bandwidths, reduction=reduction)
        period = bandwidth_to_period(bandwidth)
        if samples_per_period is not None:
            sampling_interval = sampling_interval_from_period(period, samples_per_period)

    return BandwidthEstimate(
        channel_bandwidths=bandwidths,
        reduced_bandwidth=bandwidth,
        bandwidth_period=period,
        sampling_interval=sampling_interval,
    )
