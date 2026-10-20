"""Estimate channel AC bandwidths from trajectories."""

import numpy as np
from numpy.typing import NDArray

from ..lorenz96.simulate import Trajectory
from .bandwidth import estimate_channel_bandwidths
from .spectral import compute_fft, compute_spectral_energy


def estimate_trajectory_bandwidths(
    trajectory: Trajectory,
    *,
    cumulative_energy_cutoff: float = 0.99,
) -> NDArray[np.float64]:
    """Return each channel's AC cutoff frequency directly from a trajectory.

    Compute the FFT and one-sided energy, then find the lowest frequency
    retaining the requested cumulative fraction of each channel's AC energy.
    DC is excluded during bandwidth estimation; explicit mean subtraction
    is unnecessary. The input trajectory is not modified.

    Args:
        trajectory: Real-valued states of shape (D, N) with finite, strictly
            increasing, uniformly spaced times of shape (N,), with at least
            two observations.
        cumulative_energy_cutoff: Required cumulative AC energy fraction,
            strictly between zero and one. Defaults to 0.99 (99%).

    Returns:
        Cutoff frequencies of shape (D,), in cycles per trajectory time
        unit, preserving channel order. Exactly zero AC energy gives a zero
        cutoff. These are frequencies, not cumulative energy fractions.
        No channel reduction or conversion to a sampling interval is applied.

    Raises:
        ValueError: If the trajectory or cumulative energy cutoff is invalid.
    """
    spectrum = compute_fft(trajectory)
    energy_spectrum = compute_spectral_energy(spectrum)
    return estimate_channel_bandwidths(
        energy_spectrum,
        cumulative_energy_cutoff=cumulative_energy_cutoff,
    )
