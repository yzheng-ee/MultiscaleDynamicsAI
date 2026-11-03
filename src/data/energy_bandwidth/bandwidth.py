"""Bandwidth estimation from cumulative positive-frequency energy."""

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._utils.validation import _validate_finite, _validate_nonnegative, _validate_real
from .spectral import EnergySpectrum


def estimate_channel_bandwidths(
    spectrum: EnergySpectrum,
    *,
    cumulative_energy_cutoff: float = 0.99,
) -> NDArray[np.float64]:
    """Find each channel's lowest frequency retaining the requested AC energy.

    spectrum contains one-sided energies of shape (D, K) and strictly
    increasing frequencies of shape (K,), starting at zero. DC (bin zero)
    is excluded from both total and cumulative energy. Each channel is
    normalized independently, without modifying the input spectrum.

    cumulative_energy_cutoff must lie strictly between zero and one. The
    default 0.99 retains at least 99% of each channel's positive-frequency
    energy at or below the selected frequency.

    Returns cutoff frequencies of shape (D,) in the spectrum's frequency
    units. Channels with exactly zero AC energy return zero. Nearly constant
    channels are processed using their computed AC energy, without a tolerance.
    """
    _validate_real([("cumulative_energy_cutoff", cumulative_energy_cutoff)])
    _validate_finite([("cumulative_energy_cutoff", cumulative_energy_cutoff)])
    if not 0 < cumulative_energy_cutoff < 1:
        raise ValueError("cumulative_energy_cutoff must be between 0 and 1")

    # Reject complex inputs before conversion can discard imaginary components
    _validate_real([("frequency", spectrum.frequency), ("energy", spectrum.energy)])
    frequency = np.asarray(spectrum.frequency, dtype=np.float64)
    energy = np.asarray(spectrum.energy, dtype=np.float64)
    if frequency.ndim != 1 or frequency.size < 2:
        raise ValueError("frequency must have shape (K,) with K >= 2")
    _validate_finite([("frequency", frequency)])
    if frequency[0] != 0 or np.any(np.diff(frequency) <= 0):
        raise ValueError("frequencies must be strictly increasing and start at zero")

    # Each channel must supply one energy value per frequency bin
    if energy.ndim != 2 or energy.shape[0] == 0 or energy.shape[1] != frequency.size:
        raise ValueError("energy must have shape (D, K) with D > 0")
    _validate_finite([("energy", energy)])
    _validate_nonnegative([("energy", energy)])

    bandwidths = np.zeros(energy.shape[0], dtype=np.float64)

    # Exclude DC so the cutoff depends on fluctuations rather than the channel mean
    for channel, positive_energy in enumerate(energy[:, 1:]):
        scale = positive_energy.max()
        if scale == 0:
            continue  # Keep zero bandwidth when the channel has no AC energy

        # Scaling leaves fractions unchanged and avoids overflow in the sum
        cumulative = np.cumsum(positive_energy / scale)
        cumulative /= cumulative[-1]

        # Select the first frequency whose cumulative fraction reaches the cutoff
        cutoff_index = np.searchsorted(cumulative, cumulative_energy_cutoff, side="left")
        bandwidths[channel] = frequency[cutoff_index + 1]  # Restore the index offset from excluding DC

    return bandwidths


def reduce_bandwidths(
    bandwidths: ArrayLike,
    *,
    reduction: Literal["max", "min", "mean", "median"] = "median",
) -> float:
    """Combine channel bandwidths into one finite, nonnegative bandwidth.

    bandwidths is a nonempty one-dimensional array of finite, nonnegative
    frequencies. Zero values are included in the reduction.

    reduction accepts "max", "min", "mean", or "median" (the default).
    Only "max" guarantees a result at least as large as every channel's
    bandwidth.

    The result uses the same frequency units as the input bandwidths.
    """
    _validate_real([("bandwidths", bandwidths)])
    values = np.asarray(bandwidths, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("bandwidths must be a nonempty one-dimensional array")
    _validate_finite([("bandwidths", values)])
    _validate_nonnegative([("bandwidths", values)])

    reductions = {
        "max": np.max,
        "min": np.min,
        "mean": np.mean,
        "median": np.median,
    }
    if not isinstance(reduction, str) or reduction not in reductions:
        raise ValueError("reduction must be 'max', 'min', 'mean', or 'median'")

    bandwidth = float(reductions[reduction](values))
    _validate_finite([("reduced bandwidth", bandwidth)])  # Mean and even-length median can overflow despite finite inputs

    return bandwidth


def bandwidth_to_period(bandwidth: float) -> float:
    """Return the reciprocal of a finite, nonnegative bandwidth frequency.

    The period uses the reciprocal of the input frequency units. Zero
    bandwidth represents an unbounded period and returns infinity.
    """
    _validate_real([("bandwidth", bandwidth)])
    if np.ndim(bandwidth) != 0:
        raise ValueError("bandwidth must be a scalar")
    _validate_finite([("bandwidth", bandwidth)])
    _validate_nonnegative([("bandwidth", bandwidth)])

    return 1.0 / float(bandwidth) if bandwidth > 0 else float("inf")
