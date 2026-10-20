"""Bandwidth estimation from cumulative positive-frequency energy."""

from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .spectral import EnergySpectrum


def estimate_channel_bandwidths(
    spectrum: EnergySpectrum,
    *,
    cumulative_energy_cutoff: float = 0.99,
) -> NDArray[np.float64]:
    """Find the lowest cutoff retaining the requested energy per channel.

    DC (bin zero) is excluded from both total and cumulative energy, without
    modifying the input spectrum. Each channel is normalized independently.

    Args:
        spectrum: One-sided energies of shape (D, K), with increasing
            frequencies of shape (K,) starting at zero.
        cumulative_energy_cutoff: Required cumulative fraction of each
            channel's positive-frequency energy at or below the selected
            frequency. Must lie strictly between zero and one; the default
            0.999 retains at least 99.9% of the fluctuation energy.

    Returns:
        Cutoff frequencies of shape (D,), in the spectrum's frequency units.
        Channels with exactly zero positive-frequency energy return zero.
        No amplitude tolerance is applied to nearly constant channels.

    Raises:
        ValueError: If the cutoff, frequency grid, or energy values or
            shapes are invalid.
    """
    if (
        not np.isfinite(cumulative_energy_cutoff)
        or not 0 < cumulative_energy_cutoff < 1
    ):
        raise ValueError("cumulative_energy_cutoff must be finite and between 0 and 1")
    if np.iscomplexobj(spectrum.frequency) or np.iscomplexobj(spectrum.energy):
        raise ValueError("frequencies and energies must be real")
    frequency = np.asarray(spectrum.frequency, dtype=np.float64)
    energy = np.asarray(spectrum.energy, dtype=np.float64)
    if frequency.ndim != 1 or frequency.size < 2:
        raise ValueError("frequency must have shape (K,) with K >= 2")
    if (
        not np.isfinite(frequency).all()
        or frequency[0] != 0
        or np.any(np.diff(frequency) <= 0)
    ):
        raise ValueError("frequencies must be finite, strictly increasing, and start at zero")
    if energy.ndim != 2 or energy.shape[0] == 0 or energy.shape[1] != frequency.size:
        raise ValueError("energy must have shape (D, K) with D > 0")
    if not np.isfinite(energy).all() or np.any(energy < 0):
        raise ValueError("energy must contain only finite, nonnegative values")

    bandwidths = np.zeros(energy.shape[0], dtype=np.float64)
    for channel, positive_energy in enumerate(energy[:, 1:]):
        scale = positive_energy.max()
        if scale == 0:
            continue
        # Scaling leaves fractions unchanged and avoids overflow in the sum.
        cumulative = np.cumsum(positive_energy / scale)
        cumulative /= cumulative[-1]
        cutoff_index = np.searchsorted(cumulative, cumulative_energy_cutoff, side="left")
        bandwidths[channel] = frequency[cutoff_index + 1]

    return bandwidths


def reduce_bandwidths(
    bandwidths: ArrayLike,
    *,
    reduction: Literal["max", "min", "mean", "median"]
    | Callable[[NDArray[np.float64]], float] = "max",
) -> float:
    """Combine channel bandwidths into a single nonnegative bandwidth.

    Args:
        bandwidths: Nonempty one-dimensional array of finite, nonnegative
            channel bandwidths. Zeros are included in the reduction.
        reduction: "max" (default), "min", "mean", "median", or a callable
            accepting the bandwidth array and returning a scalar. Only
            "max" guarantees a result at least as large as every channel's
            bandwidth among the named reductions. Custom callables receive
            a copy so they cannot modify the input array.

    Returns:
        One finite, nonnegative bandwidth in the input frequency units.

    Raises:
        ValueError: If bandwidths are invalid, the reduction name is unknown,
            or the reduction returns a non-scalar, non-finite, negative, or
            complex value.
    """
    if np.iscomplexobj(bandwidths):
        raise ValueError("bandwidths must be real")
    values = np.array(bandwidths, dtype=np.float64, copy=True)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("bandwidths must be a nonempty one-dimensional array")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("bandwidths must contain only finite, nonnegative values")

    if callable(reduction):
        result = reduction(values)
    else:
        reductions = {
            "max": np.max,
            "min": np.min,
            "mean": np.mean,
            "median": np.median,
        }
        if not isinstance(reduction, str) or reduction not in reductions:
            raise ValueError("reduction must be 'max', 'min', 'mean', 'median', or a callable")
        result = reductions[reduction](values)

    scalar = np.asarray(result)
    if scalar.ndim != 0 or np.iscomplexobj(scalar):
        raise ValueError("reduction must return a real scalar")
    bandwidth = float(scalar)
    if not np.isfinite(bandwidth) or bandwidth < 0:
        raise ValueError("the reduced bandwidth must be finite and nonnegative")
    return bandwidth
