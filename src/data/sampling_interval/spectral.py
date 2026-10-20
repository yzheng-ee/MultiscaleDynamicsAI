"""Spectral utilities for sampling-interval estimation."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..lorenz96.simulate import Trajectory


@dataclass(frozen=True)
class FrequencySpectrum:
    """One-sided real-input FFT with its original sampling metadata.

    frequency has shape (K,) in cycles per unit of time; coefficients has
    shape (D, K), where K = n_samples // 2 + 1. Coefficients use NumPy's
    unscaled forward transform and refer to sample indices relative to the
    trajectory's first timestamp. n_samples retains the original odd/even
    parity needed for one-sided energy weighting and inverse transforms.
    """

    frequency: NDArray[np.float64]
    coefficients: NDArray[np.complex128]
    n_samples: int
    sampling_interval: float


@dataclass(frozen=True)
class EnergySpectrum:
    """One-sided spectral energies, including DC.

    frequency has shape (K,) and energy has shape (D, K). Energies are
    unnormalized: their sum per channel equals n_samples times the sum of
    squared values of that channel's original time-domain samples. They are neither
    normalized energy fractions nor a power spectral density.
    """

    frequency: NDArray[np.float64]
    energy: NDArray[np.float64]
    n_samples: int
    sampling_interval: float


def subtract_mean(trajectory: Trajectory) -> Trajectory:
    """Subtract each channel's temporal mean from one trajectory.

    Args:
        trajectory: Real-valued states of shape (D, N) and increasing times
            of shape (N,). Times need not be uniformly spaced.

    Returns:
        A new Trajectory with copied timestamps and each channel centered
        around zero. The input is not modified.

    Raises:
        ValueError: If times or states have invalid shapes, are empty,
            complex, or non-finite, or times are not strictly increasing.
    """
    time, state = _validate_trajectory(trajectory)
    return Trajectory(time=time.copy(), state=state - state.mean(axis=-1, keepdims=True))


def compute_fft(trajectory: Trajectory) -> FrequencySpectrum:
    """Compute real-input FFT coefficients and their physical frequencies.

    Args:
        trajectory: Real-valued states of shape (D, N) and uniformly spaced,
            increasing times of shape (N,). Mean subtraction, if desired,
            must be performed separately.

    Returns:
        A FrequencySpectrum containing the frequency bins, complex FFT
        coefficients, original sample count, and inferred sampling interval.
        The input is not modified.

    Raises:
        ValueError: If the trajectory is invalid, contains fewer than two
            observations, or its sampling interval is non-finite or not
            uniform within a relative tolerance of 1e-7.
    """
    time, state = _validate_trajectory(trajectory)
    n_samples = time.size
    if n_samples < 2:
        raise ValueError("FFT requires at least two observations")
    sampling_interval = float((time[-1] - time[0]) / (n_samples - 1))
    if not np.isfinite(sampling_interval) or sampling_interval <= 0:
        raise ValueError("sampling interval must be finite and positive")
    if not np.allclose(np.diff(time), sampling_interval, rtol=1e-7, atol=0):
        raise ValueError("FFT requires uniformly spaced timestamps")

    return FrequencySpectrum(
        frequency=np.fft.rfftfreq(n_samples, d=sampling_interval),
        coefficients=np.fft.rfft(state, axis=-1),
        n_samples=n_samples,
        sampling_interval=sampling_interval,
    )


def compute_spectral_energy(spectrum: FrequencySpectrum) -> EnergySpectrum:
    """Calculate one-sided FFT energies, including the constant offset.

    Ordinary positive-frequency bins receive twice their squared magnitude
    to include the omitted negative-frequency partner. For even sample
    counts, the final (Nyquist) bin is not doubled; for odd counts it is.
    DC retains its squared magnitude without doubling. Callers selecting a
    bandwidth from fluctuation energy should exclude DC in that calculation.
    The input spectrum is not modified. Zero-energy channels remain zero;
    normalization is left to a separate operation.

    Raises:
        ValueError: If the sample count, frequency or coefficient shapes,
            or coefficient values are invalid.
    """
    n_samples = spectrum.n_samples
    if (
        isinstance(n_samples, (bool, np.bool_))
        or not isinstance(n_samples, (int, np.integer))
        or n_samples < 2
    ):
        raise ValueError("n_samples must be an integer >= 2")
    frequency = np.asarray(spectrum.frequency, dtype=np.float64)
    coefficients = np.asarray(spectrum.coefficients, dtype=np.complex128)
    n_bins = n_samples // 2 + 1
    if frequency.shape != (n_bins,):
        raise ValueError("frequency must have shape (n_samples // 2 + 1,)")
    if (
        coefficients.ndim != 2
        or coefficients.shape[0] == 0
        or coefficients.shape[1] != n_bins
    ):
        raise ValueError("coefficients must have shape (D, n_samples // 2 + 1) with D > 0")
    if not np.isfinite(coefficients).all():
        raise ValueError("coefficients must contain only finite values")

    energy = np.abs(coefficients) ** 2
    if n_samples % 2 == 0:
        energy[:, 1:-1] *= 2
    else:
        energy[:, 1:] *= 2

    return EnergySpectrum(
        frequency=frequency.copy(),
        energy=energy,
        n_samples=n_samples,
        sampling_interval=spectrum.sampling_interval,
    )


def _validate_trajectory(
    trajectory: Trajectory,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate the shared time/state layout without requiring uniform times."""
    if np.iscomplexobj(trajectory.time) or np.iscomplexobj(trajectory.state):
        raise ValueError("trajectory must contain only real values")
    time = np.asarray(trajectory.time, dtype=np.float64)
    state = np.asarray(trajectory.state, dtype=np.float64)
    if time.ndim != 1 or time.size == 0:
        raise ValueError("trajectory.time must have nonempty shape (N,)")
    if state.ndim != 2 or state.shape[0] == 0 or state.shape[1] != time.size:
        raise ValueError("trajectory.state must have shape (D, N) with D > 0")
    if not np.isfinite(time).all() or not np.isfinite(state).all():
        raise ValueError("trajectory must contain only finite values")
    if np.any(np.diff(time) <= 0):
        raise ValueError("trajectory.time must be strictly increasing")
    return time, state
