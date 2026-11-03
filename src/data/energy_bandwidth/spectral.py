"""Spectral utilities for energy bandwidth estimation."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..lorenz96.simulate import Trajectory
from .._utils.validation import _validate_finite, _validate_real


@dataclass(frozen=True)
class FrequencySpectrum:
    """One-sided FFT coefficients and frequencies for a real-valued trajectory.

    frequency has shape (K,) in cycles per unit of time; coefficients has
    shape (D, K), where K = n_samples // 2 + 1. Coefficients use NumPy's
    unscaled forward transform and refer to sample indices relative to the
    trajectory's first timestamp. n_samples retains the original odd/even
    parity needed for one-sided energy weighting. sampling_interval is the
    time between observations.
    """

    frequency: NDArray[np.float64]
    coefficients: NDArray[np.complex128]
    n_samples: int
    sampling_interval: float


@dataclass(frozen=True)
class EnergySpectrum:
    """One-sided spectral energies, including DC.

    frequency has shape (K,) and energy has shape (D, K), where
    K = n_samples // 2 + 1. With NumPy's unscaled forward FFT, the energy sum
    per channel equals n_samples times the sum of squared time-domain values.
    sampling_interval is the time between observations. Energy values retain
    this FFT normalization; divide by the total when energy fractions are needed.
    """

    frequency: NDArray[np.float64]
    energy: NDArray[np.float64]
    n_samples: int
    sampling_interval: float


def compute_fft(trajectory: Trajectory) -> FrequencySpectrum:
    """Compute one-sided FFT coefficients and their physical frequencies.

    trajectory contains finite real states of shape (D, N) and finite,
    strictly increasing times of shape (N,), with at least two observations.
    Timestamps must be uniformly spaced within a relative tolerance of 1e-7.

    Returns a FrequencySpectrum with frequencies in cycles per trajectory
    time unit, complex coefficients, the sample count, and the inferred
    sampling interval. The input is not modified and the DC bin is retained.
    Nonfinite coefficients and nonfinite or collapsed frequency grids caused
    by numerical overflow are rejected.
    """
    # Reject complex inputs before conversion can discard imaginary components
    _validate_real([("trajectory.time", trajectory.time), ("trajectory.state", trajectory.state)])
    time = np.asarray(trajectory.time, dtype=np.float64)
    state = np.asarray(trajectory.state, dtype=np.float64)
    if time.ndim != 1 or time.size == 0:
        raise ValueError("trajectory.time must have nonempty shape (N,)")
    if state.ndim != 2 or state.shape[0] == 0 or state.shape[1] != time.size:
        raise ValueError("trajectory.state must have shape (D, N) with D > 0")

    _validate_finite([("trajectory.time", time), ("trajectory.state", state)])
    if np.any(np.diff(time) <= 0):
        raise ValueError("trajectory.time must be strictly increasing")

    n_samples = time.size
    if n_samples < 2:
        raise ValueError("FFT requires at least two observations")

    # N observations span N - 1 intervals between the first and last timestamps
    sampling_interval = float((time[-1] - time[0]) / (n_samples - 1))
    _validate_finite([("sampling interval", sampling_interval)])
    if sampling_interval <= 0:
        raise ValueError("sampling interval must be positive")

    # A single sampling interval must describe every step to define the FFT frequency grid
    if not np.allclose(np.diff(time), sampling_interval, rtol=1e-7, atol=0):
        raise ValueError("FFT requires uniformly spaced timestamps")

    # Finite inputs can still overflow during FFT computation or frequency scaling
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        frequency = np.fft.rfftfreq(n_samples, d=sampling_interval)
        coefficients = np.fft.rfft(state, axis=-1)
    _validate_finite([("frequency", frequency), ("coefficients", coefficients)])
    if frequency[0] != 0 or np.any(np.diff(frequency) <= 0):
        raise ValueError("sampling interval cannot produce a strictly increasing FFT frequency grid starting at zero")

    return FrequencySpectrum(
        frequency=frequency,
        coefficients=coefficients,
        n_samples=n_samples,
        sampling_interval=sampling_interval,
    )


def compute_spectral_energy(trajectory: Trajectory) -> EnergySpectrum:
    """Compute one-sided spectral energies from a uniformly sampled trajectory.

    Uses compute_fft to validate the trajectory and obtain its coefficients
    and frequency grid. The trajectory follows the same input requirements
    as compute_fft and is not modified.

    Ordinary positive-frequency bins receive twice their squared magnitude
    to include the omitted negative-frequency partner. For even sample
    counts, the final (Nyquist) bin is not doubled; for odd counts it is.
    DC retains its squared magnitude without doubling. Bandwidth estimation
    excludes that bin when selecting frequencies from AC energy.

    Returns an EnergySpectrum with the original frequency grid, sample count,
    and sampling interval. Zero-energy channels remain zero, and energy values
    are not normalized to fractions.
    Nonfinite energies caused by numerical overflow are rejected.
    """
    spectrum = compute_fft(trajectory)
    n_samples = spectrum.n_samples

    # Check the final energies explicitly rather than returning overflowed values
    with np.errstate(over="ignore", invalid="ignore"):
        energy = np.abs(spectrum.coefficients) ** 2

        # Account for omitted conjugate partners without doubling DC or the even-length Nyquist bin
        if n_samples % 2 == 0:
            energy[:, 1:-1] *= 2
        else:
            energy[:, 1:] *= 2
    _validate_finite([("spectral energy", energy)])

    return EnergySpectrum(
        frequency=spectrum.frequency,
        energy=energy,
        n_samples=n_samples,
        sampling_interval=spectrum.sampling_interval,
    )
