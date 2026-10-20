"""Public API for spectral analysis and AC bandwidth estimation."""

from .bandwidth import estimate_channel_bandwidths, reduce_bandwidths
from .estimation import estimate_trajectory_bandwidths
from .spectral import (
    EnergySpectrum,
    FrequencySpectrum,
    compute_fft,
    compute_spectral_energy,
    subtract_mean,
)

__all__ = [
    "EnergySpectrum",
    "FrequencySpectrum",
    "compute_fft",
    "compute_spectral_energy",
    "estimate_channel_bandwidths",
    "estimate_trajectory_bandwidths",
    "reduce_bandwidths",
    "subtract_mean",
]
