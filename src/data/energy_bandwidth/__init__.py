"""Public API for spectral analysis and AC energy bandwidth estimation."""

from .bandwidth import bandwidth_to_period, estimate_channel_bandwidths, reduce_bandwidths
from .estimation import BandwidthEstimate, estimate
from .sampling import align_multiscale_sampling, sampling_interval_from_period
from .spectral import (
    EnergySpectrum,
    FrequencySpectrum,
    compute_fft,
    compute_spectral_energy,
)

__all__ = [
    "EnergySpectrum",
    "FrequencySpectrum",
    "BandwidthEstimate",
    "align_multiscale_sampling",
    "bandwidth_to_period",
    "compute_fft",
    "compute_spectral_energy",
    "estimate_channel_bandwidths",
    "estimate",
    "reduce_bandwidths",
    "sampling_interval_from_period",
]
