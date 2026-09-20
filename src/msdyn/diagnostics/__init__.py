"""Dynamical diagnostics: timescales, spectra, and attractor statistics."""

from msdyn.diagnostics.dimension import (
    PANDA_BASE_SYSTEMS_GP_DIM,
    PANDA_SKEW_SYSTEMS_GP_DIM,
    correlation_dimension,
    correlation_dimension_ensemble,
)
from msdyn.diagnostics.lyapunov import lyapunov_time, max_lyapunov_exponent
from msdyn.diagnostics.spectra import (
    dominant_period,
    points_per_period,
    power_spectrum,
    resample_to_points_per_period,
    subsample_for_points_per_period,
)

__all__ = [
    "correlation_dimension",
    "correlation_dimension_ensemble",
    "PANDA_BASE_SYSTEMS_GP_DIM",
    "PANDA_SKEW_SYSTEMS_GP_DIM",
    "max_lyapunov_exponent",
    "lyapunov_time",
    "power_spectrum",
    "dominant_period",
    "points_per_period",
    "subsample_for_points_per_period",
    "resample_to_points_per_period",
]
