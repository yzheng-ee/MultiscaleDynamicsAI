"""Forecasters: baselines, physics-based reductions, and pretrained models."""

from msdyn.models.base import Forecaster, validate_forecast
from msdyn.models.baselines import (
    ClimatologyForecaster,
    ClosureModelForecaster,
    LinearAutoregressiveForecaster,
    PerfectModelForecaster,
    PersistenceForecaster,
)

__all__ = [
    "Forecaster",
    "validate_forecast",
    "PersistenceForecaster",
    "ClimatologyForecaster",
    "LinearAutoregressiveForecaster",
    "ClosureModelForecaster",
    "PerfectModelForecaster",
    "PandaForecaster",
]


def __getattr__(name: str):
    # Imported lazily: pulling in PANDA drags in torch and transformers, which
    # nothing else in this package needs.
    if name == "PandaForecaster":
        from msdyn.models.panda import PandaForecaster

        return PandaForecaster
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
