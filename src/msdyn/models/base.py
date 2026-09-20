"""The forecaster interface every model in this package implements.

Keeping PANDA, the physics baselines and the trivial baselines behind one signature is 
what makes the comparison honest: the evaluation harness cannot accidentally hand one of them information the others do not get.
"""

from __future__ import annotations
from typing import Protocol, runtime_checkable
import numpy as np

@runtime_checkable
class Forecaster(Protocol):
    """Maps a batch of context windows to a batch of forecasts.

    Implementations must be *causal and stateless*: the forecast for window i may depend only on context[i], 
    never on the ground-truth futureor on other windows.
    """

    name: str
    def forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        """Forecast horizon steps ahead.
        horizon: Number of steps to predict.
        """
        ...


def validate_forecast(context: np.ndarray, forecast: np.ndarray, horizon: int) -> None:
    """Assert a forecaster honoured the contract. Cheap; call it in every implementation."""
    expected = (context.shape[0], horizon, context.shape[2])
    if forecast.shape != expected:
        raise ValueError(f"forecast shape {forecast.shape} != expected {expected}")
    if not np.all(np.isfinite(forecast)):
        raise ValueError("forecast contains non-finite values")
