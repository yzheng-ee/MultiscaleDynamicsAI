"""Panda forecasting data and native HF training interface."""

from typing import TYPE_CHECKING

from .data import PandaForecastDataset, dataset_from_cache, windows_from_trajectories, windows_from_trajectory

if TYPE_CHECKING:
    from .trainer import PandaTrainer

__all__ = [
    "PandaForecastDataset",
    "PandaTrainer",
    "windows_from_trajectory",
    "windows_from_trajectories",
    "dataset_from_cache",
]


def __getattr__(name: str) -> type:
    # Data preparation does not need the upstream Panda model or HF Trainer.
    if name == "PandaTrainer":
        from .trainer import PandaTrainer

        return PandaTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
