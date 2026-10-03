"""Configuration for reusable Panda forecasting fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PandaTrainerConfig:
    """Optimization and data-loading settings for :class:`PandaTrainer`."""

    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    use_cpu: bool = False
    num_workers: int = 0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
