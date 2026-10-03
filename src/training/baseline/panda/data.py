"""Panda-compatible forecasting and masked-pretraining datasets."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class _PandaWindowDataset(Dataset[dict[str, torch.Tensor]]):
    """Shared fixed-length window indexing for Panda datasets."""

    def __init__(
        self,
        trajectories: Sequence[np.ndarray | torch.Tensor],
        *,
        window_length: int,
        stride: int,
    ) -> None:
        if not trajectories:
            raise ValueError("at least one trajectory is required")
        if window_length <= 0:
            raise ValueError("window_length must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")

        self.stride = stride
        self.trajectories: list[torch.Tensor] = []
        self._windows: list[tuple[int, int]] = []
        channel_count: int | None = None

        for trajectory_index, trajectory in enumerate(trajectories):
            values = torch.as_tensor(trajectory, dtype=torch.float32)
            if values.ndim == 1:
                values = values.unsqueeze(-1)
            if values.ndim != 2:
                raise ValueError(
                    f"trajectory {trajectory_index} must have shape [time, channels], "
                    f"got {tuple(values.shape)}"
                )
            if values.shape[1] == 0:
                raise ValueError(f"trajectory {trajectory_index} has no channels")
            if values.shape[0] < window_length:
                raise ValueError(
                    f"trajectory {trajectory_index} has {values.shape[0]} time points; "
                    f"at least {window_length} are required"
                )
            if channel_count is None:
                channel_count = values.shape[1]
            elif values.shape[1] != channel_count:
                raise ValueError("all trajectories must have the same channel count")

            self.trajectories.append(values.contiguous())
            self._windows.extend(
                (trajectory_index, start)
                for start in range(
                    0,
                    values.shape[0] - window_length + 1,
                    stride,
                )
            )

    def __len__(self) -> int:
        return len(self._windows)

    def _window(self, index: int, length: int) -> torch.Tensor:
        trajectory_index, start = self._windows[index]
        return self.trajectories[trajectory_index][start : start + length]


class PandaForecastDataset(_PandaWindowDataset):
    """Create supervised forecasting windows from time-major trajectories."""

    def __init__(
        self,
        trajectories: Sequence[np.ndarray | torch.Tensor],
        *,
        context_length: int,
        prediction_length: int,
        stride: int = 1,
    ) -> None:
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if prediction_length <= 0:
            raise ValueError("prediction_length must be positive")

        self.context_length = context_length
        self.prediction_length = prediction_length
        super().__init__(
            trajectories,
            window_length=context_length + prediction_length,
            stride=stride,
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        values = self._window(
            index, self.context_length + self.prediction_length
        )
        past_values = values[: self.context_length]
        future_values = values[self.context_length :]

        if not torch.isfinite(future_values).all():
            raise ValueError("future training targets must contain only finite values")

        return {
            "past_values": torch.nan_to_num(past_values),
            "past_observed_mask": torch.isfinite(past_values),
            "future_values": future_values,
        }


class PandaMLMDataset(_PandaWindowDataset):
    """Create masked-pretraining inputs from time-major trajectories.

    Patch masking is performed inside :class:`PatchTSTForPretraining` according
    to the model's configuration. This dataset supplies only the unmasked
    context and its observation mask.
    """

    def __init__(
        self,
        trajectories: Sequence[np.ndarray | torch.Tensor],
        *,
        context_length: int,
        stride: int = 1,
    ) -> None:
        if context_length <= 0:
            raise ValueError("context_length must be positive")

        self.context_length = context_length
        super().__init__(
            trajectories,
            window_length=context_length,
            stride=stride,
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        past_values = self._window(index, self.context_length)
        return {
            "past_values": torch.nan_to_num(past_values),
            "past_observed_mask": torch.isfinite(past_values),
        }
