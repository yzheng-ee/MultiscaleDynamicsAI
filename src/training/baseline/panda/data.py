"""Prepare trajectory windows and expose them through Panda's dataset interface."""

from collections.abc import Sequence
import operator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from src.data.lorenz96.simulate import Trajectory


def _validate_window_settings(context_length: int, prediction_length: int, stride: int = 1) -> None:
    for name, value in (("context_length", context_length), ("prediction_length", prediction_length), ("stride", stride)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")


class PandaForecastDataset(Dataset):
    """Expose prepared windows as Panda history/target samples.

    Float32 NumPy arrays, including memory-mapped caches, are retained without
    copying the full array. Each sample is copied into a CPU tensor and checked
    for finite values when read. Do not modify the underlying windows during use.
    """

    def __init__(
        self,
        windows: np.ndarray,  # [num_windows, context_length + prediction_length, channels].
        *,
        context_length: int = 512,
        prediction_length: int = 128,
    ) -> None:
        _validate_window_settings(context_length, prediction_length)
        length = context_length + prediction_length
        if windows.ndim != 3 or windows.shape[0] == 0 or windows.shape[1] != length or windows.shape[2] == 0:
            raise ValueError(f"expected nonempty windows [num_windows, {length}, channels]")
        if np.iscomplexobj(windows):
            raise ValueError("windows must be real-valued")
        self.context_length, self.prediction_length = context_length, prediction_length
        self.windows = np.asarray(windows, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        index = operator.index(index)
        if not -len(self) <= index < len(self):
            raise IndexError(index)
        window = torch.from_numpy(np.array(self.windows[index], copy=True))
        if not torch.isfinite(window).all():
            raise ValueError("windows must contain only finite float32 values")
        past = window[: self.context_length]
        return {
            "past_values": past,
            "past_observed_mask": torch.ones_like(past, dtype=torch.bool),
            "future_values": window[self.context_length :],
        }


def windows_from_trajectory(
    trajectory: "Trajectory",
    *,
    context_length: int = 512,
    prediction_length: int = 128,
    stride: int = 128,
    channels: Sequence[int] | None = None,
) -> np.ndarray:
    """Return float32 windows [num_windows, context + prediction, channels].

    Select zero-based state rows in the supplied order; None uses all rows.
    Explicitly select slow rows when passing a full multiscale trajectory.
    Windows include only complete history/target pairs and own their data.
    """
    _validate_window_settings(context_length, prediction_length, stride)
    if channels is not None:
        channels = [operator.index(index) for index in channels]
        if not channels or min(channels) < 0 or len(set(channels)) != len(channels):
            raise ValueError("channels must contain distinct nonnegative indices")

    state = trajectory.state
    if state.ndim != 2 or trajectory.time.ndim != 1 or state.shape[1] != len(trajectory.time):
        raise ValueError("expected state [channels, time] matching time [time]")
    if state.shape[0] == 0 or np.iscomplexobj(state):
        raise ValueError("trajectory state must have real-valued channels")
    if channels is not None and max(channels) >= state.shape[0]:
        raise ValueError("selected channel outside trajectory state")
    length = context_length + prediction_length
    if state.shape[1] < length:
        raise ValueError(f"trajectory must contain at least {length} observations")
    values = np.asarray((state if channels is None else state[channels]).T, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("observations must contain only finite float32 values")
    windows = np.lib.stride_tricks.sliding_window_view(values, length, axis=0)[::stride]
    return np.array(windows.transpose(0, 2, 1), dtype=np.float32, order="C", copy=True)


def windows_from_trajectories(
    trajectories: Sequence["Trajectory"],
    *,
    context_length: int = 512,
    prediction_length: int = 128,
    stride: int = 128,
    channels: Sequence[int] | None = None,
) -> np.ndarray:
    """Concatenate trajectory windows without crossing trajectory boundaries.

    Return float32 [num_windows, context + prediction, channels]. Trajectories
    must already belong to the same split and share a selected channel count.
    """
    if not len(trajectories):
        raise ValueError("at least one trajectory is required")
    windows = [
        windows_from_trajectory(
            trajectory, context_length=context_length, prediction_length=prediction_length,
            stride=stride, channels=channels,
        )
        for trajectory in trajectories
    ]
    if any(value.shape[2] != windows[0].shape[2] for value in windows):
        raise ValueError("trajectories must share a channel count")
    return windows[0] if len(windows) == 1 else np.concatenate(windows, axis=0)


def dataset_from_cache(
    windows_path: str | Path,
    *,
    context_length: int,
    prediction_length: int,
) -> PandaForecastDataset:
    """Load a windows .npy file using explicitly supplied history/target lengths.

    Windows remain memory-mapped read-only; individual samples are copied when
    accessed. No metadata, source trajectories, or manifest are needed.
    """
    windows = np.load(windows_path, mmap_mode="r", allow_pickle=False)
    if windows.dtype != np.float32:
        raise ValueError("Cached windows must use float32")
    return PandaForecastDataset(
        windows,
        context_length=context_length,
        prediction_length=prediction_length,
    )
