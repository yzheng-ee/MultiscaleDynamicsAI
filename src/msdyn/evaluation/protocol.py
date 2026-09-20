"""Windowing and the head-to-head evaluation harness.

Every forecaster sees exactly the same context windows and is scored on exactly the same targets, with the same metrics. 
Horizons are reported in Lyapunov times wherever one is supplied, because "128 steps" means something different at every sampling rate and every eps.
"""

from __future__ import annotations
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any
import numpy as np
from msdyn.evaluation.metrics import (error_vs_lead_time,invariant_measure_kl,mae,rmse,smape,spectral_hellinger,valid_prediction_time,)
from msdyn.models.base import Forecaster


@dataclass(frozen=True)
class EvaluationWindows:
    """Aligned context/target pairs drawn from one or more trajectories."""
    context: np.ndarray  # (n_windows, context_length, n_channels)
    target: np.ndarray  # (n_windows, horizon, n_channels)
    dt: float
    climatology_std: np.ndarray  # (n_channels,) from the full source trajectories
    lyapunov_time: float | None = None

    @property
    def n_windows(self) -> int:
        return self.context.shape[0]

    @property
    def horizon(self) -> int:
        return self.target.shape[1]

    @property
    def context_length(self) -> int:
        return self.context.shape[1]

    @property
    def lead_times(self) -> np.ndarray:
        """Lead time of each forecast step, in time units."""
        return np.arange(1, self.horizon + 1) * self.dt

    @property
    def lead_times_lyapunov(self) -> np.ndarray:
        """Lead times in Lyapunov times, the only cross-system-comparable axis."""
        if self.lyapunov_time is None:
            raise AttributeError("no lyapunov_time supplied for these windows")
        return self.lead_times / self.lyapunov_time

    def select_channels(self, channels: Sequence[int]) -> EvaluationWindows:
        """Restrict to a channel subset (e.g. the slow variables of a multiscale run)."""
        idx = list(channels)
        return EvaluationWindows(
            context=self.context[:, :, idx],
            target=self.target[:, :, idx],
            dt=self.dt,
            climatology_std=self.climatology_std[idx],
            lyapunov_time=self.lyapunov_time,
        )


def make_windows(trajectories: np.ndarray,context_length: int,horizon: int,dt: float,*,
    stride: int | None = None,
    max_windows: int | None = None,
    lyapunov_time: float | None = None,
    seed: int = 0,
) -> EvaluationWindows:
    """Cut aligned (context, target) pairs out of an ensemble of trajectories.
    stride: Step between window starts. Defaults to horizon, which makes the targets non-overlapping,
        overlapping targets inflate apparent confidence by correlating the errors being averaged.
    max_windows: Cap on the number of windows; a reproducible random subset is taken.
    """
    trajectories = np.asarray(trajectories)
    if trajectories.ndim == 2:
        # A single trajectory (n_times, n_channels). Note np.atleast_3d would append
        # the new axis rather than prepend it, silently transposing the meaning.
        trajectories = trajectories[None]
    if trajectories.ndim != 3:
        raise ValueError(f"expected (n_trajectories, n_times, n_channels) or (n_times, n_channels), got shape {trajectories.shape}")
    n_traj, n_times, n_channels = trajectories.shape

    span = context_length + horizon
    if n_times < span:
        raise ValueError(f"trajectories have {n_times} samples, need at least context_length + horizon = {span}")

    stride = stride if stride is not None else horizon
    starts = np.arange(0, n_times - span + 1, stride)
    index = [(traj, start) for traj in range(n_traj) for start in starts]

    if max_windows is not None and len(index) > max_windows:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(index), size=max_windows, replace=False)
        index = [index[i] for i in sorted(chosen)]

    context = np.empty((len(index), context_length, n_channels))
    target = np.empty((len(index), horizon, n_channels))
    for i, (traj, start) in enumerate(index):
        context[i] = trajectories[traj, start : start + context_length]
        target[i] = trajectories[traj, start + context_length : start + span]

    return EvaluationWindows(
        context=context,
        target=target,
        dt=dt,
        climatology_std=trajectories.reshape(-1, n_channels).std(axis=0),
        lyapunov_time=lyapunov_time,
    )


def evaluate_forecaster(forecaster: Forecaster,windows: EvaluationWindows,*,vpt_threshold: float = 0.3,keep_prediction: bool = False,) -> dict[str, Any]:
    """Score one forecaster on one window set.
    Returns a flat record: summaries plus the full error-vs-lead-time curve.
    """
    prediction = forecaster.forecast(windows.context, windows.horizon)

    vpt = valid_prediction_time(
        prediction,
        windows.target,
        windows.dt,
        threshold=vpt_threshold,
        climatology_std=windows.climatology_std,
    )

    record: dict[str, Any] = {
        "model": forecaster.name,
        "n_windows": windows.n_windows,
        "horizon": windows.horizon,
        "smape": float(smape(prediction, windows.target)),
        "mae": float(mae(prediction, windows.target)),
        "rmse": float(rmse(prediction, windows.target)),
        "vpt": float(np.mean(vpt)),
        "vpt_median": float(np.median(vpt)),
        "kl_invariant_measure": invariant_measure_kl(prediction, windows.target),
        "spectral_hellinger": spectral_hellinger(prediction, windows.target, windows.dt),
        "rmse_vs_lead": error_vs_lead_time(prediction, windows.target, "rmse"),
        "smape_vs_lead": error_vs_lead_time(prediction, windows.target, "smape"),
    }
    if windows.lyapunov_time is not None:
        record["vpt_lyapunov"] = record["vpt"] / windows.lyapunov_time
    if keep_prediction:
        record["prediction"] = prediction
    return record


def compare_forecasters(forecasters: Iterable[Forecaster],windows: EvaluationWindows,**kwargs: Any,) -> list[dict[str, Any]]:
    """Score several forecasters on identical windows."""
    return [evaluate_forecaster(f, windows, **kwargs) for f in forecasters]

def to_dataframe(records: list[dict[str, Any]]):
    """Summary table of scalar columns (drops the per-lead-time curves)."""
    import pandas as pd

    scalar = [{k: v for k, v in record.items() if np.isscalar(v) or isinstance(v, str)}for record in records]
    return pd.DataFrame(scalar).set_index("model")
