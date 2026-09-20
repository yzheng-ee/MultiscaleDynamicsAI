"""On-disk format for generated trajectories.

A dataset is a single .npz holding an ensemble of trajectories plus a JSON metadata blob. 
Metadata is not optional: a Lorenz '96 trajectory is uninterpretable without eps, dt and the sampling stride, and forecast experiments compare across exactly those axes.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
from msdyn.systems.l96 import L96Params

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TrajectoryDataset:
    """An ensemble of trajectories on a common time grid.
    params: Lorenz '96 parameters used to generate it.
    kind
        "multiscale" (channels are K slow then K*J fast) or
        "singlescale" (channels are the K slow variables only).
    metadata: Free-form provenance: integrator step, sampling stride, burn-in time, seed, closure description, and so on.
    """

    trajectories: np.ndarray
    times: np.ndarray
    params: L96Params
    kind: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.trajectories.ndim != 3:
            raise ValueError(f"trajectories must be (n_traj, n_times, n_channels), got {self.trajectories.shape}")
        if self.trajectories.shape[1] != self.times.size:
            raise ValueError(f"time axis mismatch: {self.trajectories.shape[1]} samples vs {self.times.size} times")
        if self.kind not in {"multiscale", "singlescale"}:
            raise ValueError(f"unknown kind {self.kind!r}")

    @property
    def dt(self) -> float:
        """Sampling interval of the stored trajectories (not the integrator step)."""
        return float(self.times[1] - self.times[0])

    @property
    def n_trajectories(self) -> int:
        return self.trajectories.shape[0]

    @property
    def slow(self) -> np.ndarray:
        """Slow variables."""
        return self.trajectories[..., : self.params.K]

    @property
    def fast(self) -> np.ndarray:
        """Fast variables. Multiscale only."""
        if self.kind != "multiscale":
            raise AttributeError(f"a {self.kind} dataset has no fast variables")
        return self.trajectories[..., self.params.K :]

    def fast_average(self) -> np.ndarray:
        """ybar, the exact closure target."""
        from msdyn.systems.l96 import average_fast
        return average_fast(self.fast, self.params)

    def describe(self) -> str:
        p = self.params
        return (
            f"{self.kind} | {self.n_trajectories} x {self.times.size} x "
            f"{self.trajectories.shape[2]} | dt={self.dt:g} | "
            f"T={self.times[-1] - self.times[0]:g} | "
            f"K={p.K} J={p.J} F={p.F} h_x={p.h_x} h_y={p.h_y} eps={p.eps:g}"
        )


def save_dataset(dataset: TrajectoryDataset, path: str | Path) -> Path:
    """Write a dataset to path (.npz), creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "kind": dataset.kind,
        "params": dataset.params.as_dict(),
        **dataset.metadata,
    }
    np.savez_compressed(
        path,
        trajectories=dataset.trajectories,
        times=dataset.times,
        metadata=json.dumps(metadata),
    )
    return path


def load_dataset(path: str | Path) -> TrajectoryDataset:
    """Read a dataset written by :func:`save_dataset`."""
    with np.load(path, allow_pickle=False) as handle:
        metadata = json.loads(str(handle["metadata"]))
        version = metadata.pop("schema_version", None)
        if version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema version {version!r} in {path}")
        return TrajectoryDataset(
            trajectories=handle["trajectories"],
            times=handle["times"],
            params=L96Params(**metadata.pop("params")),
            kind=metadata.pop("kind"),
            metadata=metadata,
        )
