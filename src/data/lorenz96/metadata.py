"""Build metadata for Lorenz–96 trajectories, fitted closures, and downsampled views.

Callers supply settings, input provenance, and generated or derived trajectories.
These functions do not modify the supplied records.
max_step is recorded as supplied: None uses the library's sampling-interval
default, a finite value sets an explicit cap, and positive infinity is encoded
as "inf" for standard JSON compatibility.
duration and n_samples are recorded as supplied, including None for the
unspecified length option. The sampling section records the actual sample count.
"""

from collections.abc import Mapping
from typing import Any

from .._utils.json import _json_number
from .simulate import Trajectory


def _build_metadata(
    *,
    filename: str,
    family: str,
    parameters: Mapping[str, Any],
    settings: Mapping[str, Any],
    trajectory: Trajectory,
    generation_seconds: float,
) -> dict[str, Any]:
    n_samples = int(trajectory.time.size)
    arrays = {"time": [n_samples], "slow_states": [int(parameters["K"]), n_samples]}
    if family == "multiscale":
        arrays["fast_states"] = [int(parameters["K"]), int(parameters["J"]), n_samples]

    return {
        "format_version": 1,
        "family": family,
        "filename": filename,
        "parameters": dict(parameters),
        "settings": dict(settings) | {
            "max_step": _json_number(settings["max_step"]),
        },
        "sampling": {
            "sample_dt": settings["sample_dt"],
            "n_samples": int(trajectory.time.size),
        },
        "arrays": arrays,
        "time_range": [float(trajectory.time[0]), float(trajectory.time[-1])],
        "generation_seconds": generation_seconds,
    }


def build_multiscale_metadata(
    *,
    filename: str,
    parameters: Mapping[str, Any],
    settings: Mapping[str, Any],
    trajectory: Trajectory,
    generation_seconds: float,
) -> dict[str, Any]:
    """Describe a named full trajectory with supplied settings and actual sampling information."""
    return _build_metadata(
        filename=filename, family="multiscale", parameters=parameters, settings=settings,
        trajectory=trajectory, generation_seconds=generation_seconds,
    )


def build_singlescale_metadata(
    *,
    filename: str,
    parameters: Mapping[str, Any],
    settings: Mapping[str, Any],
    closure: Mapping[str, Any],
    trajectory: Trajectory,
    generation_seconds: float,
) -> dict[str, Any]:
    """Describe a named reduced trajectory using caller-supplied closure provenance.

    closure contains the saved closure path and checksum. Reading and hashing
    that input remain the caller's responsibility.
    """
    metadata = _build_metadata(
        filename=filename, family="single_scale", parameters=parameters, settings=settings,
        trajectory=trajectory, generation_seconds=generation_seconds,
    )
    metadata["closure"] = dict(closure)
    return metadata


def build_closure_metadata(
    *,
    filename: str,
    source_trajectory: str,
    source_sha256: str,
    settings: Mapping[str, Any],
    fit_results: Mapping[str, Any],
    training_seconds: float,
) -> dict[str, Any]:
    """Record the source path and checksum, supplied settings, and training diagnostics."""
    return {
        "format_version": 1,
        "filename": filename,
        "source_trajectory": source_trajectory,
        "source_sha256": source_sha256,
        "settings": dict(settings),
        "fit_results": dict(fit_results),
        "training_seconds": training_seconds,
    }


def build_downsampling_metadata(
    *,
    filename: str,
    source_trajectory: str,
    source_sha256: str,
    settings: Mapping[str, Any],
    trajectory: Trajectory,
    downsampling_seconds: float,
) -> dict[str, Any]:
    """Describe a slow-channel view with its source checksum and supplied stride.

    trajectory contains the already downsampled slow states. Sampling spacing,
    shapes, and time range describe that result. No source files are read.
    """
    n_samples = int(trajectory.time.size)
    return {
        "format_version": 1,
        "family": "multiscale",
        "view": "slow_downsampled",
        "filename": filename,
        "source_trajectory": source_trajectory,
        "source_sha256": source_sha256,
        "settings": dict(settings),
        "sampling": {
            "sample_dt": float((trajectory.time[-1] - trajectory.time[0]) / (n_samples - 1)),
            "n_samples": n_samples,
        },
        "arrays": {"time": [n_samples], "slow_states": list(trajectory.state.shape)},
        "time_range": [float(trajectory.time[0]), float(trajectory.time[-1])],
        "downsampling_seconds": downsampling_seconds,
    }
