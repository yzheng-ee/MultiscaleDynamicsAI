"""Build metadata from bandwidth estimation settings and results."""

from collections.abc import Mapping
from typing import Any

from .._utils.json import _json_number
from .estimation import BandwidthEstimate


def _estimate_metadata(result: BandwidthEstimate | None) -> dict[str, Any] | None:
    """Format one group's result, preserving None when the group is absent."""
    if result is None:
        return None
    return {
        "channel_bandwidths": [_json_number(value) for value in result.channel_bandwidths],
        "reduced_bandwidth": _json_number(result.reduced_bandwidth),
        "bandwidth_period": _json_number(result.bandwidth_period),
        "sampling_interval": _json_number(result.sampling_interval),
    }


def build_estimation_metadata(
    *,
    source_trajectory: str,
    source_sha256: str,
    settings: Mapping[str, Any],
    slow_result: BandwidthEstimate,
    estimation_seconds: float,
    fast_result: BandwidthEstimate | None = None,
    sampling: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Record the source path and checksum, settings, and separate group estimates.

    Arrays become lists, unrequested fields remain None, and infinity is
    encoded as "inf" or "-inf" for standard JSON compatibility. Scientific
    calculations and channel grouping are handled by the caller. Inputs are
    not modified. A missing fast group is recorded as None. sampling contains
    caller-computed multiscale alignment values, or None for single-scale input.
    """
    return {
        "format_version": 1,
        "source_trajectory": source_trajectory,
        "source_sha256": source_sha256,
        "settings": dict(settings),
        "estimate": {
            "slow": _estimate_metadata(slow_result),
            "fast": _estimate_metadata(fast_result),
        },
        "sampling": dict(sampling) if sampling is not None else None,
        "estimation_seconds": estimation_seconds,
    }
