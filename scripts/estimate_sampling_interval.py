#!/usr/bin/env python3
"""Estimate a period-normalized sampling interval from a dense trajectory."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.timescales import estimate_channel_periods, estimate_sampling_interval


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    # Input parameters identify the trajectory array and its time dimension.
    parser.add_argument("--input", type=Path, required=True, help="input .npy or .npz file")
    parser.add_argument("--array-key", default="states", help="array key for an .npz file (default: states)")
    parser.add_argument("--sampling-interval", type=float, required=True, help="physical time between observations in the dense input")
    parser.add_argument("--time-axis", type=int, default=-1, help="axis containing time (default: -1)")
    parser.add_argument("--channels", help="flattened channel indices as a slice (for example 0:9) or list (0,2,4)")

    # Period-estimation parameters control how channel periods are aggregated.
    parser.add_argument("--reduction", choices=("median", "mean", "min", "max"), default="median", help="channel-period reduction (default: median)")

    # Target-sampling parameters define the duration and size of the output grid.
    parser.add_argument("--num-periods", type=float, default=40, help="target period span (default: 40)")
    parser.add_argument("--num-points", type=int, default=4096, help="number of target observations (default: 4096)")

    # Output parameters control the destination directory and result filename.
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="directory for the result file (default: current directory)")
    parser.add_argument("--output-filename", default="sampling_interval.json", help="result filename (default: sampling_interval.json)")
    return parser


def _load_trajectory(path: Path, array_key: str) -> NDArray[np.generic]:
    if path.suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False))
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if array_key not in archive:
                available = ", ".join(archive.files)
                raise ValueError(
                    f"array key {array_key!r} not found; available keys: {available}"
                )
            return np.asarray(archive[array_key])
    raise ValueError("input must be a .npy or .npz file")


def _parse_channel_indices(spec: str, num_channels: int) -> NDArray[np.int64]:
    try:
        if ":" in spec:
            if "," in spec:
                raise ValueError("a channel selection cannot mix slice and list syntax")
            parts = spec.split(":")
            if len(parts) not in (2, 3):
                raise ValueError("a channel slice must have two or three fields")
            values = [int(part) if part else None for part in parts]
            selector = slice(*values)
            indices = np.arange(num_channels, dtype=np.int64)[selector]
        else:
            indices = np.asarray([int(part) for part in spec.split(",")], dtype=np.int64)
            indices = np.arange(num_channels, dtype=np.int64)[indices]
    except (IndexError, ValueError) as error:
        raise ValueError(f"invalid channel selection {spec!r}: {error}") from error

    if indices.size == 0:
        raise ValueError("channel selection must contain at least one channel")
    return np.atleast_1d(indices)


def _select_channels(
    trajectory: NDArray[np.generic],
    time_axis: int,
    channel_spec: str | None,
) -> tuple[NDArray[np.generic], list[int], int]:
    if trajectory.ndim == 0:
        raise ValueError("trajectory must contain a time dimension")
    if time_axis < -trajectory.ndim or time_axis >= trajectory.ndim:
        raise ValueError(
            f"invalid time axis {time_axis} for trajectory shape {trajectory.shape}"
        )
    normalized_time_axis = time_axis % trajectory.ndim

    time_last = np.moveaxis(trajectory, normalized_time_axis, -1)
    channels = time_last.reshape(-1, time_last.shape[-1])
    if channel_spec is None:
        indices = np.arange(channels.shape[0], dtype=np.int64)
        if channels.shape[0] > 1:
            print(
                "warning: --channels was omitted; using every flattened channel",
                file=sys.stderr,
            )
    else:
        indices = _parse_channel_indices(channel_spec, channels.shape[0])
    return channels[indices], indices.tolist(), normalized_time_axis


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        trajectory = _load_trajectory(args.input, args.array_key)
        selected, channel_indices, normalized_time_axis = _select_channels(
            trajectory,
            args.time_axis,
            args.channels,
        )
        channel_periods = estimate_channel_periods(
            selected,
            args.sampling_interval,
            time_axis=-1,
        )
        target_sampling_interval = estimate_sampling_interval(
            selected,
            args.sampling_interval,
            time_axis=-1,
            reduction=args.reduction,
            num_periods=args.num_periods,
            num_points=args.num_points,
        )
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))

    output_filename = Path(args.output_filename)
    if output_filename.name != args.output_filename:
        parser.error("output-filename must be a filename, not a path")

    reduction = getattr(np, args.reduction)
    characteristic_period = float(reduction(channel_periods))
    result = {
        "input": str(args.input),
        "array_key": args.array_key if args.input.suffix == ".npz" else None,
        "input_shape": list(trajectory.shape),
        "time_axis": normalized_time_axis,
        "channel_indices": channel_indices,
        "dense_sampling_interval": args.sampling_interval,
        "channel_periods": channel_periods.tolist(),
        "reduction": args.reduction,
        "characteristic_period": characteristic_period,
        "num_periods": args.num_periods,
        "num_points": args.num_points,
        "target_sampling_interval": target_sampling_interval,
    }
    rendered = json.dumps(result, indent=2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / output_filename
    output_path.write_text(f"{rendered}\n", encoding="utf-8")
    print(f"sampling_interval: {output_path}")


if __name__ == "__main__":
    main()
