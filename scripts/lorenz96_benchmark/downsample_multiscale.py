r"""Save a downsampled slow-channel view of one multiscale trajectory.

Example from the repository root in the lorenz96_benchmark environment::

    python scripts/lorenz96_benchmark/downsample_multiscale.py \
        --input-path data/lorenz96_benchmark/training/multiscale/a/trajectory_001.npz \
        --stride 128 \
        --output-prefix data/lorenz96_benchmark/training/multiscale/a/slow_downsampled/trajectory_001

The input NPZ contains time [N], slow_states [K, N], and fast_states [K, J, N].
Only time and slow_states are loaded. Every stride-th sample is selected,
including both endpoints; stride must divide N - 1. No filtering or new
integration occurs. The source trajectory is unchanged.

Writes <output-prefix>.npz containing time and slow_states, plus matching
<output-prefix>.json with source provenance and downsampling information.
Rerunning replaces that output pair. Relative paths use the working directory.
Experiment shell files select collections and launch multiple processes.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--input-path", type=Path, required=True, help="Multiscale trajectory NPZ containing time, slow_states, and fast_states.")
    parser.add_argument("--stride", type=int, required=True, help="Positive number of source intervals between retained samples; must divide N - 1 to preserve both endpoints.")
    parser.add_argument("--output-prefix", type=Path, required=True, help="Output path without an extension; writes <prefix>.npz and <prefix>.json, replacing existing files.")

    args = parser.parse_args()

    if args.stride < 1:
        parser.error("--stride must be positive")

    # Both output extensions are appended to the supplied filename stem
    if (
        not args.output_prefix.name
        or args.output_prefix.name in (".", "..")
        or args.output_prefix.name != args.output_prefix.name.strip()
        or args.output_prefix.suffix
    ):
        parser.error("--output-prefix must include a nonempty filename stem without an extension")

    return args


def main() -> None:
    args = parse_args()

    # Keep --help usable without scientific dependencies
    import numpy as np

    from src.data.lorenz96 import Trajectory, downsample_trajectory
    from src.data.lorenz96.metadata import build_downsampling_metadata

    source = args.input_path.resolve()
    output = args.output_prefix.parent.resolve()
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(output)

    trajectory_path = output / f"{args.output_prefix.name}.npz"
    metadata_path = output / f"{args.output_prefix.name}.json"
    for path in (trajectory_path, metadata_path):
        if path.resolve() == source:
            raise ValueError("Output must not overwrite the input trajectory")
        if path.is_dir():
            raise IsADirectoryError(path)

    # Stream the checksum without decompressing the fast states or loading the entire file into memory
    with source.open("rb") as file:
        source_hash = hashlib.file_digest(file, "sha256").hexdigest()
        file.seek(0)
        with np.load(file, allow_pickle=False) as data:
            if "fast_states" not in data.files:
                raise ValueError("Input must be a multiscale trajectory containing fast_states")
            trajectory = Trajectory(time=data["time"], state=data["slow_states"])

    started = perf_counter()
    downsampled = downsample_trajectory(trajectory, stride=args.stride)
    downsampling_seconds = perf_counter() - started
    del trajectory

    metadata = build_downsampling_metadata(
        filename=trajectory_path.name, source_trajectory=str(source), source_sha256=source_hash,
        settings=dict(stride=args.stride), trajectory=downsampled, downsampling_seconds=downsampling_seconds,
    )
    metadata_text = json.dumps(metadata, indent=2, allow_nan=False) + "\n"

    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{args.output_prefix.name}-", dir=output) as temporary:
        staging = Path(temporary)
        np.savez_compressed(staging / trajectory_path.name, time=downsampled.time, slow_states=downsampled.state)
        (staging / metadata_path.name).write_text(metadata_text, encoding="utf-8")

        # Prepare both files before replacing either member of the output pair
        for name in (trajectory_path.name, metadata_path.name):
            (staging / name).replace(output / name)

    print(f"Saved downsampled slow trajectory {list(downsampled.state.shape)} in {downsampling_seconds:.2f}s: {trajectory_path}")


if __name__ == "__main__":
    main()
