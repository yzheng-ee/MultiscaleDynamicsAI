"""Cache Panda windows from one JSON list of trajectory file paths.

Example from the repository root:
    python scripts/panda/prepare_data.py \
        --manifest-path outputs/panda_lorenz96/trajectory_paths/multiscale_training.json \
        --output-dir outputs/panda_lorenz96/windows/multiscale_training

Each input NPZ must contain time [samples] and state [channels, samples].
The cache contains windows.npy, a float32 array [windows, context + prediction,
channels], and metadata.json with settings and source paths. Load the array with
np.load(path, mmap_mode="r") and wrap it in PandaForecastDataset for training.
Rerunning replaces windows.npy and metadata.json in the output directory.
"""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.lorenz96.simulate import Trajectory
from src.training.baseline.panda import windows_from_trajectories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest-path", type=Path, required=True, help="JSON list produced by collect_trajectory_paths.py.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--prediction-length", type=int, default=128)
    parser.add_argument("--stride", type=int, default=128, help="Window start spacing, in observation samples.")
    parser.add_argument("--channels", type=int, nargs="+", help="Zero-based state rows in selection order; default: all.")
    args = parser.parse_args()
    for name in ("context_length", "prediction_length", "stride"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.channels is not None and (min(args.channels) < 0 or len(set(args.channels)) != len(args.channels)):
        parser.error("--channels must contain distinct nonnegative indices")
    return args


def load_trajectories(manifest: Path) -> tuple[list[Trajectory], list[str]]:
    """Load NPZ files in manifest order; relative paths are relative to the manifest."""
    paths = json.loads(manifest.read_text())
    if not isinstance(paths, list) or not paths or any(not isinstance(path, str) or not path for path in paths):
        raise ValueError("Manifest must be a nonempty JSON list of trajectory file paths")
    paths = [(Path(path) if Path(path).is_absolute() else manifest.parent / path).resolve() for path in paths]
    if len(set(paths)) != len(paths):
        raise ValueError("Manifest must not contain duplicate trajectory paths")
    trajectories = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            trajectories.append(Trajectory(time=archive["time"], state=archive["state"]))
    return trajectories, [str(path) for path in paths]


def main() -> None:
    args = parse_args()
    manifest, output = args.manifest_path.resolve(), args.output_dir.resolve()
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(output)
    trajectories, paths = load_trajectories(manifest)
    windows = windows_from_trajectories(
        trajectories, context_length=args.context_length, prediction_length=args.prediction_length,
        stride=args.stride, channels=args.channels,
    )
    length = args.context_length + args.prediction_length
    counts = [(len(trajectory.time) - length) // args.stride + 1 for trajectory in trajectories]
    metadata = {
        "format_version": 1,
        "file": "windows.npy",
        "shape": list(windows.shape),
        "dtype": str(windows.dtype),
        "settings": vars(args) | {"manifest_path": str(manifest), "output_dir": str(output)},
        "trajectory_paths": paths,
        "trajectory_offsets": [0, *np.cumsum(counts).tolist()],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Prepare both files before replacing the existing cache files individually.
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary) / "cache"
        staging.mkdir()
        np.save(staging / "windows.npy", windows, allow_pickle=False)
        (staging / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        output.mkdir(exist_ok=True)
        (staging / "windows.npy").replace(output / "windows.npy")
        (staging / "metadata.json").replace(output / "metadata.json")
    print(f"Saved {len(windows)} windows with shape {list(windows.shape)}: {output}")


if __name__ == "__main__":
    main()
