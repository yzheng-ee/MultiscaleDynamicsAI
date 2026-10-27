"""Collect trajectory file paths from one directory into a flat JSON list.

Example from the repository root:
    python scripts/panda/collect_trajectory_paths.py \
        --input-dir data/lorenz96_benchmark/training/multiscale/slow_downsampled \
        --output-path outputs/panda_lorenz96/trajectory_paths/multiscale_training.json --first-n 16

Files are selected directly from the input directory, without recursion or
metadata parsing, in filename order. Paths are saved as absolute strings.
Trajectory arrays are not loaded or modified. Rerunning replaces the manifest.
"""

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory directly containing trajectory files.")
    parser.add_argument("--extension", default=".npz", help="File suffix to collect, e.g. .npz or .npy.")
    parser.add_argument("--first-n", type=int, help="Collect the first N filenames; default: all matching files.")
    parser.add_argument("--output-path", type=Path, required=True, help="Destination JSON manifest path.")
    args = parser.parse_args()
    if args.first_n is not None and args.first_n <= 0:
        parser.error("--first-n must be positive")
    if not args.extension.startswith(".") or Path(args.extension).name != args.extension:
        parser.error("--extension must be a file suffix such as .npz")
    return args


def collect_trajectory_paths(input_dir: Path, *, extension: str = ".npz", first_n: int | None = None) -> list[str]:
    """Return absolute paths to the first N matching files in filename order.

    If fewer than N files exist, return all matches. An empty selection is an error.
    """
    if first_n is not None and first_n <= 0:
        raise ValueError("first_n must be positive")
    paths = sorted(
        (path for path in input_dir.iterdir() if path.is_file() and path.suffix == extension),
        key=lambda path: path.name,
    )
    if not paths:
        raise ValueError(f"No {extension} files found in {input_dir}")
    if first_n is not None:
        paths = paths[:first_n]
    return [str(path.resolve()) for path in paths]


def main() -> None:
    args = parse_args()
    paths = collect_trajectory_paths(args.input_dir.resolve(), extension=args.extension, first_n=args.first_n)
    output = args.output_path.resolve()
    temporary = output.with_name(output.name + ".tmp")
    if str(output) in paths or str(temporary) in paths:
        raise ValueError("Manifest destination must not overwrite a selected trajectory")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(paths, indent=2) + "\n")
    temporary.replace(output)
    print(f"Collected {len(paths)} trajectories: {output}")


if __name__ == "__main__":
    main()
