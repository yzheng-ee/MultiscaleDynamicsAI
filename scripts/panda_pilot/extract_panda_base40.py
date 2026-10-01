#!/usr/bin/env python3
"""Extract one system family from the raw Panda/base40 dataset.

Example:
    python scripts/panda_pilot/extract_panda_base40.py Lorenz \
        --output-dir data/panda_pilot/panda_lorenz63
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


DEFAULT_DATA_ROOT = Path("data/panda_pilot/panda_base40")
SOURCE_ID_PATTERN = re.compile(r"^(\d+)_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract every trajectory for one Panda system family. Each output directory contains trajectory.npz and metadata.json.")
    parser.add_argument("trajectory_name", help='Panda system-family name, for example "Lorenz" or "QiChen".')
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help=f"Directory containing Panda JSON and Parquet files (default: {DEFAULT_DATA_ROOT}).")
    parser.add_argument("--split", choices=("train", "test_zeroshot"), default="train", help="Dataset split to read (default: train).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory in which trajectory_0, trajectory_1, ... are created.")
    parser.add_argument("--overwrite", action="store_true", help="Replace files inside trajectory directories that already exist.")
    return parser.parse_args()


def source_id(source_filename: str) -> int | None:
    match = SOURCE_ID_PATTERN.match(source_filename)
    return int(match.group(1)) if match else None


def write_trajectory(
    output_dir: Path,
    trajectory: np.ndarray,
    metadata: dict[str, Any],
    overwrite: bool,
) -> None:
    npz_path = output_dir / "trajectory.npz"
    metadata_path = output_dir / "metadata.json"
    existing = [path for path in (npz_path, metadata_path) if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite {names}. Pass --overwrite to replace them."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, trajectory=trajectory)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    json_path = args.data_root / f"{args.split}.json"
    parquet_path = args.data_root / f"{args.split}.parquet"

    if not json_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {json_path}")
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Trajectory file not found: {parquet_path}")

    all_json_metadata = json.loads(json_path.read_text())
    family_json_metadata = all_json_metadata.get(args.trajectory_name, [])
    json_metadata_by_id = {
        record["sample_idx"]: record for record in family_json_metadata
    }

    table = pq.read_table(
        parquet_path,
        columns=[
            "start",
            "target._np_shape",
            "target",
            "_source_directory",
            "_source_filename",
        ],
    )

    selected_rows: list[dict[str, Any]] = []
    available_families: set[str] = set()
    for row_index in range(table.num_rows):
        family = table["_source_directory"][row_index].as_py()
        available_families.add(family)
        if family != args.trajectory_name:
            continue

        filename = table["_source_filename"][row_index].as_py()
        selected_rows.append(
            {
                "row_index": row_index,
                "source_id": source_id(filename),
                "source_filename": filename,
            }
        )

    if not selected_rows:
        choices = ", ".join(sorted(available_families))
        raise ValueError(
            f'No family named "{args.trajectory_name}" in the {args.split} split. '
            f"Available families: {choices}"
        )

    selected_rows.sort(
        key=lambda row: (
            row["source_id"] is None,
            row["source_id"] if row["source_id"] is not None else 0,
            row["source_filename"],
        )
    )

    matched_json_ids: set[int] = set()
    missing_json_ids: list[int | None] = []
    for output_index, selected in enumerate(selected_rows):
        row_index = selected["row_index"]
        shape = tuple(table["target._np_shape"][row_index].as_py())
        trajectory = np.asarray(
            table["target"][row_index].as_py(), dtype=np.float64
        ).reshape(shape)

        identifier = selected["source_id"]
        json_metadata = json_metadata_by_id.get(identifier)
        if json_metadata is None:
            missing_json_ids.append(identifier)
        else:
            matched_json_ids.add(identifier)

        # Both sections contain only fields present in the two raw Panda files.
        metadata = {
            "parquet": {
                "start": table["start"][row_index].as_py(),
                "target._np_shape": list(shape),
                "_source_directory": table["_source_directory"][row_index].as_py(),
                "_source_filename": selected["source_filename"],
            },
            "json": json_metadata,
        }
        write_trajectory(
            args.output_dir / f"trajectory_{output_index}",
            trajectory,
            metadata,
            args.overwrite,
        )

    unmatched_json_ids = sorted(set(json_metadata_by_id) - matched_json_ids)
    print(
        f'Extracted {len(selected_rows)} "{args.trajectory_name}" trajectories '
        f"from {args.split} to {args.output_dir}."
    )
    print(f"Matched JSON metadata: {len(matched_json_ids)}/{len(selected_rows)}")
    if missing_json_ids:
        print(f"Parquet source IDs without JSON metadata: {missing_json_ids}")
    if unmatched_json_ids:
        print(f"JSON sample_idx values without a Parquet trajectory: {unmatched_json_ids}")


if __name__ == "__main__":
    main()
