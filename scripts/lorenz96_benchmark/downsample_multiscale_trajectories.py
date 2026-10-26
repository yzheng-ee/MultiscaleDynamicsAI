"""Save the 8,192-point slow view of every multiscale forecasting trajectory.

Run using the panda_pilot environment:
    python scripts/lorenz96_benchmark/downsample_multiscale_trajectories.py

Outputs retain the original filenames under each multiscale/slow_downsampled/
directory, with a metadata.json list describing the derived views. Original
NPZ files and metadata are read only. No integration or filtering is applied:
the saved arrays are time[::multiplier] and state[:K, ::multiplier].
Dense calibration trajectories are preparation data and are not downsampled.
Rerunning replaces the derived NPZ files and rebuilds their metadata lists.
Four worker processes write independent NPZ files; only the parent process
writes metadata and updates the progress bar. A final summary reports totals.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
MAX_WORKERS = 4
OUTPUT_SUBDIRECTORY = "slow_downsampled"
COLLECTIONS = (
    ("training", 32),
    ("validation", 8),
    ("test/test_a", 16),
    ("test/test_b_k", 4),
    ("test/test_b_j", 4),
    ("test/test_b_hx", 4),
    ("test/test_b_hy", 4),
    ("test/test_b_f", 4),
)


def downsample_and_save(record, source_dir, output_dir):
    """Extract the exact slow grid, preserving both stored endpoints."""
    started = perf_counter()
    source_path = source_dir / record["filename"]
    output_path = output_dir / record["filename"]
    K = record["parameters"]["K"]
    multiplier = record["sampling"]["fast_intervals_per_slow_interval"]
    expected_count = 8191 * multiplier + 1
    if record["family"] != "multiscale" or not isinstance(multiplier, int) or multiplier < 1:
        raise ValueError(f"Invalid multiscale sampling metadata: {source_path}")
    with np.load(source_path, allow_pickle=False) as data:
        time, state = data["time"], data["state"]
    if (
        time.shape != (expected_count,)
        or state.shape != tuple(record["state_shape"])
        or state.shape != (K * (1 + record["parameters"]["J"]), expected_count)
        or time.dtype != np.float64
        or state.dtype != np.float64
    ):
        raise ValueError(f"Unexpected source shape or dtype: {source_path}")
    slow_time = time[::multiplier].copy()
    slow_state = state[:K, ::multiplier].copy()
    if (
        slow_time.shape != (8192,)
        or slow_state.shape != (K, 8192)
        or slow_time[0] != time[0]
        or slow_time[-1] != time[-1]
        or not np.isfinite(slow_time).all()
        or not np.isfinite(slow_state).all()
        or not np.allclose(np.diff(slow_time), record["sampling"]["slow_dt"], rtol=1e-7, atol=0)
    ):
        raise ValueError(f"Invalid derived slow grid: {source_path}")
    del time, state
    temporary_path = output_path.with_suffix(".npz.tmp")
    with temporary_path.open("wb") as file:
        np.savez_compressed(file, time=slow_time, state=slow_state)
    temporary_path.replace(output_path)
    elapsed = perf_counter() - started
    metadata = {
        "dataset_version": "v1.0",
        "config_id": record["config_id"],
        "family": "multiscale",
        "view": "slow_downsampled",
        "trajectory_id": record["trajectory_id"],
        "pair_id": record["pair_id"],
        "split": record["split"],
        "test_subset": record["test_subset"],
        "filename": output_path.name,
        "parameters": record["parameters"],
        "source_trajectory": str(source_path.relative_to(ROOT)),
        "source_metadata": str((source_dir / "metadata.json").relative_to(ROOT)),
        "downsampling": {
            "stride": multiplier,
            "channel_selection": "first K channels (slow variables)",
            "method": "direct selection without filtering",
        },
        "sampling": {
            "bandwidth_source": record["sampling"]["bandwidth_source"],
            "cumulative_energy_cutoff": record["sampling"]["cumulative_energy_cutoff"],
            "slow_bandwidth_period": record["sampling"]["slow_bandwidth_period"],
            "sample_dt": record["sampling"]["slow_dt"],
            "n_samples": 8192,
            "duration": record["settings"]["duration"],
            "endpoint": True,
        },
        "state_layout": "(K, N): slow variables of the full multiscale system",
        "state_shape": list(slow_state.shape),
        "time_range": [float(slow_time[0]), float(slow_time[-1])],
        "downsampling_seconds": elapsed,
    }
    return metadata


def main():
    started = perf_counter()
    collections = []
    # Check all sources before processing the first large file.
    for relative, count in COLLECTIONS:
        source_dir = ROOT / "data/lorenz96_benchmark" / relative / "multiscale"
        output_dir = source_dir / OUTPUT_SUBDIRECTORY
        records = json.loads((source_dir / "metadata.json").read_text())
        if len(records) != count or len({r["filename"] for r in records}) != count:
            raise ValueError(f"Unexpected trajectory count or duplicate filenames: {source_dir}")
        for record in records:
            if not (source_dir / record["filename"]).is_file():
                raise FileNotFoundError(source_dir / record["filename"])
        collections.append((source_dir, output_dir, records))
    total = sum(len(records) for _, _, records in collections)
    completed = 0
    failures = []
    records_by_directory = {}
    with tqdm(total=total, desc="Downsampling", unit="trajectory") as progress:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=get_context("spawn")) as executor:
            futures = {}
            for source_dir, output_dir, records in collections:
                output_dir.mkdir(parents=True, exist_ok=True)
                records_by_directory[output_dir] = []
                for record in sorted(records, key=lambda item: item["trajectory_id"]):
                    future = executor.submit(downsample_and_save, record, source_dir, output_dir)
                    futures[future] = (record["trajectory_id"], output_dir)
            for future in as_completed(futures):
                trajectory_id, output_dir = futures[future]
                try:
                    metadata = future.result()
                except Exception as error:
                    failures.append(f"{trajectory_id}: {error}")
                    progress.update(1)
                    continue
                derived_records = records_by_directory[output_dir]
                derived_records.append(metadata)
                derived_records.sort(key=lambda item: item["trajectory_id"])
                metadata_path = output_dir / "metadata.json"
                temporary_path = metadata_path.with_suffix(".json.tmp")
                temporary_path.write_text(json.dumps(derived_records, indent=2, allow_nan=False) + "\n")
                temporary_path.replace(metadata_path)
                completed += 1
                progress.update(1)
    elapsed = perf_counter() - started
    print(
        f"Downsampling finished: {completed}/{total} saved, {len(failures)} failed, "
        f"{len(collections)} collections, {MAX_WORKERS} workers, {elapsed:.2f}s elapsed.\n"
        f"Each output contains slow channels only with 8,192 samples per channel; "
        f"saved under multiscale/{OUTPUT_SUBDIRECTORY}/.",
        flush=True,
    )
    if failures:
        raise RuntimeError("Downsampling failed:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
