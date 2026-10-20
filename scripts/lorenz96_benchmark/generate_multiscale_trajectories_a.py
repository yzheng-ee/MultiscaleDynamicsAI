"""Generate full multiscale trajectories for training, validation, and Test A.

Run using the panda_pilot environment:
    python scripts/lorenz96_benchmark/generate_multiscale_trajectories_a.py

All trajectories reuse configuration A's calibrated bandwidth periods.
Each NPZ stores time and state on the fine grid, including both endpoints.
Four worker processes generate independent trajectories concurrently.
Only the main process writes each split's shared metadata.json list.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from math import ceil, log2
from multiprocessing import get_context
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.lorenz96 import generate_multiscale_trajectory


def generate_and_save_trajectory(split, index, seed, output_dir, sampling, duration):
    """Run one independent simulation; return metadata without transferring arrays."""
    trajectory_id = f"{split}_multiscale_{index:03d}"
    output_path = output_dir / f"{trajectory_id}.npz"
    n_slow_samples = sampling["n_slow_samples"]
    n_fast_samples = sampling["n_fast_samples"]
    fast_intervals_per_slow_interval = sampling["fast_intervals_per_slow_interval"]
    slow_dt = sampling["slow_dt"]
    fast_dt = sampling["fast_dt"]
    print(f"Generating {trajectory_id}, seed={seed}...", flush=True)
    started = perf_counter()
    trajectory = generate_multiscale_trajectory(
        K=9,
        J=8,
        hx=-0.8,
        hy=1.0,
        F=10.0,
        eps=2**-7,
        seed=seed,
        low=-5.0,
        high=10.0,
        sample_dt=fast_dt,
        burn_in=50.0,
        duration=duration,
        t0=0.0,
        endpoint=True,
        method="DOP853",
        rtol=1e-8,
        atol=1e-10,
        max_step=1000.0,  # Effectively uncapped: let the adaptive solver choose internal step sizes.
    )
    elapsed = perf_counter() - started

    slow_time = trajectory.time[::fast_intervals_per_slow_interval]
    if trajectory.time.shape != (n_fast_samples,) or trajectory.state.shape != (81, n_fast_samples):
        raise RuntimeError(f"Unexpected trajectory shape for {trajectory_id}.")
    if slow_time.size != n_slow_samples or slow_time[-1] != trajectory.time[-1]:
        raise RuntimeError(f"Slow downsampling count or endpoint mismatch for {trajectory_id}.")
    if not (
        np.isclose(trajectory.time[0], 50.0, rtol=0, atol=1e-12)
        and np.isclose(trajectory.time[-1], 50.0 + duration, rtol=0, atol=1e-12)
        and np.allclose(np.diff(trajectory.time), fast_dt, rtol=1e-7, atol=0)
        and np.allclose(np.diff(slow_time), slow_dt, rtol=1e-7, atol=0)
    ):
        raise RuntimeError(f"Unexpected sampling grid for {trajectory_id}.")

    np.savez_compressed(output_path, time=trajectory.time, state=trajectory.state)
    record = {
        "config_id": "config_a",
        "family": "multiscale",
        "trajectory_id": trajectory_id,
        "pair_id": f"{split}_pair_{index:03d}",
        "split": "test" if split == "test_a" else split,
        "test_subset": "test_a" if split == "test_a" else None,
        "filename": output_path.name,
        "parameters": {"K": 9, "J": 8, "hx": -0.8, "hy": 1.0, "F": 10.0, "eps": 2**-7},
        "settings": {
            "seed": seed,
            "low": -5.0,
            "high": 10.0,
            "sample_dt": fast_dt,
            "burn_in": 50.0,
            "duration": duration,
            "t0": 0.0,
            "endpoint": True,
            "method": "DOP853",
            "rtol": 1e-8,
            "atol": 1e-10,
            "max_step": 1000.0,
        },
        "sampling": sampling,
        "state_layout": "(K + K*J, N): slow first, fast in C order",
        "state_shape": list(trajectory.state.shape),
        "generation_seconds": elapsed,
    }
    print(f"Saved {output_path}; generation took {elapsed:.2f}s.", flush=True)
    return record


def main():
    bandwidth_path = (
        ROOT / "outputs/lorenz96_benchmark/dense_calibration_trajectory_a/bandwidth_periods.json"
    )
    bandwidth_record = json.loads(bandwidth_path.read_text())
    slow_bandwidth_period = float(bandwidth_record["slow_bandwidth_period"])
    fast_bandwidth_period = float(bandwidth_record["fast_bandwidth_period"])
    if not all(
        np.isfinite(period) and period > 0
        for period in (slow_bandwidth_period, fast_bandwidth_period)
    ):
        raise ValueError("Calibration bandwidth periods must be positive and finite.")

    # Derived values below correspond to the current configuration A calibration.
    n_slow_samples = 8192  # 8,192 slow samples
    period_ratio = slow_bandwidth_period / fast_bandwidth_period  # Approximately 90.4
    fast_intervals_per_slow_interval = 2 ** max(0, ceil(log2(period_ratio)))  # 128
    n_fast_intervals = (n_slow_samples - 1) * fast_intervals_per_slow_interval  # 1,048,448
    n_fast_samples = n_fast_intervals + 1  # 1,048,449
    slow_dt = slow_bandwidth_period / 32  # Approximately 0.013075313807531382 time units
    fast_dt = slow_dt / fast_intervals_per_slow_interval  # Approximately 0.00010215088912133892 time units
    duration = n_fast_intervals * fast_dt  # Approximately 107.09989539748955 time units

    # Disjoint initialization seeds, also distinct from calibration seed 42.
    splits = [
        ("training", "training", 32, 1001),
        ("validation", "validation", 8, 2001),
        ("test_a", "test/test_a", 16, 3001),
    ]
    for split, directory, count, first_seed in splits:
        output_dir = ROOT / "data/lorenz96_benchmark" / directory / "multiscale"
        if (output_dir / "metadata.json").exists() or any(output_dir.glob("*.npz")):
            raise FileExistsError(f"Trajectory outputs already exist in {output_dir}.")

    print(
        f"slow_dt={slow_dt:.17g}, fast_dt={fast_dt:.17g}, duration={duration:.17g}\n"
        f"Fine intervals per slow interval: {fast_intervals_per_slow_interval}; "
        f"saved samples per trajectory: {n_fast_samples:,}",
        flush=True,
    )
    sampling = {
        "bandwidth_source": str(bandwidth_path.relative_to(ROOT)),
        "cumulative_energy_cutoff": bandwidth_record["cumulative_energy_cutoff"],
        "slow_bandwidth_period": slow_bandwidth_period,
        "fast_bandwidth_period": fast_bandwidth_period,
        "slow_samples_per_bandwidth_period": 32,
        "slow_dt": slow_dt,
        "fast_dt": fast_dt,
        "fast_intervals_per_slow_interval": fast_intervals_per_slow_interval,
        "n_slow_samples": n_slow_samples,
        "n_fast_samples": n_fast_samples,
    }
    max_workers = 4  # Increase only when allocated CPU cores and memory permit.
    records_by_split = {split: [] for split, _, _, _ in splits}
    failures = []
    print(f"Using {max_workers} worker processes.", flush=True)
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=get_context("spawn")) as executor:
        futures = {}
        for split, directory, count, first_seed in splits:
            output_dir = ROOT / "data/lorenz96_benchmark" / directory / "multiscale"
            output_dir.mkdir(parents=True, exist_ok=True)
            for index in range(1, count + 1):
                future = executor.submit(
                    generate_and_save_trajectory,
                    split, index, first_seed + index - 1, output_dir, sampling, duration,
                )
                futures[future] = (split, index, output_dir)

        for future in as_completed(futures):
            split, index, output_dir = futures[future]
            try:
                record = future.result()
            except Exception as error:
                failures.append(f"{split}_multiscale_{index:03d}: {error}")
                print(f"Failed: {failures[-1]}", flush=True)
                continue
            records = records_by_split[split]
            records.append(record)
            records.sort(key=lambda item: item["trajectory_id"])
            # Only the main process writes shared metadata, after each completion.
            metadata_path = output_dir / "metadata.json"
            temporary_path = metadata_path.with_suffix(".json.tmp")
            temporary_path.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
            temporary_path.replace(metadata_path)

    if failures:
        raise RuntimeError("Trajectory generation failed:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
