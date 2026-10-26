"""Generate all 76 full multiscale trajectories across training, validation and tests.

Run using the multiscale_dynamics environment:
    python scripts/lorenz96_benchmark/generate_multiscale_trajectories.py

Each configuration reuses its own calibrated bandwidth periods.
Each NPZ stores time and state on the fine grid, including both endpoints.
Four worker processes generate independent trajectories concurrently.
Only the main process writes each split/subset's shared metadata.json list.
Rerunning overwrites this script's outputs; files are replaced after writing
their temporary files successfully.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from math import ceil, log2
from multiprocessing import get_context
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.lorenz96 import (
    MultiscaleLorenz96,
    generate_multiscale_trajectory,
    sample_initial_state,
)


def load_configuration(subset, parameters):
    """Validate calibration provenance and derive the retained sampling grid."""
    calibration_dir = ROOT / "data/lorenz96_benchmark/dense_calibration/multiscale" / f"multiscale_dense_calibration_trajectory_{subset}"
    bandwidth_path = calibration_dir / "bandwidth_periods.json"
    bandwidth_record = json.loads(bandwidth_path.read_text())
    calibration_metadata = json.loads((calibration_dir / "metadata.json").read_text())
    if (
        calibration_metadata["parameters"] != parameters
        or calibration_metadata["config_id"] != f"config_{subset}"
        or bandwidth_record["config_id"] != f"config_{subset}"
        or bandwidth_record["cumulative_energy_cutoff"] != 0.99
    ):
        raise ValueError(f"Calibration configuration or cutoff mismatch for {subset}.")
    slow_bandwidth_period = float(bandwidth_record["slow_bandwidth_period"])
    fast_bandwidth_period = float(bandwidth_record["fast_bandwidth_period"])
    if not all(
        np.isfinite(period) and period > 0
        for period in (slow_bandwidth_period, fast_bandwidth_period)
    ):
        raise ValueError(f"Calibration bandwidth periods must be positive and finite for {subset}.")

    n_slow_samples = 8192  # 8,192 slow samples for every configuration.
    period_ratio = slow_bandwidth_period / fast_bandwidth_period
    fast_intervals_per_slow_interval = 2 ** max(0, ceil(log2(period_ratio)))  # Currently 128 for all six configurations.
    n_fast_intervals = (n_slow_samples - 1) * fast_intervals_per_slow_interval  # Currently 1,048,448.
    n_fast_samples = n_fast_intervals + 1  # Currently 1,048,449.
    slow_dt = slow_bandwidth_period / 32
    fast_dt = slow_dt / fast_intervals_per_slow_interval
    duration = n_fast_intervals * fast_dt
    sampling = {
        "bandwidth_source": str(bandwidth_path.relative_to(ROOT)),
        "cumulative_energy_cutoff": bandwidth_record["cumulative_energy_cutoff"],
        "channel_bandwidth_periods": bandwidth_record["channel_bandwidth_periods"],
        "slow_bandwidth_period": slow_bandwidth_period,
        "fast_bandwidth_period": fast_bandwidth_period,
        "slow_samples_per_bandwidth_period": 32,
        "slow_dt": slow_dt,
        "fast_dt": fast_dt,
        "fast_intervals_per_slow_interval": fast_intervals_per_slow_interval,
        "n_slow_samples": n_slow_samples,
        "n_fast_samples": n_fast_samples,
    }
    return sampling, duration


def generate_and_save_trajectory(split, index, seed, output_dir, parameters, sampling, duration, config_id):
    """Run one independent simulation; return metadata without transferring arrays."""
    trajectory_id = f"{split}_multiscale_{index:03d}"
    output_path = output_dir / f"{trajectory_id}.npz"
    n_slow_samples = sampling["n_slow_samples"]
    n_fast_samples = sampling["n_fast_samples"]
    fast_intervals_per_slow_interval = sampling["fast_intervals_per_slow_interval"]
    slow_dt = sampling["slow_dt"]
    fast_dt = sampling["fast_dt"]
    started = perf_counter()
    trajectory = generate_multiscale_trajectory(
        **parameters,
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
    n_channels = parameters["K"] * (1 + parameters["J"])
    if trajectory.time.shape != (n_fast_samples,) or trajectory.state.shape != (n_channels, n_fast_samples):
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

    temporary_path = output_path.with_suffix(".npz.tmp")
    with temporary_path.open("wb") as file:
        np.savez_compressed(file, time=trajectory.time, state=trajectory.state)
    temporary_path.replace(output_path)
    record = {
        "config_id": config_id,
        "family": "multiscale",
        "trajectory_id": trajectory_id,
        "pair_id": f"{split}_pair_{index:03d}",
        "split": "test" if split.startswith("test_") else split,
        "test_subset": split if split.startswith("test_") else None,
        "filename": output_path.name,
        "parameters": parameters,
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
        "initialization": {
            "distribution": "uniform on [low, high), independently across slow sites",
            "initial_slow_state": sample_initial_state(
                MultiscaleLorenz96(**parameters), seed=seed, low=-5.0, high=10.0,
            )[:parameters["K"]].tolist(),
            "lineage": "same initial slow state in both family members before spin-up",
        },
        "paired_singlescale_trajectory": {
            "trajectory_id": f"{split}_singlescale_{index:03d}",
            "source_metadata": str((output_dir.parent / "singlescale/metadata.json").relative_to(ROOT)),
            "source_trajectory": str((output_dir.parent / "singlescale" / f"{split}_singlescale_{index:03d}.npz").relative_to(ROOT)),
        },
        "state_layout": "(K + K*J, N): slow first, fast in C order",
        "state_shape": list(trajectory.state.shape),
        "time_range": [float(trajectory.time[0]), float(trajectory.time[-1])],
        "generation_seconds": elapsed,
    }
    return record


def main():
    run_started = perf_counter()
    # Split, output directory, configuration, K, J, hx, hy, F, count, first seed.
    # Preserve the original A/B split counts and disjoint initialization seeds.
    collections = [
        ("training", "training", "a", 9, 8, -0.8, 1.0, 10.0, 32, 1001),
        ("validation", "validation", "a", 9, 8, -0.8, 1.0, 10.0, 8, 2001),
        ("test_a", "test/test_a", "a", 9, 8, -0.8, 1.0, 10.0, 16, 3001),
        ("test_b_k", "test/test_b_k", "b_k", 18, 8, -0.8, 1.0, 10.0, 4, 4001),
        ("test_b_j", "test/test_b_j", "b_j", 9, 16, -0.8, 1.0, 10.0, 4, 5001),
        ("test_b_hx", "test/test_b_hx", "b_hx", 9, 8, -1.2, 1.0, 10.0, 4, 6001),
        ("test_b_hy", "test/test_b_hy", "b_hy", 9, 8, -0.8, 1.4, 10.0, 4, 7001),
        ("test_b_f", "test/test_b_f", "b_f", 9, 8, -0.8, 1.0, 14.0, 4, 8001),
    ]
    jobs = []
    for split, directory, subset, K, J, hx, hy, F, count, first_seed in collections:
        output_dir = ROOT / "data/lorenz96_benchmark" / directory / "multiscale"

        parameters = {"K": K, "J": J, "hx": hx, "hy": hy, "F": F, "eps": 2**-7}
        sampling, duration = load_configuration(subset, parameters)
        for index in range(1, count + 1):
            jobs.append((split, index, first_seed + index - 1, output_dir, parameters, sampling, duration, f"config_{subset}"))

    max_workers = 4  # Increase only when allocated CPU cores and memory permit.
    records_by_directory = {}
    failures = []
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=get_context("spawn")) as executor:
        futures = {}
        for job in jobs:
            split, index, seed, output_dir, parameters, sampling, duration, config_id = job
            output_dir.mkdir(parents=True, exist_ok=True)
            future = executor.submit(generate_and_save_trajectory, *job)
            futures[future] = (split, index, output_dir)

        for future in tqdm(as_completed(futures), total=len(futures), desc="Multiscale generation", unit="trajectory"):
            split, index, output_dir = futures[future]
            try:
                record = future.result()
            except Exception as error:
                failures.append(f"{split}_multiscale_{index:03d}: {error}")
                continue
            records = records_by_directory.setdefault(output_dir, [])
            records.append(record)
            records.sort(key=lambda item: item["trajectory_id"])
            # Only the main process writes shared metadata, after each completion.
            metadata_path = output_dir / "metadata.json"
            temporary_path = metadata_path.with_suffix(".json.tmp")
            temporary_path.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
            temporary_path.replace(metadata_path)

    print(f"Multiscale generation finished: {len(jobs) - len(failures)}/{len(jobs)} saved, {len(failures)} failed, {perf_counter() - run_started:.2f}s elapsed.", flush=True)
    if failures:
        raise RuntimeError("Trajectory generation failed:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
