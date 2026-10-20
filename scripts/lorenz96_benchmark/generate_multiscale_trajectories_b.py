"""Generate full multiscale trajectories for the five Test B configurations.

Run using the panda_pilot environment:
    python scripts/lorenz96_benchmark/generate_multiscale_trajectories_b.py

Each configuration reuses its own calibrated bandwidth periods.
Each NPZ stores time and state on the fine grid, including both endpoints.
Four worker processes generate independent trajectories concurrently.
Only the main process writes each configuration's shared metadata.json list.
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


def generate_and_save_trajectory(subset, index, seed, output_dir, parameters, sampling, duration):
    """Run one independent simulation; return metadata without transferring arrays."""
    trajectory_id = f"test_{subset}_multiscale_{index:03d}"
    output_path = output_dir / f"{trajectory_id}.npz"
    n_slow_samples = sampling["n_slow_samples"]
    n_fast_samples = sampling["n_fast_samples"]
    fast_intervals_per_slow_interval = sampling["fast_intervals_per_slow_interval"]
    slow_dt = sampling["slow_dt"]
    fast_dt = sampling["fast_dt"]
    print(f"Generating {trajectory_id}, seed={seed}...", flush=True)
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

    np.savez_compressed(output_path, time=trajectory.time, state=trajectory.state)
    record = {
        "config_id": f"config_{subset}",
        "family": "multiscale",
        "trajectory_id": trajectory_id,
        "pair_id": f"test_{subset}_pair_{index:03d}",
        "split": "test",
        "test_subset": f"test_{subset}",
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
        "state_layout": "(K + K*J, N): slow first, fast in C order",
        "state_shape": list(trajectory.state.shape),
        "generation_seconds": elapsed,
    }
    print(f"Saved {output_path}; generation took {elapsed:.2f}s.", flush=True)
    return record


def main():
    # Distinct seed ranges, separate from configuration A and calibration seed 42.
    configurations = [
        # Subset, K, J, hx, hy, F, first seed
        ("b_k", 18, 8, -0.8, 1.0, 10.0, 4001),
        ("b_j", 9, 16, -0.8, 1.0, 10.0, 5001),
        ("b_hx", 9, 8, -1.2, 1.0, 10.0, 6001),
        ("b_hy", 9, 8, -0.8, 1.4, 10.0, 7001),
        ("b_f", 9, 8, -0.8, 1.0, 14.0, 8001),
    ]
    jobs = []
    for subset, K, J, hx, hy, F, first_seed in configurations:
        output_dir = ROOT / "data/lorenz96_benchmark/test" / f"test_{subset}" / "multiscale"
        if (output_dir / "metadata.json").exists() or any(output_dir.glob("*.npz")):
            raise FileExistsError(f"Trajectory outputs already exist in {output_dir}.")

        calibration_dir = ROOT / "outputs/lorenz96_benchmark" / f"dense_calibration_trajectory_{subset}"
        bandwidth_path = calibration_dir / "bandwidth_periods.json"
        bandwidth_record = json.loads(bandwidth_path.read_text())
        calibration_metadata = json.loads((calibration_dir / "metadata.json").read_text())
        parameters = {"K": K, "J": J, "hx": hx, "hy": hy, "F": F, "eps": 2**-7}
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
        fast_intervals_per_slow_interval = 2 ** max(0, ceil(log2(period_ratio)))  # Currently 128 for all five.
        n_fast_intervals = (n_slow_samples - 1) * fast_intervals_per_slow_interval  # Currently 1,048,448.
        n_fast_samples = n_fast_intervals + 1  # Currently 1,048,449.
        slow_dt = slow_bandwidth_period / 32
        fast_dt = slow_dt / fast_intervals_per_slow_interval
        duration = n_fast_intervals * fast_dt
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
        print(
            f"{subset}: slow_dt={slow_dt:.17g}, fast_dt={fast_dt:.17g}, duration={duration:.17g}\n"
            f"Fine intervals per slow interval: {fast_intervals_per_slow_interval}; "
            f"saved samples per trajectory: {n_fast_samples:,}",
            flush=True,
        )
        for index in range(1, 5):  # 4 trajectories per Test B configuration.
            jobs.append((subset, index, first_seed + index - 1, output_dir, parameters, sampling, duration))

    max_workers = 4  # Increase only when allocated CPU cores and memory permit.
    records_by_subset = {subset: [] for subset, *_ in configurations}
    failures = []
    print(f"Using {max_workers} worker processes for {len(jobs)} trajectories.", flush=True)
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=get_context("spawn")) as executor:
        futures = {}
        for job in jobs:
            subset, index, seed, output_dir, parameters, sampling, duration = job
            output_dir.mkdir(parents=True, exist_ok=True)
            future = executor.submit(generate_and_save_trajectory, *job)
            futures[future] = (subset, index, output_dir)

        for future in as_completed(futures):
            subset, index, output_dir = futures[future]
            try:
                record = future.result()
            except Exception as error:
                failures.append(f"test_{subset}_multiscale_{index:03d}: {error}")
                print(f"Failed: {failures[-1]}", flush=True)
                continue
            records = records_by_subset[subset]
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
