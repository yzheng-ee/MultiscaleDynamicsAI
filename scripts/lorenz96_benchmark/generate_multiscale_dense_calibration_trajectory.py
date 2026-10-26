"""Generate all six multiscale dense calibration trajectories sequentially.

Run using the panda_pilot environment:
    python scripts/lorenz96_benchmark/generate_multiscale_dense_calibration_trajectory.py

Each configuration has its own directory, NPZ, and metadata.json.
Spin-up samples are excluded; both recorded endpoints are included.
The 99.9999-unit recorded span contains exactly 1,000,000 samples.
Rerunning overwrites this script's outputs; files are replaced after writing
their temporary files successfully.
"""

import json
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


def generate_and_save_trajectory(setting, parameters):
    """Generate and save one full multiscale dense calibration."""
    K, J, hx, hy, F = (parameters[name] for name in ("K", "J", "hx", "hy", "F"))
    output_dir = ROOT / "data/lorenz96_benchmark/dense_calibration/multiscale" / f"multiscale_dense_calibration_trajectory_{setting}"
    output_path = output_dir / f"multiscale_dense_calibration_trajectory_{setting}.npz"
    metadata_path = output_dir / "metadata.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    started = perf_counter()
    trajectory = generate_multiscale_trajectory(
        K=K,
        J=J,
        hx=hx,
        hy=hy,
        F=F,
        eps=2**-7,
        seed=42,
        low=-5.0,
        high=10.0,
        sample_dt=0.0001,
        burn_in=50.0,
        duration=100.0 - 0.0001,
        t0=0.0,
        endpoint=True,
        method="DOP853",
        rtol=1e-8,
        atol=1e-10,
    )
    elapsed = perf_counter() - started

    temporary_path = output_path.with_suffix(".npz.tmp")
    with temporary_path.open("wb") as file:
        np.savez_compressed(file, time=trajectory.time, state=trajectory.state)
    temporary_path.replace(output_path)
    metadata = {
        "config_id": f"config_{setting}",
        "family": "multiscale",
        "filename": output_path.name,
        "parameters": parameters,
        "settings": {
            "seed": 42,
            "low": -5.0,
            "high": 10.0,
            "sample_dt": 0.0001,
            "burn_in": 50.0,
            "duration": 100.0 - 0.0001,
            "t0": 0.0,
            "endpoint": True,
            "method": "DOP853",
            "rtol": 1e-8,
            "atol": 1e-10,
            "max_step": None,
        },
        "initialization": {
            "distribution": "uniform on [low, high), independently across slow sites",
            "initial_slow_state": sample_initial_state(
                MultiscaleLorenz96(K=K, J=J, hx=hx, hy=hy, F=F, eps=2**-7),
                seed=42, low=-5.0, high=10.0,
            )[:K].tolist(),
            "lineage": "same initial slow state in both calibration families for the same K and seed",
        },
        "state_layout": "(K + K*J, N): slow first, fast in C order",
        "state_shape": list(trajectory.state.shape),
        "time_range": [float(trajectory.time[0]), float(trajectory.time[-1])],
        "purpose": "closure fitting and multiscale bandwidth calibration; excluded from forecasting splits",
        "generation_seconds": elapsed,
    }
    temporary_path = metadata_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    temporary_path.replace(metadata_path)


def main():
    run_started = perf_counter()
    # A is followed by the five one-parameter Test B configurations.
    configurations = [
        # Setting, K, J, hx, hy, F
        ("a", 9, 8, -0.8, 1.0, 10.0),
        ("b_k", 18, 8, -0.8, 1.0, 10.0),
        ("b_j", 9, 16, -0.8, 1.0, 10.0),
        ("b_hx", 9, 8, -1.2, 1.0, 10.0),
        ("b_hy", 9, 8, -0.8, 1.4, 10.0),
        ("b_f", 9, 8, -0.8, 1.0, 14.0),
    ]
    jobs = [
        (setting, {"K": K, "J": J, "hx": hx, "hy": hy, "F": F, "eps": 2**-7})
        for setting, K, J, hx, hy, F in configurations
    ]
    for setting, parameters in tqdm(jobs, desc="Multiscale calibration", unit="configuration"):
        generate_and_save_trajectory(setting, parameters)

    print(f"Multiscale calibration finished: {len(jobs)} configurations saved in {perf_counter() - run_started:.2f}s.", flush=True)


if __name__ == "__main__":
    main()
