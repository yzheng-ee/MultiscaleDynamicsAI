"""Generate the dense calibration trajectory for configuration A.

Configuration A is shared by training, validation, and Test A.

Run using the panda_pilot environment:
    python scripts/lorenz96_benchmark/generate_dense_calibration_trajectory_a.py

Settings are specified directly in the API call. Saves arrays and metadata;
Spin-up samples are excluded; both recorded endpoints are included.
The 99.9999-unit recorded span contains exactly 1,000,000 samples.
"""

import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.lorenz96 import generate_multiscale_trajectory


def main():
    output_dir = ROOT / "outputs/lorenz96_benchmark/dense_calibration_trajectory_a"
    output_path = output_dir / "dense_calibration_trajectory_a.npz"
    metadata_path = output_dir / "metadata.json"
    if output_path.exists() or metadata_path.exists():
        raise FileExistsError("Output already exists; choose a new output_dir before running.")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating the dense calibration trajectory for configuration A...", flush=True)
    started = perf_counter()
    trajectory = generate_multiscale_trajectory(
        K=9,
        J=8,
        hx=-0.8,
        hy=1.0,
        F=10.0,
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

    np.savez_compressed(output_path, time=trajectory.time, state=trajectory.state)
    metadata = {
        "config_id": "config_a",
        "filename": output_path.name,
        "parameters": {"K": 9, "J": 8, "hx": -0.8, "hy": 1.0, "F": 10.0, "eps": 2**-7},
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
        "state_layout": "(K + K*J, N): slow first, fast in C order",
        "state_shape": list(trajectory.state.shape),
        "generation_seconds": elapsed,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(f"Saved {trajectory.state.shape} to {output_path}", flush=True)
    print(f"Generation took {elapsed:.2f}s. Metadata: {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
