"""Generate the five dense calibration trajectories for Test B sequentially.

Run using the panda_pilot environment:
    python scripts/lorenz96_benchmark/generate_dense_calibration_trajectories_b.py

Each configuration has its own directory, NPZ, and metadata.json.
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
    # Each Test B configuration changes one parameter from configuration A.
    configurations = [
        # Subset, K, J, hx, hy, F
        ("b_k", 18, 8, -0.8, 1.0, 10.0),
        ("b_j", 9, 16, -0.8, 1.0, 10.0),
        ("b_hx", 9, 8, -1.2, 1.0, 10.0),
        ("b_hy", 9, 8, -0.8, 1.4, 10.0),
        ("b_f", 9, 8, -0.8, 1.0, 14.0),
    ]
    # Check all destinations before starting any expensive simulation.
    for subset, K, J, hx, hy, F in configurations:
        output_dir = ROOT / "outputs/lorenz96_benchmark" / f"dense_calibration_trajectory_{subset}"
        output_path = output_dir / f"dense_calibration_trajectory_{subset}.npz"
        metadata_path = output_dir / "metadata.json"
        if output_path.exists() or metadata_path.exists():
            raise FileExistsError(f"Output already exists in {output_dir}.")

    for subset, K, J, hx, hy, F in configurations:
        output_dir = ROOT / "outputs/lorenz96_benchmark" / f"dense_calibration_trajectory_{subset}"
        output_path = output_dir / f"dense_calibration_trajectory_{subset}.npz"
        metadata_path = output_dir / "metadata.json"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generating the dense calibration trajectory for configuration {subset.upper()}...", flush=True)
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

        np.savez_compressed(output_path, time=trajectory.time, state=trajectory.state)
        metadata = {
            "config_id": f"config_{subset}",
            "filename": output_path.name,
            "parameters": {"K": K, "J": J, "hx": hx, "hy": hy, "F": F, "eps": 2**-7},
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
        del trajectory


if __name__ == "__main__":
    main()
