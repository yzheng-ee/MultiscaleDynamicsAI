"""Generate all six single-scale dense calibration trajectories sequentially.

Run using the panda_pilot environment:
    python scripts/lorenz96_benchmark/generate_singlescale_dense_calibration_trajectory.py

Loads the frozen closure without refitting. After 50 time units of spin-up,
saves 1,000,000 samples at spacing 0.0001, including both endpoints of the
99.9999-unit recorded span. Internal solver steps remain adaptive.
Configuration A is followed by all five Test B configurations.
Rerunning overwrites this script's outputs; files are replaced after writing
their temporary files successfully.
"""

import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import joblib
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.lorenz96 import (
    GPClosure,
    SingleScaleLorenz96,
    sample_initial_state,
    simulate_ode,
)


def check_configuration(setting, parameters):
    """Check closure identity and determine output paths before expensive integration."""
    output_dir = ROOT / "data/lorenz96_benchmark/dense_calibration/singlescale" / f"singlescale_dense_calibration_trajectory_{setting}"
    output_path = output_dir / f"singlescale_dense_calibration_trajectory_{setting}.npz"
    closure_dir = ROOT / "data/lorenz96_benchmark/closure" / setting
    closure_path = closure_dir / "closure.joblib"
    if not closure_path.is_file():
        raise FileNotFoundError(closure_path)
    metadata = json.loads((closure_dir / "metadata.json").read_text())
    if (
        metadata["config_id"] != f"config_{setting}"
        or metadata["parameters"] != parameters
        or metadata["family"] != "single_scale"
        or metadata["filename"] != closure_path.name
    ):
        raise ValueError(f"Closure configuration mismatch for {setting}.")
    return output_path, closure_path, metadata


def generate_and_save_trajectory(setting, parameters):
    """Generate one reduced calibration using its configuration's frozen GP."""
    output_path, closure_path, closure_metadata = check_configuration(setting, parameters)
    closure = joblib.load(closure_path)
    if not isinstance(closure, GPClosure):
        raise TypeError(f"Expected a saved GPClosure for {setting}.")
    model = SingleScaleLorenz96(
        closure=closure, K=parameters["K"], hx=parameters["hx"], F=parameters["F"],
    )
    settings = {
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
        # Output spacing need not cap internal steps for the reduced dynamics.
        "max_step": 1000.0,
    }
    initial_state = sample_initial_state(
        model, seed=settings["seed"], low=settings["low"], high=settings["high"],
    )
    started = perf_counter()
    trajectory = simulate_ode(
        model.rhs,
        initial_state=initial_state,
        **{key: value for key, value in settings.items() if key not in ("seed", "low", "high")},
    )
    elapsed = perf_counter() - started
    if (
        trajectory.time.shape != (1_000_000,)
        or trajectory.state.shape != (model.K, 1_000_000)
        or not np.isfinite(trajectory.state).all()
        or not np.isclose(trajectory.time[0], 50.0, rtol=0, atol=1e-12)
        or not np.isclose(trajectory.time[-1], 50.0 + settings["duration"], rtol=0, atol=1e-12)
        or not np.allclose(np.diff(trajectory.time), settings["sample_dt"], rtol=1e-7, atol=0)
    ):
        raise RuntimeError(f"Unexpected single-scale calibration shape or sampling grid for {setting}.")

    metadata = {
        "config_id": f"config_{setting}",
        "family": "single_scale",
        "filename": output_path.name,
        "parameters": parameters,
        "settings": settings,
        "closure": {
            "closure_id": closure_metadata["closure_id"],
            "source_closure": str(closure_path.relative_to(ROOT)),
            "source_metadata": str(closure_path.with_name("metadata.json").relative_to(ROOT)),
            "sha256": hashlib.sha256(closure_path.read_bytes()).hexdigest(),
            "definition": closure_metadata["definition"],
            "calibration": closure_metadata["calibration"],
        },
        "initialization": {
            "distribution": "uniform on [low, high), independently across slow sites",
            "initial_slow_state": initial_state.tolist(),
            "lineage": "same initial slow state in both calibration families for the same K and seed",
        },
        "state_layout": "(K, N): slow variables only, ordered by slow site",
        "state_shape": list(trajectory.state.shape),
        "time_range": [float(trajectory.time[0]), float(trajectory.time[-1])],
        "purpose": "single-scale bandwidth calibration; excluded from forecasting splits",
        "generation_seconds": elapsed,
    }
    metadata_text = json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".npz.tmp")
    with temporary_path.open("wb") as file:
        np.savez_compressed(file, time=trajectory.time, state=trajectory.state)
    temporary_path.replace(output_path)
    metadata_path = output_path.with_name("metadata.json")
    temporary_path = metadata_path.with_suffix(".json.tmp")
    temporary_path.write_text(metadata_text)
    temporary_path.replace(metadata_path)


def main():
    run_started = perf_counter()
    configurations = [
        # Setting, K, J, hx, hy, F
        ("a", 9, 8, -0.8, 1.0, 10.0),
        ("b_f", 9, 8, -0.8, 1.0, 14.0),
        ("b_hx", 9, 8, -1.2, 1.0, 10.0),
        ("b_hy", 9, 8, -0.8, 1.4, 10.0),
        ("b_j", 9, 16, -0.8, 1.0, 10.0),
        ("b_k", 18, 8, -0.8, 1.0, 10.0),
    ]
    jobs = [
        (setting, {"K": K, "J": J, "hx": hx, "hy": hy, "F": F, "eps": 2**-7})
        for setting, K, J, hx, hy, F in configurations
    ]
    # Check every closure before starting any integration.
    for setting, parameters in jobs:
        check_configuration(setting, parameters)
    for setting, parameters in tqdm(jobs, desc="Single-scale calibration", unit="configuration"):
        generate_and_save_trajectory(setting, parameters)

    print(f"Single-scale calibration finished: {len(jobs)} configurations saved in {perf_counter() - run_started:.2f}s.", flush=True)


if __name__ == "__main__":
    main()
