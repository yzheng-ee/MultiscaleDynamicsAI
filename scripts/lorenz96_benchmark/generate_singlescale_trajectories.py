"""Generate all 76 single-scale trajectories paired with the multiscale data.

Run using the multiscale_dynamics environment:
    python scripts/lorenz96_benchmark/generate_singlescale_trajectories.py

Each configuration loads its frozen GP closure and single-scale bandwidth
period. Saves exactly 8,192 observations, including both endpoints, with
sample_dt = bandwidth_period / 32 and duration = 8191 * sample_dt.
Initialization seeds, pair IDs and splits match the existing multiscale
metadata. Four worker processes integrate independent trajectories; only the
main process writes each split/subset's shared metadata.json list.
Rerunning overwrites this script's outputs; files are replaced after writing
their temporary files successfully.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import hashlib
import json
from multiprocessing import get_context
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


@lru_cache(maxsize=6)
def load_closure(path, sha256):
    """Load each frozen closure once per worker and verify its identity."""
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
        raise ValueError(f"Closure checksum mismatch: {path}")
    closure = joblib.load(path)
    if not isinstance(closure, GPClosure):
        raise TypeError(f"Expected a saved GPClosure: {path}")
    return closure


def load_configuration(subset, parameters):
    """Validate closure/calibration provenance and derive the retained grid."""
    config_id = f"config_{subset}"
    closure_dir = ROOT / "data/lorenz96_benchmark/closure" / subset
    closure_path = closure_dir / "closure.joblib"
    closure_metadata_path = closure_dir / "metadata.json"
    closure_metadata = json.loads(closure_metadata_path.read_text())
    calibration_dir = ROOT / "data/lorenz96_benchmark/dense_calibration/singlescale" / f"singlescale_dense_calibration_trajectory_{subset}"
    calibration_metadata = json.loads((calibration_dir / "metadata.json").read_text())
    bandwidth_path = calibration_dir / "bandwidth_periods.json"
    bandwidth = json.loads(bandwidth_path.read_text())
    sha256 = hashlib.sha256(closure_path.read_bytes()).hexdigest()
    if (
        closure_metadata["config_id"] != config_id
        or closure_metadata["family"] != "single_scale"
        or closure_metadata["parameters"] != parameters
        or closure_metadata["filename"] != closure_path.name
        or calibration_metadata["config_id"] != config_id
        or calibration_metadata["family"] != "single_scale"
        or calibration_metadata["parameters"] != parameters
        or calibration_metadata["closure"]["closure_id"] != closure_metadata["closure_id"]
        or calibration_metadata["closure"]["sha256"] != sha256
        or bandwidth["config_id"] != config_id
        or bandwidth["source_trajectory"] != calibration_metadata["filename"]
        or bandwidth["source_metadata"] != "metadata.json"
        or bandwidth["cumulative_energy_cutoff"] != 0.99
    ):
        raise ValueError(f"Closure or bandwidth-calibration provenance mismatch for {subset}.")
    period = bandwidth["bandwidth_period"]
    if period is None or not np.isfinite(period) or period <= 0:
        raise ValueError(f"Single-scale bandwidth period must be positive and finite for {subset}.")
    if len(bandwidth["channel_bandwidth_periods"]) != parameters["K"]:
        raise ValueError(f"Channel bandwidth-period count mismatch for {subset}.")
    load_closure(closure_path, sha256)
    sample_dt = float(period) / 32
    duration = 8191 * sample_dt
    sampling = {
        "bandwidth_source": str(bandwidth_path.relative_to(ROOT)),
        "cumulative_energy_cutoff": bandwidth["cumulative_energy_cutoff"],
        "bandwidth_period": float(period),
        "channel_bandwidth_periods": bandwidth["channel_bandwidth_periods"],
        "samples_per_bandwidth_period": 32,
        "sample_dt": sample_dt,
        "n_samples": 8192,
    }
    closure_record = {
        "closure_id": closure_metadata["closure_id"],
        "source_closure": str(closure_path.relative_to(ROOT)),
        "source_metadata": str(closure_metadata_path.relative_to(ROOT)),
        "sha256": sha256,
        "definition": closure_metadata["definition"],
        "calibration": closure_metadata["calibration"],
    }
    return sampling, duration, closure_record


def generate_and_save_trajectory(partner, partner_metadata_path, output_dir, sampling, duration, closure_record):
    """Integrate one paired reduced system and return its metadata record."""
    trajectory_id = partner["trajectory_id"].replace("_multiscale_", "_singlescale_")
    output_path = output_dir / f"{trajectory_id}.npz"
    parameters = partner["parameters"]
    closure_path = ROOT / closure_record["source_closure"]
    model = SingleScaleLorenz96(
        closure=load_closure(closure_path, closure_record["sha256"]),
        K=parameters["K"],
        hx=parameters["hx"],
        F=parameters["F"],
    )
    # Inherit initialization and numerical controls from the multiscale partner.
    settings = {
        key: partner["settings"][key]
        for key in ("seed", "low", "high", "burn_in", "t0", "method", "rtol", "atol", "max_step")
    }
    settings.update(sample_dt=sampling["sample_dt"], duration=duration, endpoint=True)
    initial_state = sample_initial_state(model, seed=settings["seed"], low=settings["low"], high=settings["high"])
    started = perf_counter()
    trajectory = simulate_ode(
        model.rhs,
        initial_state=initial_state,
        **{key: value for key, value in settings.items() if key not in ("seed", "low", "high")},
    )
    elapsed = perf_counter() - started
    start = settings["t0"] + settings["burn_in"]
    if (
        trajectory.time.shape != (8192,)
        or trajectory.state.shape != (model.K, 8192)
        or not np.isfinite(trajectory.state).all()
        or not np.isclose(trajectory.time[0], start, rtol=0, atol=1e-12)
        or not np.isclose(trajectory.time[-1], start + duration, rtol=0, atol=1e-12)
        or not np.allclose(np.diff(trajectory.time), sampling["sample_dt"], rtol=1e-7, atol=0)
    ):
        raise RuntimeError(f"Unexpected single-scale trajectory shape or sampling grid for {trajectory_id}.")
    record = {
        "config_id": partner["config_id"],
        "family": "single_scale",
        "trajectory_id": trajectory_id,
        "pair_id": partner["pair_id"],
        "split": partner["split"],
        "test_subset": partner["test_subset"],
        "filename": output_path.name,
        "parameters": parameters,
        "settings": settings,
        "sampling": sampling,
        "closure": closure_record,
        "initialization": {
            "distribution": "uniform on [low, high), independently across slow sites",
            "initial_slow_state": initial_state.tolist(),
            "lineage": "same initial slow state in both family members before spin-up",
        },
        "paired_multiscale_trajectory": {
            "trajectory_id": partner["trajectory_id"],
            "source_metadata": str(partner_metadata_path.relative_to(ROOT)),
            "source_trajectory": str((partner_metadata_path.parent / partner["filename"]).relative_to(ROOT)),
        },
        "state_layout": "(K, N): slow variables only, ordered by slow site",
        "state_shape": list(trajectory.state.shape),
        "time_range": [float(trajectory.time[0]), float(trajectory.time[-1])],
        "generation_seconds": elapsed,
    }
    # Check that the metadata can be serialized before saving the trajectory.
    json.dumps(record, allow_nan=False)
    temporary_path = output_path.with_suffix(".npz.tmp")
    with temporary_path.open("wb") as file:
        np.savez_compressed(file, time=trajectory.time, state=trajectory.state)
    temporary_path.replace(output_path)
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
    configurations = {}
    for split, directory, subset, K, J, hx, hy, F, count, first_seed in collections:
        output_dir = ROOT / "data/lorenz96_benchmark" / directory / "singlescale"
        parameters = {"K": K, "J": J, "hx": hx, "hy": hy, "F": F, "eps": 2**-7}
        if subset not in configurations:
            configurations[subset] = load_configuration(subset, parameters)
        sampling, duration, closure_record = configurations[subset]
        partner_metadata_path = output_dir.parent / "multiscale/metadata.json"
        partners = json.loads(partner_metadata_path.read_text())
        by_id = {partner["trajectory_id"]: partner for partner in partners}
        if len(partners) != count or len(by_id) != count:
            raise ValueError(f"Unexpected multiscale trajectory count or duplicate IDs for {split}.")
        for index in range(1, count + 1):
            partner = by_id[f"{split}_multiscale_{index:03d}"]
            if (
                partner["family"] != "multiscale"
                or partner["config_id"] != f"config_{subset}"
                or partner["parameters"] != parameters
                or partner["pair_id"] != f"{split}_pair_{index:03d}"
                or partner["split"] != ("test" if split.startswith("test_") else split)
                or partner["test_subset"] != (split if split.startswith("test_") else None)
                or partner["settings"]["seed"] != first_seed + index - 1
                or partner["settings"]["burn_in"] < 50.0
            ):
                raise ValueError(f"Multiscale pairing mismatch for {split}, index {index}.")
            if not (partner_metadata_path.parent / partner["filename"]).is_file():
                raise FileNotFoundError(partner_metadata_path.parent / partner["filename"])
            jobs.append((partner, partner_metadata_path, output_dir, sampling, duration, closure_record))

    max_workers = 4
    records_by_directory = {}
    failures = []
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=get_context("spawn")) as executor:
        futures = {}
        for job in jobs:
            output_dir = job[2]
            output_dir.mkdir(parents=True, exist_ok=True)
            future = executor.submit(generate_and_save_trajectory, *job)
            futures[future] = (job[0]["trajectory_id"], output_dir)

        for future in tqdm(as_completed(futures), total=len(futures), desc="Single-scale generation", unit="trajectory"):
            partner_id, output_dir = futures[future]
            try:
                record = future.result()
            except Exception as error:
                failures.append(f"{partner_id}: {error}")
                continue
            records = records_by_directory.setdefault(output_dir, [])
            records.append(record)
            records.sort(key=lambda item: item["trajectory_id"])
            # Only the main process writes shared metadata, after each completion.
            metadata_path = output_dir / "metadata.json"
            temporary_path = metadata_path.with_suffix(".json.tmp")
            temporary_path.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
            temporary_path.replace(metadata_path)

    print(f"Single-scale generation finished: {len(jobs) - len(failures)}/{len(jobs)} saved, {len(failures)} failed, {perf_counter() - run_started:.2f}s elapsed.", flush=True)
    if failures:
        raise RuntimeError("Single-scale trajectory generation failed:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
