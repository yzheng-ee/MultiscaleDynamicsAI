"""Read-only validation of the complete paired Lorenz–96 benchmark dataset.

Run from the repository root using environments/lorenz96_benchmark.yml:
    python scripts/lorenz96_benchmark/validate_dataset.py

Writes data/lorenz96_benchmark/validation_report.json.
Checks both forecasting families, dense calibrations, bandwidth records,
frozen closures and their seeded training pairs, and pairing/provenance.
Only the validation report is written; generated data and metadata are
unchanged. Four worker processes validate independent trajectories; each loads one trajectory at a time and evaluates FFTs in
batches of eight channels. Configuration constants are defined below.
Reports include elapsed seconds for every trajectory and calibration check.

For a partial check, set SELECTED_COLLECTIONS and/or PAIR_LIMIT below.
The report explicitly identifies partial runs. Exit codes: 0 = pass, 1 = failure,
2 = statistical observations requiring review. Screening thresholds are
diagnostics, not rigorous error bounds or proofs of chaotic dynamics.
Rerunning replaces the existing validation report.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from math import ceil, log2
from multiprocessing import get_context
from pathlib import Path
import platform
import sys
from time import perf_counter

import joblib
import numpy as np
import scipy
import sklearn
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.lorenz96 import (
    GPClosure,
    MultiscaleLorenz96,
    SingleScaleLorenz96,
    Trajectory,
    sample_closure_training_data,
    sample_initial_state,
    simulate_ode,
)
from src.data.sampling_interval import estimate_trajectory_bandwidths


# Script settings. Leave selection and limit as None for the full dataset.
MAX_WORKERS = 4
SELECTED_COLLECTIONS = None  # Example: ("test_b_f",)
PAIR_LIMIT = None          # Example: 1 (one trajectory from each family)
REINTEGRATION_TOLERANCE = 1e-6
REPORT_PATH = ROOT / "data/lorenz96_benchmark/validation_report.json"


# Label, directory, configuration, expected count, first initialization seed.
COLLECTIONS = (
    ("training", "training", "a", 32, 1001),
    ("validation", "validation", "a", 8, 2001),
    ("test_a", "test/test_a", "a", 16, 3001),
    ("test_b_k", "test/test_b_k", "b_k", 4, 4001),
    ("test_b_j", "test/test_b_j", "b_j", 4, 5001),
    ("test_b_hx", "test/test_b_hx", "b_hx", 4, 6001),
    ("test_b_hy", "test/test_b_hy", "b_hy", 4, 7001),
    ("test_b_f", "test/test_b_f", "b_f", 4, 8001),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def group_statistics(state):
    """Pooled and per-channel statistics without a full-size scratch array."""
    means = np.array([row.mean() for row in state])
    stds = np.array([row.std() for row in state])
    require(np.isfinite(means).all() and np.isfinite(stds).all(), "Nonfinite state statistics.")
    mean = float(means.mean())
    std = float(np.sqrt(max(0.0, np.mean(stds**2 + means**2) - mean**2)))
    blocks = []
    for block in np.array_split(state, 4, axis=1):
        block_means = np.array([row.mean() for row in block])
        block_stds = np.array([row.std() for row in block])
        block_mean = float(block_means.mean())
        block_std = float(np.sqrt(max(0.0, np.mean(block_stds**2 + block_means**2) - block_mean**2)))
        blocks.append({
            "mean": block_mean,
            "std": block_std,
        })
    block_means = [b["mean"] for b in blocks]
    block_stds = [b["std"] for b in blocks]
    return {
        "min": float(state.min()),
        "max": float(state.max()),
        "mean": mean,
        "std": std,
        "channel_means": means.tolist(),
        "channel_stds": stds.tolist(),
        "near_constant_channels": np.flatnonzero(stds <= 1e-10).tolist(),
        "time_blocks": blocks,
        "block_mean_spread_in_std": (max(block_means) - min(block_means)) / std if std > 0 else None,
        "block_std_ratio": max(block_stds) / min(block_stds) if min(block_stds) > 0 else None,
    }


def state_digest(state):
    """Hash logical channel/time order, regardless of array memory layout."""
    digest = hashlib.sha256()
    for row in state:
        digest.update(memoryview(np.ascontiguousarray(row)).cast("B"))
    return digest.hexdigest()


def statistical_observations(statistics, reference, group):
    """Flag large departures for inspection; ordinary variation is reported."""
    observations = []
    if statistics["near_constant_channels"]:
        observations.append(f"{group}: near-constant channels {statistics['near_constant_channels']} (group-local indices).")
    if statistics["block_mean_spread_in_std"] is not None and statistics["block_mean_spread_in_std"] > 0.5:
        observations.append(f"{group}: time-block mean spread exceeds 0.5 pooled standard deviations.")
    if statistics["block_std_ratio"] is None or statistics["block_std_ratio"] > 1.5:
        observations.append(f"{group}: time-block standard deviations vary by more than a factor of 1.5 or include zero.")
    if reference is None:
        return observations
    reference_std = reference["std"]
    if reference_std > 0:
        statistics["mean_shift_in_calibration_std"] = (statistics["mean"] - reference["mean"]) / reference_std
        statistics["std_to_calibration_ratio"] = statistics["std"] / reference_std
        if abs(statistics["mean_shift_in_calibration_std"]) > 0.5:
            observations.append(f"{group}: mean differs from calibration by more than 0.5 calibration standard deviations.")
        if not 0.5 <= statistics["std_to_calibration_ratio"] <= 2.0:
            observations.append(f"{group}: standard deviation is outside 0.5–2 times calibration.")
    reference_amplitude = max(abs(reference["min"]), abs(reference["max"]))
    if reference_amplitude > 0 and max(abs(statistics["min"]), abs(statistics["max"])) > 5 * reference_amplitude:
        observations.append(f"{group}: amplitude exceeds five times the calibration maximum absolute value.")
    return observations


def check_reintegration(time, state, model, fast_dt, tolerance):
    """Restart three 16-interval windows at saved full coupled states."""
    intervals = 16
    last_start = time.size - intervals - 1
    starts = (0, last_start // 2, last_start)
    windows = []
    discrepancies = []
    for index in starts:
        restarted = simulate_ode(
            model.rhs, state[:, index], t0=float(time[index]),
            duration=intervals * fast_dt, sample_dt=fast_dt, endpoint=True,
            method="DOP853", rtol=1e-10, atol=1e-12, max_step=fast_dt / 2,
        )
        saved = state[:, index:index + intervals + 1]
        difference = np.abs(restarted.state - saved)
        normalized = difference / (1 + np.abs(saved))
        entry = {
            "start_index": int(index),
            "start_time": float(time[index]),
            "max_absolute_error": float(difference.max()),
            "max_scaled_error": float(normalized.max()),
            "rms_scaled_error": float(np.sqrt(np.mean(normalized**2))),
        }
        windows.append(entry)
        if entry["max_scaled_error"] > tolerance:
            discrepancies.append(f"Window at index {index}: maximum |error|/(1+|saved|) exceeds {tolerance:g}.")
    return windows, discrepancies


def estimate_periods(time, state):
    """Use the established estimator in small channel batches."""
    bandwidths = []
    for first in range(0, state.shape[0], 8):
        bandwidths.extend(estimate_trajectory_bandwidths(Trajectory(time, state[first:first + 8]), cumulative_energy_cutoff=0.99))
    bandwidths = np.asarray(bandwidths)
    periods = np.full(bandwidths.shape, np.inf)
    np.divide(1.0, bandwidths, out=periods, where=bandwidths > 0)
    return periods


def check_bandwidths(time, state, K, sampling, family="multiscale"):
    periods = estimate_periods(time, state)
    result = {"channel_bandwidth_periods": [float(p) if np.isfinite(p) else None for p in periods]}
    observations = []
    groups = (("slow", periods[:K]), ("fast", periods[K:])) if family == "multiscale" else (("state", periods),)
    for group, selected in groups:
        period = float(np.median(selected))
        calibrated_period = sampling["bandwidth_period"] if family == "single_scale" else sampling[f"{group}_bandwidth_period"]
        ratio = period / calibrated_period
        dt = sampling["sample_dt"] if family == "single_scale" else sampling["slow_dt"] if group == "slow" else sampling["fast_dt"]
        result[group] = {
            "bandwidth_period": period if np.isfinite(period) else None,
            "measured_to_calibrated_period_ratio": ratio if np.isfinite(ratio) else None,
            "samples_per_measured_period": period / dt if np.isfinite(period) else None,
            "zero_bandwidth_channels": int(np.isinf(selected).sum()),
        }
        if not np.isfinite(ratio) or not 0.5 <= ratio <= 2.0:
            observations.append(f"{group}: measured median period is unbounded or outside 0.5–2 times calibration.")
    return result, observations


def validate_multiscale_trajectory(record, directory, calibration, reintegration_tolerance):
    """Validate one file; structural errors raise ValueError to the caller."""
    model = MultiscaleLorenz96(**record["parameters"])
    sampling = record["sampling"]
    settings = record["settings"]
    require(record["family"] == "multiscale", "Wrong trajectory family.")
    require(record["config_id"] == calibration["config_id"], "Calibration configuration ID mismatch.")
    require(record["config_id"] == calibration["bandwidth"]["config_id"], "Bandwidth configuration ID mismatch.")
    require(record["parameters"] == calibration["parameters"], "Calibration physical parameters mismatch.")
    for key in ("slow_bandwidth_period", "fast_bandwidth_period", "channel_bandwidth_periods", "cumulative_energy_cutoff"):
        require(sampling[key] == calibration["bandwidth"][key], f"Calibration mismatch: {key}.")
    require(sampling["cumulative_energy_cutoff"] == 0.99, "Unexpected AC-energy cutoff.")
    require(len(sampling["channel_bandwidth_periods"]) == model.state_dimension, "Channel bandwidth-period count mismatch.")
    require(sampling["slow_samples_per_bandwidth_period"] == 32, "Expected 32 samples per calibrated slow period.")
    slow_period, fast_period = sampling["slow_bandwidth_period"], sampling["fast_bandwidth_period"]
    require(all(np.isfinite(p) and p > 0 for p in (slow_period, fast_period)), "Invalid calibration periods.")
    multiplier = 2 ** max(0, ceil(log2(slow_period / fast_period)))
    require(sampling["fast_intervals_per_slow_interval"] == multiplier, "Fine-grid multiplier mismatch.")
    fast_dt = sampling["fast_dt"]
    require(np.isclose(sampling["slow_dt"], slow_period / 32, rtol=1e-12, atol=0), "Slow spacing mismatch.")
    require(np.isclose(fast_dt, sampling["slow_dt"] / multiplier, rtol=1e-12, atol=0), "Fine spacing mismatch.")
    expected_samples = 8191 * multiplier + 1
    require(sampling["n_slow_samples"] == 8192 and sampling["n_fast_samples"] == expected_samples, "Metadata sample count mismatch.")
    require(settings["endpoint"] is True, "Both endpoints must be included.")
    require(settings["sample_dt"] == fast_dt, "Solver/output spacing mismatch.")
    require(np.isclose(settings["duration"], (expected_samples - 1) * fast_dt, rtol=1e-12, atol=0), "Duration mismatch.")
    with np.load(directory / record["filename"], allow_pickle=False) as data:
        require(set(data.files) == {"time", "state"}, "Expected exactly time/state arrays.")
        time, state = data["time"], data["state"]
    require(time.dtype == np.float64 and state.dtype == np.float64, "Expected float64 arrays.")
    require(time.shape == (expected_samples,), "Stored time shape mismatch.")
    require(state.shape == (model.state_dimension, expected_samples) == tuple(record["state_shape"]), "Stored state shape mismatch.")
    require(np.isfinite(time).all(), "Nonfinite timestamps.")
    for row in state:
        require(np.isfinite(row).all(), "Nonfinite state values.")
    differences = np.diff(time)
    require(np.all(differences > 0), "Timestamps are not strictly increasing.")
    require(np.allclose(differences, fast_dt, rtol=1e-7, atol=0), "Nonuniform fine timestamps.")
    start = settings["t0"] + settings["burn_in"]
    require(np.allclose(time[[0, -1]], [start, start + settings["duration"]], rtol=0, atol=1e-12), "Endpoint mismatch.")
    require(np.allclose(record["time_range"], time[[0, -1]], rtol=0, atol=1e-12), "Recorded time range mismatch.")
    slow_time = time[::multiplier]
    require(slow_time.size == 8192 and slow_time[0] == time[0] and slow_time[-1] == time[-1], "Derived slow grid count/endpoints mismatch.")
    require(np.allclose(np.diff(slow_time), sampling["slow_dt"], rtol=1e-7, atol=0), "Derived slow grid spacing mismatch.")

    statistics = {
        "slow": group_statistics(state[:model.K]),
        "fast": group_statistics(state[model.K:]),
    }
    observations = []
    for group in ("slow", "fast"):
        observations.extend(statistical_observations(statistics[group], calibration["statistics"][group], group))
    try:
        windows, discrepancies = check_reintegration(time, state, model, fast_dt, reintegration_tolerance)
    except (RuntimeError, ValueError) as error:
        windows, discrepancies = [], [f"Short-window reintegration failed: {error}"]
    bandwidths, bandwidth_observations = check_bandwidths(time, state, model.K, sampling)
    observations.extend(bandwidth_observations)
    return {
        "family": "multiscale",
        "trajectory_id": record["trajectory_id"],
        "pair_id": record["pair_id"],
        "config_id": record["config_id"],
        "filename": str((directory / record["filename"]).relative_to(ROOT)),
        "structural_errors": [],
        "numerical_discrepancies": discrepancies,
        "statistical_observations": observations,
        "structural_checks_passed": True,
        "state_shape": list(state.shape),
        "time_range": [float(time[0]), float(time[-1])],
        "derived_slow_samples": int(slow_time.size),
        "state_sha256": state_digest(state),
        "statistics": statistics,
        "reintegration_windows": windows,
        "bandwidths": bandwidths,
    }



def validate_references(value):
    """Check repository-relative provenance without changing any records."""
    if isinstance(value, dict):
        for entry in value.values():
            validate_references(entry)
    elif isinstance(value, list):
        for entry in value:
            validate_references(entry)
    elif isinstance(value, str) and value.startswith("data/lorenz96_benchmark/"):
        require((ROOT / value).is_file(), f"Missing provenance file: {value}")


def check_arrays(time, state, record, dimension, count, dt):
    require(time.dtype == np.float64 and state.dtype == np.float64, "Expected float64 arrays.")
    require(time.shape == (count,), "Stored time shape mismatch.")
    require(state.shape == (dimension, count) == tuple(record["state_shape"]), "Stored state shape mismatch.")
    require(np.isfinite(time).all(), "Nonfinite timestamps.")
    for row in state:
        require(np.isfinite(row).all(), "Nonfinite state values.")
    require(np.all(np.diff(time) > 0), "Timestamps must increase.")
    require(np.allclose(np.diff(time), dt, rtol=1e-7, atol=0), "Nonuniform timestamps.")
    settings = record["settings"]
    require(settings["endpoint"] is True, "Both endpoints must be included.")
    require(np.isclose(settings["duration"], (count - 1) * dt, rtol=1e-12, atol=0), "Duration mismatch.")
    start = settings["t0"] + settings["burn_in"]
    require(np.allclose(time[[0, -1]], [start, start + settings["duration"]], rtol=0, atol=1e-12), "Endpoint mismatch.")
    require(np.allclose(record["time_range"], time[[0, -1]], rtol=0, atol=1e-12), "Recorded time range mismatch.")


@lru_cache(maxsize=6)
def load_closure(setting):
    directory = ROOT / "data/lorenz96_benchmark/closure" / setting
    metadata = json.loads((directory / "metadata.json").read_text())
    require(metadata["config_id"] == f"config_{setting}" and metadata["family"] == "single_scale", "Closure identity mismatch.")
    require(metadata["filename"] == "closure.joblib", "Unexpected closure filename.")
    validate_references(metadata)
    path = directory / metadata["filename"]
    closure = joblib.load(path)
    require(isinstance(closure, GPClosure), "Expected a GPClosure.")
    return closure, metadata, hashlib.sha256(path.read_bytes()).hexdigest()


def check_closure_reference(record):
    setting = record["config_id"].removeprefix("config_")
    closure, metadata, checksum = load_closure(setting)
    source = record["closure"]
    require(metadata["parameters"] == record["parameters"], "Closure physical parameters mismatch.")
    require(source["closure_id"] == metadata["closure_id"] and source["sha256"] == checksum, "Closure identity/checksum mismatch.")
    require(source["definition"] == metadata["definition"] and source["calibration"] == metadata["calibration"], "Closure provenance mismatch.")
    expected = Path("data/lorenz96_benchmark/closure") / setting
    require(source["source_closure"] == str(expected / "closure.joblib"), "Closure source path mismatch.")
    require(source["source_metadata"] == str(expected / "metadata.json"), "Closure metadata path mismatch.")
    return closure


def validate_closure_samples(setting, record, state):
    started = perf_counter()
    closure, metadata, checksum = load_closure(setting)
    require(metadata["parameters"] == record["parameters"], "Closure parent parameters mismatch.")
    calibration = metadata["calibration"]
    directory = Path("data/lorenz96_benchmark/dense_calibration/multiscale") / f"multiscale_dense_calibration_trajectory_{setting}"
    require(calibration["source_trajectory"] == str(directory / record["filename"]), "Closure training trajectory mismatch.")
    require(calibration["source_metadata"] == str(directory / "metadata.json"), "Closure training metadata mismatch.")
    for key in ("settings", "state_shape", "state_layout", "time_range"):
        require(calibration[key] == record[key], f"Closure calibration mismatch: {key}.")
    require(metadata["sampling"]["sample_size"] == 800 and metadata["sampling"]["seed"] == 42, "Closure requires exactly 800 samples with seed 42.")
    parent = MultiscaleLorenz96(**record["parameters"])
    slow, fast = parent.split_state(state)
    features, targets = sample_closure_training_data(slow, fast, sample_size=800, seed=42)
    regressor = closure.regressor
    require(np.array_equal(regressor.X_train_, features) and np.array_equal(regressor.y_train_, targets), "Saved GP training pairs differ from seeded calibration samples.")
    settings = metadata["fit_settings"]
    require(settings["seed"] == regressor.random_state == 42, "GP optimizer seed mismatch.")
    require(settings["alpha"] == regressor.alpha == 1.0, "GP alpha mismatch.")
    require(settings["n_restarts_optimizer"] == regressor.n_restarts_optimizer == 2, "GP optimizer restarts mismatch.")
    require(settings["normalize_y"] == regressor.normalize_y == False, "GP target normalization mismatch.")
    require(settings["optimizer"] == regressor.optimizer == "fmin_l_bfgs_b", "GP optimizer mismatch.")
    require(np.isclose(regressor.log_marginal_likelihood_value_, metadata["fit_results"]["log_marginal_likelihood"], rtol=1e-10, atol=1e-12), "GP marginal likelihood mismatch.")
    require(str(regressor.kernel) == settings["kernel"], "Initial kernel metadata mismatch.")
    require(np.isfinite(regressor.L_).all() and np.all(np.diag(regressor.L_) > 0), "Invalid GP factorization.")
    require(np.allclose(regressor.kernel_.theta, metadata["fit_results"]["optimized_kernel_theta"], rtol=1e-12, atol=0), "Fitted kernel metadata mismatch.")
    predictions = closure(features[:, 0])
    require(predictions.shape == targets.shape and np.isfinite(predictions).all(), "Invalid closure predictions.")
    rmse = float(np.sqrt(np.mean((predictions - targets) ** 2)))
    require(np.isclose(rmse, metadata["fit_results"]["training_rmse"], rtol=1e-10, atol=1e-12), "GP training RMSE mismatch.")
    grid_predictions = closure(np.linspace(float(slow.min()), float(slow.max()), 1000))
    require(np.isfinite(grid_predictions).all(), "Nonfinite closure predictions across calibration range.")
    return {
        "status": "pass",
        "config_id": record["config_id"],
        "sample_size": 800,
        "sha256": checksum,
        "training_rmse": rmse,
        "validation_seconds": perf_counter() - started,
    }


def validate_calibration(family, setting, tolerance):
    """Validate calibration arrays, recompute bandwidths, and check the GP."""
    directory = ROOT / "data/lorenz96_benchmark/dense_calibration" / family / f"{family}_dense_calibration_trajectory_{setting}"
    record = json.loads((directory / "metadata.json").read_text())
    bandwidth = json.loads((directory / "bandwidth_periods.json").read_text())
    parent = MultiscaleLorenz96(**record["parameters"])
    require(record["config_id"] == f"config_{setting}" and record["parameters"] == expected_parameters(setting), "Calibration configuration mismatch.")
    expected_family = "single_scale" if family == "singlescale" else "multiscale"
    require(record["family"] == expected_family, "Calibration family mismatch.")
    validate_references(record)
    require(bandwidth["config_id"] == record["config_id"], "Bandwidth configuration mismatch.")
    require(bandwidth["source_trajectory"] == record["filename"] and bandwidth["source_metadata"] == "metadata.json", "Bandwidth source mismatch.")
    require(bandwidth["cumulative_energy_cutoff"] == 0.99 and bandwidth["energy_component"] == "AC (DC excluded)", "Bandwidth energy convention mismatch.")
    require(bandwidth["period_units"] == "simulation time" and bandwidth["reduction"] == "median of channel bandwidth periods", "Bandwidth units/reduction mismatch.")
    settings = record["settings"]
    for key, value in {
        "seed": 42,
        "low": -5.0,
        "high": 10.0,
        "t0": 0.0,
        "burn_in": 50.0,
        "sample_dt": 0.0001,
    }.items():
        require(settings[key] == value, f"Calibration setting mismatch: {key}.")
    initial = sample_initial_state(parent, seed=42)[:parent.K]
    require(np.array_equal(initial, record["initialization"]["initial_slow_state"]), "Calibration initial slow state mismatch.")
    model = parent if family == "multiscale" else SingleScaleLorenz96(check_closure_reference(record), K=parent.K, hx=parent.hx, F=parent.F)
    with np.load(directory / record["filename"], allow_pickle=False) as data:
        require(set(data.files) == {"time", "state"}, "Expected exactly time/state arrays.")
        time, state = data["time"], data["state"]
    check_arrays(time, state, record, model.state_dimension, 1_000_000, 0.0001)
    statistics = {
        "slow": group_statistics(state[:parent.K]),
        "fast": group_statistics(state[parent.K:]),
    } if family == "multiscale" else {"state": group_statistics(state)}
    observations = []
    for group, stats in statistics.items():
        observations.extend(statistical_observations(stats, None, group))
    periods = estimate_periods(time, state)
    saved = np.array([np.inf if p is None else p for p in bandwidth["channel_bandwidth_periods"]])
    require(saved.shape == periods.shape and np.allclose(saved, periods, rtol=1e-10, atol=0), "Stored channel bandwidth periods do not match recomputation.")
    groups = (("slow_bandwidth_period", periods[:parent.K]), ("fast_bandwidth_period", periods[parent.K:])) if family == "multiscale" else (("bandwidth_period", periods),)
    for key, values in groups:
        median = float(np.median(values))
        require(np.isfinite(median) and median > 0 and np.isclose(bandwidth[key], median, rtol=1e-10, atol=0), f"Invalid or inconsistent aggregate: {key}.")
    discrepancies, windows = [], []
    try:
        windows, discrepancies = check_reintegration(time, state, model, 0.0001, tolerance)
    except (RuntimeError, ValueError) as error:
        discrepancies.append(f"Calibration short-window reintegration failed: {error}")
    closure_result = None
    if family == "multiscale":
        closure_started = perf_counter()
        try:
            closure_result = validate_closure_samples(setting, record, state)
        except Exception as error:
            closure_result = validation_failure(f"closure/{setting}", f"{type(error).__name__}: {error}", config_id=record["config_id"])
            closure_result.update(status="fail", validation_seconds=perf_counter() - closure_started)
    return {
        "family": expected_family,
        "config_id": record["config_id"],
        "parameters": record["parameters"],
        "source_metadata": str((directory / "metadata.json").relative_to(ROOT)),
        "structural_errors": [],
        "numerical_discrepancies": discrepancies,
        "statistical_observations": observations,
        "state_shape": list(state.shape),
        "time_range": [float(time[0]), float(time[-1])],
        "state_sha256": state_digest(state),
        "statistics": statistics,
        "bandwidth": bandwidth,
        "bandwidth_recomputation_passed": True,
        "closure_validation": closure_result,
        "reintegration_windows": windows,
    }


def validate_singlescale_trajectory(record, directory, calibration, tolerance):
    parent = MultiscaleLorenz96(**record["parameters"])
    require(record["family"] == "single_scale", "Wrong trajectory family.")
    require(record["config_id"] == calibration["config_id"] and record["parameters"] == calibration["parameters"], "Calibration configuration mismatch.")
    sampling = record["sampling"]
    for key in ("bandwidth_period", "channel_bandwidth_periods", "cumulative_energy_cutoff"):
        require(sampling[key] == calibration["bandwidth"][key], f"Calibration mismatch: {key}.")
    require(sampling["n_samples"] == 8192 and sampling["samples_per_bandwidth_period"] == 32, "Single-scale sample count/resolution mismatch.")
    dt = sampling["bandwidth_period"] / 32
    require(np.isfinite(dt) and dt > 0, "Invalid single-scale sampling interval.")
    require(sampling["sample_dt"] == record["settings"]["sample_dt"] == dt, "Single-scale spacing mismatch.")
    model = SingleScaleLorenz96(check_closure_reference(record), K=parent.K, hx=parent.hx, F=parent.F)
    with np.load(directory / record["filename"], allow_pickle=False) as data:
        require(set(data.files) == {"time", "state"}, "Expected exactly time/state arrays.")
        time, state = data["time"], data["state"]
    check_arrays(time, state, record, parent.K, 8192, dt)
    statistics = {"state": group_statistics(state)}
    observations = statistical_observations(statistics["state"], calibration["statistics"]["state"], "state")
    try:
        windows, discrepancies = check_reintegration(time, state, model, dt, tolerance)
    except (RuntimeError, ValueError) as error:
        windows, discrepancies = [], [f"Short-window reintegration failed: {error}"]
    bandwidths, bandwidth_observations = check_bandwidths(time, state, parent.K, sampling, "single_scale")
    observations.extend(bandwidth_observations)
    return {
        "family": "single_scale",
        "trajectory_id": record["trajectory_id"],
        "pair_id": record["pair_id"],
        "config_id": record["config_id"],
        "filename": str((directory / record["filename"]).relative_to(ROOT)),
        "structural_errors": [],
        "numerical_discrepancies": discrepancies,
        "statistical_observations": observations,
        "structural_checks_passed": True,
        "state_shape": list(state.shape),
        "time_range": [float(time[0]), float(time[-1])],
        "state_sha256": state_digest(state),
        "statistics": statistics,
        "reintegration_windows": windows,
        "bandwidths": bandwidths,
    }


def validation_failure(identifier, error, **identity):
    return {
        "item_id": identifier,
        **identity,
        "structural_errors": [str(error)],
        "numerical_discrepancies": [],
        "statistical_observations": [],
        "structural_checks_passed": False,
    }


def result_status(result):
    if result["structural_errors"] or result["numerical_discrepancies"]:
        return "fail"
    return "review" if result["statistical_observations"] else "pass"


def validate_calibration_job(family, setting, tolerance):
    started = perf_counter()
    try:
        result = validate_calibration(family, setting, tolerance)
    except Exception as error:
        result = validation_failure(f"{family}/{setting}", f"{type(error).__name__}: {error}",
                                    family="single_scale" if family == "singlescale" else family, config_id=f"config_{setting}")
    result["validation_seconds"] = perf_counter() - started
    result["status"] = result_status(result)
    return result


def validate_job(record, directory, calibration, tolerance):
    started = perf_counter()
    try:
        if record["family"] == "multiscale":
            result = validate_multiscale_trajectory(record, directory, calibration, tolerance)
        else:
            result = validate_singlescale_trajectory(record, directory, calibration, tolerance)
    except Exception as error:
        result = validation_failure(record["trajectory_id"], f"{type(error).__name__}: {error}",
                                    family=record["family"], trajectory_id=record["trajectory_id"], config_id=record["config_id"])
    result["validation_seconds"] = perf_counter() - started
    result["status"] = result_status(result)
    return result


def expected_parameters(setting):
    values = {
        "a": (9, 8, -0.8, 1.0, 10.0),
        "b_k": (18, 8, -0.8, 1.0, 10.0),
        "b_j": (9, 16, -0.8, 1.0, 10.0),
        "b_hx": (9, 8, -1.2, 1.0, 10.0),
        "b_hy": (9, 8, -0.8, 1.4, 10.0),
        "b_f": (9, 8, -0.8, 1.0, 14.0),
    }
    return dict(zip(("K", "J", "hx", "hy", "F", "eps"), (*values[setting], 2**-7)))


def collect_pairs(selected):
    pairs, errors = [], []
    identifiers, pair_ids, seeds = set(), set(), set()
    base = ROOT / "data/lorenz96_benchmark"
    for label, relative, setting, count, first_seed in selected:
        try:
            members = {}
            for family in ("multiscale", "singlescale"):
                directory = base / relative / family
                records = json.loads((directory / "metadata.json").read_text())
                require(isinstance(records, list) and len(records) == count, f"Expected {count} {family} records.")
                require({r["filename"] for r in records} == {p.name for p in directory.glob("*.npz")}, f"{family} NPZ inventory mismatch.")
                by_id = {r["trajectory_id"]: r for r in records}
                require(set(by_id) == {f"{label}_{family}_{i:03d}" for i in range(1, count + 1)}, f"Unexpected {family} IDs.")
                members[family] = by_id
            for index in range(1, count + 1):
                pair = []
                for family in ("multiscale", "singlescale"):
                    record = members[family][f"{label}_{family}_{index:03d}"]
                    directory = base / relative / family
                    require(record["family"] == ("single_scale" if family == "singlescale" else family), "Family label mismatch.")
                    require(record["config_id"] == f"config_{setting}" and record["parameters"] == expected_parameters(setting), "Physical configuration mismatch.")
                    require(record["pair_id"] == f"{label}_pair_{index:03d}", "Pair ID mismatch.")
                    require(record["split"] == ("test" if label.startswith("test_") else label), "Split mismatch.")
                    require(record["test_subset"] == (label if label.startswith("test_") else None), "Test subset mismatch.")
                    require(record["settings"]["seed"] == first_seed + index - 1, "Initialization seed mismatch.")
                    require(record["trajectory_id"] not in identifiers, "Duplicate trajectory ID.")
                    identifiers.add(record["trajectory_id"])
                    validate_references(record)
                    settings = record["settings"]
                    require(settings["burn_in"] == 50.0 and settings["t0"] == 0.0, "Forecasting preparation mismatch.")
                    require(settings["low"] == -5.0 and settings["high"] == 10.0, "Initialization bounds mismatch.")
                    parent = MultiscaleLorenz96(**record["parameters"])
                    initial = sample_initial_state(parent, seed=settings["seed"])[:parent.K]
                    require(np.array_equal(initial, record["initialization"]["initial_slow_state"]), "Initial slow vector mismatch.")
                    pair.append((record, directory))
                m, s = pair[0][0], pair[1][0]
                for key in ("config_id", "parameters", "pair_id", "split", "test_subset", "initialization"):
                    require(m[key] == s[key], f"Pair mismatch: {key}.")
                for key in ("seed", "low", "high", "burn_in", "t0", "method", "rtol", "atol", "max_step"):
                    require(m["settings"][key] == s["settings"][key], f"Pair settings mismatch: {key}.")
                for source, partner, family in ((m, s, "singlescale"), (s, m, "multiscale")):
                    link = source[f"paired_{family}_trajectory"]
                    require(link["trajectory_id"] == partner["trajectory_id"], "Partner ID mismatch.")
                    expected = Path("data/lorenz96_benchmark") / relative / family
                    require(link["source_trajectory"] == str(expected / partner["filename"]), "Partner file mismatch.")
                    require(link["source_metadata"] == str(expected / "metadata.json"), "Partner metadata mismatch.")
                require(m["pair_id"] not in pair_ids and m["settings"]["seed"] not in seeds, "Duplicate pair ID or seed.")
                pair_ids.add(m["pair_id"])
                seeds.add(m["settings"]["seed"])
                pairs.append((setting, pair))
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(f"{label}: {error}")
    return pairs, errors


def run_validation(jobs, settings):
    """Run preparation and forecasting checks with one shared progress bar."""
    calibrations, results = {}, []
    with (
        tqdm(total=2 * len(settings) + len(jobs), desc="Validating dataset", unit="item") as progress,
        ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=get_context("spawn")) as executor,
    ):
        futures = {executor.submit(validate_calibration_job, family, setting, REINTEGRATION_TOLERANCE): (family, setting)
                   for setting in settings for family in ("multiscale", "singlescale")}
        for future in as_completed(futures):
            family, setting = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = validation_failure(f"{family}/{setting}", f"Worker failed: {error}", family="single_scale" if family == "singlescale" else family, config_id=f"config_{setting}")
                result.update(status="fail", validation_seconds=None)
            calibrations[family, setting] = result
            progress.update(1)
        futures = {}
        for record, directory in jobs:
            family = "singlescale" if record["family"] == "single_scale" else "multiscale"
            setting = record["config_id"].removeprefix("config_")
            reference = calibrations[family, setting]
            if reference["structural_errors"]:
                result = validation_failure(
                    record["trajectory_id"],
                    "Calibration validation failed; forecasting check skipped.",
                    trajectory_id=record["trajectory_id"],
                    family=record["family"],
                    config_id=record["config_id"]
                )
                result.update(status="fail", validation_seconds=0.0)
                results.append(result)
                progress.update(1)
            else:
                future = executor.submit(validate_job, record, directory, reference, REINTEGRATION_TOLERANCE)
                futures[future] = record
        for future in as_completed(futures):
            record = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = validation_failure(record["trajectory_id"], f"Worker failed: {error}", trajectory_id=record["trajectory_id"], family=record["family"], config_id=record["config_id"])
                result.update(status="fail", validation_seconds=None)
            results.append(result)
            progress.update(1)
    return calibrations, results


def summarize_results(jobs, settings, selected, pairs, inventory_errors, calibrations, results):
    """Restore output order, detect duplicates, and determine final status."""
    order = {record["trajectory_id"]: index for index, (record, _) in enumerate(jobs)}
    results.sort(key=lambda result: order[result["trajectory_id"]])
    digests = {}
    for result in [*calibrations.values(), *results]:
        digest = result.get("state_sha256")
        if digest is not None:
            identity = result.get("trajectory_id", result.get("source_metadata", result["config_id"]))
            if digest in digests:
                result["structural_errors"].append(f"State array duplicates {digests[digest]}.")
                result["structural_checks_passed"] = False
            else:
                digests[digest] = identity
        result["status"] = result_status(result)
    closures = []
    for setting in settings:
        result = calibrations["multiscale", setting].get("closure_validation")
        if result is None:
            result = validation_failure(f"closure/{setting}", "Multiscale calibration failed; closure training-sample check skipped.", config_id=f"config_{setting}")
            result.update(status="fail", validation_seconds=0.0)
        closures.append(result)
    all_results = [*calibrations.values(), *closures, *results]
    status = "fail" if inventory_errors or any(r["status"] == "fail" for r in all_results) else "review" if any(r["status"] == "review" for r in all_results) else "pass"
    full = len(selected) == 8 and len(pairs) == 76 and len(results) == 152 and len(calibrations) == 12 and len(closures) == 6 and not inventory_errors
    return closures, status, full


def build_report(selected, inventory_pairs, pairs, calibrations, closures, results,
                 inventory_errors, status, full, run_started):
    """Build the existing report schema from completed checks."""
    return {
        "dataset_version": "v0.5",
        "validation": "complete paired Lorenz–96 dataset",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "scope": {
            "collections": [row[0] for row in selected],
            "inventory_pairs": inventory_pairs,
            "validated_pairs": len(pairs),
            "forecasting_trajectories": len(results),
            "dense_calibrations": len(calibrations),
            "closures": len(closures),
            "expected_full_dataset": {
                "pairs": 76,
                "forecasting_trajectories": 152,
                "dense_calibrations": 12,
                "closures": 6,
            },
            "full_dataset_validation": full,
        },
        "summary": {section: {state: sum(r["status"] == state for r in entries) for state in ("pass", "review", "fail")}
                    for section, entries in (("forecasting", results), ("calibrations", list(calibrations.values())), ("closures", closures))},
        "inventory_errors": inventory_errors,
        "validation_settings": {
            "worker_processes": MAX_WORKERS,
            "reintegration": {
                "windows_per_trajectory": 3,
                "intervals_per_window": 16,
                "method": "DOP853",
                "rtol": 1e-10,
                "atol": 1e-12,
                "max_step": "sample_dt / 2",
                "scaled_error": "abs(reintegrated - saved) / (1 + abs(saved))",
                "scaled_error_threshold": REINTEGRATION_TOLERANCE,
            },
            "statistics": {
                "time_blocks": 4,
                "near_constant_channel_std": 1e-10,
                "block_mean_spread_in_std_threshold": 0.5,
                "block_std_ratio_threshold": 1.5,
                "mean_shift_in_calibration_std_threshold": 0.5,
                "std_to_calibration_ratio_range": [0.5, 2.0],
                "amplitude_to_calibration_threshold": 5.0,
            },
            "bandwidth": {
                "cumulative_energy_cutoff": 0.99,
                "channel_batch_size": 8,
                "period_ratio_review_range": [0.5, 2.0],
                "stored_calibration_period_rtol": 1e-10,
            },
            "interpretation": "Statistical thresholds flag observations for review, not invalid dynamics. Bandwidth comparisons do not establish absence of aliasing; these checks do not prove chaotic dynamics.",
        },
        "calibrations": {f"{family}/{setting}": result for (family, setting), result in sorted(calibrations.items())},
        "closures": sorted(closures, key=lambda r: r["config_id"]),
        "trajectories": results,
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "validation_seconds": perf_counter() - run_started,
    }


def save_report(report, report_path):
    """Replace the output report only after its temporary file is written."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(report_path)


def main():
    run_started = perf_counter()
    report_path = REPORT_PATH.resolve()
    require(MAX_WORKERS > 0 and (PAIR_LIMIT is None or PAIR_LIMIT > 0), "Worker count and pair limit must be positive.")
    selected = [row for row in COLLECTIONS if SELECTED_COLLECTIONS is None or row[0] in SELECTED_COLLECTIONS]
    require(selected and (SELECTED_COLLECTIONS is None or set(SELECTED_COLLECTIONS) <= {row[0] for row in COLLECTIONS}), "Unknown or empty collection selection.")
    pairs, inventory_errors = collect_pairs(selected)
    inventory_pairs = len(pairs)
    if PAIR_LIMIT is not None:
        pairs = pairs[:PAIR_LIMIT]
    jobs = [member for _, pair in pairs for member in pair]
    settings = list(dict.fromkeys(setting for setting, _ in pairs))
    calibrations, results = run_validation(jobs, settings)
    closures, status, full = summarize_results(jobs, settings, selected, pairs, inventory_errors, calibrations, results)
    report = build_report(selected, inventory_pairs, pairs, calibrations, closures, results, inventory_errors, status, full, run_started)
    save_report(report, report_path)
    print(f"{status.upper()} ({'full dataset' if full else 'partial dataset'}): {report['summary']}; {perf_counter() - run_started:.2f}s elapsed; report: {report_path}", flush=True)
    return {
        "pass": 0,
        "fail": 1,
        "review": 2,
    }[status]


if __name__ == "__main__":
    sys.exit(main())
