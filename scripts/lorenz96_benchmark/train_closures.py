"""Fit the six frozen single-scale closures from dense calibration data.

Run from the repository root using environments/lorenz96_benchmark.yml:
    python scripts/lorenz96_benchmark/train_closures.py

Each configuration samples exactly 800 site/time pairs without replacement,
using seed 42 for both sampling and GP optimization. Writes closure.joblib
and metadata.json to data/lorenz96_benchmark/closure/<setting>/.
Load the callable closure with joblib.load(path), with the repository root
on Python's import path so that src.data.lorenz96.GPClosure can be imported.
Rerunning overwrites this script's outputs; files are replaced after writing
their temporary files successfully.
"""

import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import joblib
import numpy as np
from tqdm import tqdm
import scipy
import sklearn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.lorenz96 import (
    MultiscaleLorenz96,
    fit_closure,
    sample_closure_training_data,
)


SETTINGS = ("a", "b_f", "b_hx", "b_hy", "b_j", "b_k")
SAMPLE_SIZE = 800
SAMPLING_SEED = 42
FIT_SEED = 42


def train_and_save_closure(setting):
    """Fit one configuration's closure, keeping calibration data out of splits."""
    source_dir = ROOT / "data/lorenz96_benchmark/dense_calibration/multiscale" / f"multiscale_dense_calibration_trajectory_{setting}"
    source_metadata_path = source_dir / "metadata.json"
    source_path = source_dir / f"multiscale_dense_calibration_trajectory_{setting}.npz"
    source_metadata = json.loads(source_metadata_path.read_text())
    if (
        source_metadata["config_id"] != f"config_{setting}"
        or source_metadata["filename"] != source_path.name
    ):
        raise ValueError(f"Calibration identity mismatch for {setting}.")
    model = MultiscaleLorenz96(**source_metadata["parameters"])

    output_dir = ROOT / "data/lorenz96_benchmark/closure" / setting
    closure_path = output_dir / "closure.joblib"
    metadata_path = output_dir / "metadata.json"

    started = perf_counter()
    with np.load(source_path, allow_pickle=False) as data:
        time = data["time"]
        state = data["state"]
    if (
        time.ndim != 1
        or time.size < 2
        or state.shape != (model.state_dimension, time.size)
        or list(state.shape) != source_metadata["state_shape"]
    ):
        raise ValueError(f"Calibration shape mismatch for {setting}.")
    sample_dt = float((time[-1] - time[0]) / (time.size - 1))
    if (
        not np.isfinite(time).all()
        or not np.isfinite(sample_dt)
        or sample_dt <= 0
        or not np.allclose(np.diff(time), sample_dt, rtol=1e-7, atol=0)
    ):
        raise ValueError(f"Calibration timestamps must be finite and uniform for {setting}.")
    slow, fast = model.split_state(state)
    if slow.size < SAMPLE_SIZE:
        raise ValueError(f"Calibration must contain at least {SAMPLE_SIZE} site/time pairs.")
    features, targets = sample_closure_training_data(slow, fast, sample_size=SAMPLE_SIZE, seed=SAMPLING_SEED)
    calibration_time_range = [float(time[0]), float(time[-1])]
    available_pairs = int(slow.size)
    del time, state, slow, fast

    closure = fit_closure(
        features,
        targets,
        seed=FIT_SEED,
        kernel=None,
        alpha=1.0,
        n_restarts_optimizer=2,
    )
    predictions = closure(features[:, 0])
    if predictions.shape != targets.shape or not np.isfinite(predictions).all():
        raise RuntimeError(f"Invalid fitted closure predictions for {setting}.")
    elapsed = perf_counter() - started
    regressor = closure.regressor
    record = {
        "config_id": source_metadata["config_id"],
        "closure_id": f"config_{setting}_gp_closure",
        "family": "single_scale",
        "filename": closure_path.name,
        "serialization": "joblib (src.data.lorenz96.closure.GPClosure)",
        "parameters": source_metadata["parameters"],
        "definition": {
            "type": "shared local instantaneous Gaussian-process conditional-mean estimate",
            "input": "x_k(t), one scalar slow value",
            "target": "mean_j y_{k,j}(t), before multiplication by hx",
            "slow_rhs_feedback": "hx * closure(slow_state)",
            "shared_across_slow_sites": True,
        },
        "calibration": {
            "source_trajectory": str(source_path.relative_to(ROOT)),
            "source_metadata": str(source_metadata_path.relative_to(ROOT)),
            "settings": source_metadata["settings"],
            "state_layout": source_metadata["state_layout"],
            "state_shape": source_metadata["state_shape"],
            "time_range": calibration_time_range,
            "forecasting_split": None,
        },
        "sampling": {
            "sample_size": SAMPLE_SIZE,
            "seed": SAMPLING_SEED,
            "method": "uniform site/time pairs without replacement",
            "implementation": "src.data.lorenz96.sample_closure_training_data",
            "flatten_order": "C (slow site, then time)",
            "available_pairs": available_pairs,
            "feature_shape": list(features.shape),
            "target_shape": list(targets.shape),
            "feature_range": [float(features.min()), float(features.max())],
            "target_range": [float(targets.min()), float(targets.max())],
        },
        "fit_settings": {
            "seed": FIT_SEED,
            "kernel": str(regressor.kernel),
            "kernel_definition": "ConstantKernel(1.0) * RBF(length_scale=3.0, length_scale_bounds=(1e-3, 1e3))",
            "alpha": regressor.alpha,
            "n_restarts_optimizer": regressor.n_restarts_optimizer,
            "optimizer": regressor.optimizer,
            "normalize_y": regressor.normalize_y,
        },
        "fit_results": {
            "optimized_kernel": str(regressor.kernel_),
            "optimized_kernel_theta": regressor.kernel_.theta.tolist(),
            "log_marginal_likelihood": float(regressor.log_marginal_likelihood_value_),
            "training_rmse": float(np.sqrt(np.mean((predictions - targets) ** 2))),
        },
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "training_seconds": elapsed,
    }
    # Serialize metadata before writing either output, rejecting nonfinite values.
    metadata_text = json.dumps(record, indent=2, allow_nan=False) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = closure_path.with_suffix(".joblib.tmp")
    joblib.dump(closure, temporary_path, compress=3)
    temporary_path.replace(closure_path)
    temporary_path = metadata_path.with_suffix(".json.tmp")
    temporary_path.write_text(metadata_text)
    temporary_path.replace(metadata_path)


def main():
    run_started = perf_counter()
    # Check all source files before starting any fitting.
    for setting in SETTINGS:
        source_dir = ROOT / "data/lorenz96_benchmark/dense_calibration/multiscale" / f"multiscale_dense_calibration_trajectory_{setting}"
        for name in ("metadata.json", f"multiscale_dense_calibration_trajectory_{setting}.npz"):
            if not (source_dir / name).is_file():
                raise FileNotFoundError(source_dir / name)

    for setting in tqdm(SETTINGS, desc="Training closures", unit="configuration"):
        train_and_save_closure(setting)

    print(f"Closure training finished: {len(SETTINGS)} closures saved in {perf_counter() - run_started:.2f}s.", flush=True)


if __name__ == "__main__":
    main()
