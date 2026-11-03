r"""Fit one shared local GP closure from a multiscale Lorenz–96 trajectory.

Example from the repository root in the lorenz96_benchmark environment::

    python scripts/lorenz96_benchmark/train_closure.py \
        --input-path data/calibration/multiscale_001.npz \
        --sample-size 800 --seed 2003 \
        --output-prefix data/closures/closure_001

The input NPZ contains time [N], slow_states [K, N], and fast_states [K, J, N].
Dimensions and fast-group associations come from the array shapes. Timestamps must be uniformly
spaced, and burn-in must already be excluded. Each training pair maps one
slow value to its associated fast-group mean, before multiplication by hx.
Samples up to --sample-size site/time pairs without replacement.

Uses fit_closure's default constant-times-RBF kernel. Writes a callable
GPClosure to <output-prefix>.joblib and settings and fitting diagnostics to
<output-prefix>.json. Rerunning replaces only that pair. Relative paths use
the working directory. Parameter sweeps belong in experiment shell files.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--input-path", type=Path, required=True, help="Multiscale trajectory NPZ containing time, slow_states, and fast_states.")
    parser.add_argument("--sample-size", type=int, default=800, help="Maximum number of site/time training pairs; default: 800.")
    parser.add_argument("--seed", type=int, default=2003, help="Root seed for training-pair selection and GP optimization; default: 2003.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Nonnegative diagonal regularization/noise variance; default: 1.")
    parser.add_argument("--n-restarts-optimizer", type=int, default=2, help="Additional optimization runs after the initial fit; default: 2.")
    parser.add_argument("--output-prefix", type=Path, required=True, help="Output path without an extension; writes <prefix>.joblib and <prefix>.json, replacing existing files.")

    args = parser.parse_args()

    # Both output extensions are appended to the supplied filename stem
    if (
        not args.output_prefix.name
        or args.output_prefix.name in (".", "..")
        or args.output_prefix.name != args.output_prefix.name.strip()
        or args.output_prefix.suffix
    ):
        parser.error("--output-prefix must include a nonempty filename stem without an extension")

    if args.sample_size < 1:
        parser.error("--sample-size must be positive")
    if args.seed < 0 or args.n_restarts_optimizer < 0:
        parser.error("seed and n-restarts-optimizer must be nonnegative")
    if not math.isfinite(args.alpha) or args.alpha < 0:
        parser.error("--alpha must be finite and nonnegative")

    return args


def main() -> None:
    args = parse_args()

    # Keep --help usable without scientific dependencies
    import joblib
    import numpy as np

    from src.data._utils.validation import _validate_finite, _validate_positive, _validate_real
    from src.data.lorenz96 import fit_closure
    from src.data.lorenz96.metadata import build_closure_metadata

    source = args.input_path.resolve()
    output = args.output_prefix.parent.resolve()
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(output)

    closure_path = output / f"{args.output_prefix.name}.joblib"
    metadata_path = output / f"{args.output_prefix.name}.json"
    for path in (closure_path, metadata_path):
        if path.resolve() == source:
            raise ValueError("Output must not overwrite the input trajectory")
        if path.is_dir():
            raise IsADirectoryError(path)

    # Stream the checksum to avoid loading the entire compressed file into memory
    with source.open("rb") as file:
        source_hash = hashlib.file_digest(file, "sha256").hexdigest()
        file.seek(0)
        with np.load(file, allow_pickle=False) as data:
            if "fast_states" not in data:
                raise ValueError("Closure training requires a multiscale trajectory containing fast_states")
            time, slow, fast = data["time"], data["slow_states"], data["fast_states"]

    # Reject complex data before converting timestamps or sampling training pairs
    _validate_real([("time", time), ("slow_states", slow), ("fast_states", fast)])
    time = np.asarray(time, dtype=np.float64)
    if time.ndim != 1 or time.size < 2:
        raise ValueError("time must have shape (N,) with N >= 2")
    if slow.ndim != 2 or slow.shape[1] != time.size:
        raise ValueError("slow_states must have shape (K, N) matching the timestamps")

    # Uniform timestamps give uniformly sampled site/time pairs equal physical-time weight
    _validate_finite([("time", time)])
    sample_dt = float((time[-1] - time[0]) / (time.size - 1))
    _validate_finite([("sample_dt", sample_dt)])
    _validate_positive([("sample_dt", sample_dt)])
    if not np.allclose(np.diff(time), sample_dt, rtol=1e-7, atol=0):
        raise ValueError("Training timestamps must be uniformly spaced and increasing")

    settings = dict(
        sample_size=args.sample_size,
        seed=args.seed,
        kernel=None, alpha=args.alpha, n_restarts_optimizer=args.n_restarts_optimizer,
    )

    started = perf_counter()
    closure = fit_closure(
        slow, fast, sample_size=args.sample_size, seed=args.seed,
        alpha=args.alpha, n_restarts_optimizer=args.n_restarts_optimizer,
    )
    regressor = closure.regressor
    features, targets = regressor.X_train_, regressor.y_train_
    predictions = closure(features[:, 0])
    if predictions.shape != targets.shape:
        raise RuntimeError("Closure predictions must match the training target shape")
    _validate_finite([("closure predictions", predictions)])
    training_seconds = perf_counter() - started

    metadata = build_closure_metadata(
        filename=closure_path.name, source_trajectory=str(source), source_sha256=source_hash, settings=settings,
        fit_results={
            "training_samples": int(features.shape[0]),
            "available_pairs": int(slow.size),
            "optimized_kernel": str(regressor.kernel_),
            "log_marginal_likelihood": float(regressor.log_marginal_likelihood_value_),
            "training_rmse": float(np.sqrt(np.mean((predictions - targets) ** 2))),
        },
        training_seconds=training_seconds,
    )
    metadata_text = json.dumps(metadata, indent=2, allow_nan=False) + "\n"

    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{args.output_prefix.name}-", dir=output) as temporary:
        staging = Path(temporary)
        joblib.dump(closure, staging / closure_path.name, compress=3)
        (staging / metadata_path.name).write_text(metadata_text, encoding="utf-8")

        # Prepare both files before replacing either member of the output pair
        for name in (closure_path.name, metadata_path.name):
            (staging / name).replace(output / name)

    print(f"Fitted GP closure on {features.shape[0]} pairs in {training_seconds:.2f}s: {closure_path}")


if __name__ == "__main__":
    main()
