r"""Generate one single-scale Lorenz–96 trajectory from explicit settings.

Example from the repository root in the lorenz96_benchmark environment::

    python scripts/lorenz96_benchmark/generate_singlescale.py \
        --closure-path data/my_closure/closure.joblib --K 9 --sample-dt 0.0001 --n-samples 1000000 \
        --burn-in 50 --seed 2003 --output-prefix data/my_calibration/trajectory_001

Writes <output-prefix>.npz (time [N], slow_states [K, N]) and matching <output-prefix>.json.
Loads a saved GPClosure without fitting. One invocation generates one trajectory;
experiment shell files select splits, parameter sweeps and seeds. Existing
outputs with the same name are replaced. Relative paths use the working directory.
K, hx and forcing describe the reduced dynamics. Metadata references the saved
closure by its path and checksum.
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

    # Model parameters
    parser.add_argument("--K", type=int, default=9)
    parser.add_argument("--hx", type=float, default=-0.8)
    parser.add_argument("--forcing", type=float, default=10.0)

    # Frozen closure input
    parser.add_argument("--closure-path", type=Path, required=True, help="Saved GPClosure joblib file.")

    # Reproducibility and initial-state sampling
    parser.add_argument("--seed", type=int, default=2003)
    parser.add_argument("--low", type=float, default=-5.0)
    parser.add_argument("--high", type=float, default=10.0)

    # Simulation interval and sampling
    parser.add_argument("--sample-dt", type=float, required=True, help="Spacing of saved slow states.")
    length_group = parser.add_mutually_exclusive_group(required=True)
    length_group.add_argument("--duration", type=float, help="Recorded duration, an integer multiple of sample-dt; n-samples = duration / sample-dt + 1 includes both endpoints. Provide this or --n-samples.")
    length_group.add_argument("--n-samples", type=int, help="Number of saved states including both endpoints (>= 2); duration = (n-samples - 1) * sample-dt. Provide this or --duration.")
    parser.add_argument("--burn-in", type=float, default=0.0)
    parser.add_argument("--t0", type=float, default=0.0)

    # Numerical solver settings
    parser.add_argument("--method", choices=("RK23", "RK45", "DOP853", "Radau", "BDF", "LSODA"), default="DOP853")
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--max-step", type=float, help="Maximum integration step during burn-in and recording; default: sample-dt.")

    # Output path prefix
    parser.add_argument("--output-prefix", type=Path, required=True, help="Output path without an extension; writes <prefix>.npz and <prefix>.json, replacing existing files.")

    args = parser.parse_args()

    # The prefix must supply a filename stem because both extensions are appended
    if (
        not args.output_prefix.name
        or args.output_prefix.name in (".", "..")
        or args.output_prefix.name != args.output_prefix.name.strip()
        or args.output_prefix.suffix
    ):
        parser.error("--output-prefix must include a nonempty filename stem without an extension")

    # Check the model dimension and initialization seed before generation
    if args.K < 4 or args.seed < 0:
        parser.error("K must be >= 4 and seed nonnegative")

    # Reject nonfinite values before checking bounds, since NaN can evade comparisons
    for name in ("hx", "forcing", "low", "high", "burn_in", "t0", "rtol", "atol"):
        if not math.isfinite(getattr(args, name)):
            parser.error(f"--{name.replace('_', '-')} must be finite")
    if args.rtol <= 0 or args.atol < 0 or args.burn_in < 0 or args.low >= args.high:
        parser.error("Require rtol > 0, atol >= 0, burn-in >= 0, and low < high")

    # Duration is absent when the mutually exclusive sample-count option is selected
    for name in ("sample_dt", "duration"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")

    # Including both endpoints requires at least two samples for a positive duration
    if args.n_samples is not None and args.n_samples < 2:
        parser.error("--n-samples must be >= 2")

    # An omitted cap uses sample-dt; positive infinity explicitly leaves steps uncapped
    if args.max_step is not None and (math.isnan(args.max_step) or args.max_step <= 0):
        parser.error("--max-step must be positive; inf is allowed")

    return args


def main() -> None:
    args = parse_args()

    # Keep --help usable without scientific dependencies
    import numpy as np
    import joblib

    from src.data.lorenz96 import GPClosure, generate_singlescale_trajectory
    from src.data.lorenz96.metadata import build_singlescale_metadata

    output = args.output_prefix.parent.resolve()

    # An existing parent path must be a directory; a missing one is created before saving
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(output)

    trajectory_path = output / f"{args.output_prefix.name}.npz"
    metadata_path = output / f"{args.output_prefix.name}.json"

    # Output files may be replaced, but existing directories must not be used as file destinations
    for path in (trajectory_path, metadata_path):
        if path.is_dir():
            raise IsADirectoryError(path)

    closure_path = args.closure_path.resolve()
    if closure_path in (trajectory_path.resolve(), metadata_path.resolve()):
        raise ValueError("Output must not overwrite the closure input")
    closure_hash = hashlib.sha256(closure_path.read_bytes()).hexdigest()
    closure = joblib.load(closure_path)
    if not isinstance(closure, GPClosure):
        raise TypeError(f"Expected a saved GPClosure: {closure_path}")

    parameters = dict(K=args.K, hx=args.hx, F=args.forcing)
    closure_record = dict(source_closure=str(closure_path), sha256=closure_hash)
    settings = dict(
        seed=args.seed, low=args.low, high=args.high, sample_dt=args.sample_dt, duration=args.duration, n_samples=args.n_samples,
        burn_in=args.burn_in, t0=args.t0, method=args.method,
        rtol=args.rtol, atol=args.atol, max_step=args.max_step,
    )

    started = perf_counter()
    trajectory = generate_singlescale_trajectory(
        closure=closure, **parameters, seed=args.seed, low=args.low, high=args.high,
        sample_dt=args.sample_dt, duration=args.duration, n_samples=args.n_samples,
        burn_in=args.burn_in, t0=args.t0, method=args.method,
        rtol=args.rtol, atol=args.atol, max_step=args.max_step,
    )

    generation_seconds = perf_counter() - started
    metadata = build_singlescale_metadata(
        filename=trajectory_path.name, parameters=parameters, settings=settings, closure=closure_record,
        trajectory=trajectory,
        generation_seconds=generation_seconds,
    )
    metadata_text = json.dumps(metadata, indent=2, allow_nan=False) + "\n"

    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{args.output_prefix.name}-", dir=output) as temporary:
        staging = Path(temporary)
        np.savez_compressed(staging / trajectory_path.name, time=trajectory.time, slow_states=trajectory.state)
        (staging / metadata_path.name).write_text(metadata_text)

        # Prepare both files before replacing either member of the output pair
        for name in (trajectory_path.name, metadata_path.name):
            (staging / name).replace(output / name)

    print(f"Saved single-scale trajectory {list(trajectory.state.shape)} in {generation_seconds:.2f}s: {trajectory_path}")


if __name__ == "__main__":
    main()
