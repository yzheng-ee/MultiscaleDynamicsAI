r"""Generate one multiscale Lorenz–96 trajectory from explicit settings.

Example from the repository root in the lorenz96_benchmark environment::

    python scripts/lorenz96_benchmark/generate_multiscale.py \
        --K 9 --J 8 --sample-dt 0.0001 --n-samples 1000000 \
        --burn-in 50 --seed 2003 --output-prefix data/my_calibration/trajectory_001

Writes <output-prefix>.npz with time [N], slow_states [K, N], and fast_states
[K, J, N], plus matching <output-prefix>.json.
One invocation generates one trajectory; experiment shell files select splits,
parameter sweeps and seeds. Rerunning replaces only the named output pair.
Relative input and output paths are resolved from the working directory.
The sample count and duration always refer to the saved fine grid.
"""

import argparse
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
    parser.add_argument("--J", type=int, default=8)
    parser.add_argument("--hx", type=float, default=-0.8)
    parser.add_argument("--hy", type=float, default=1.0)
    parser.add_argument("--forcing", type=float, default=10.0)
    parser.add_argument("--epsilon", type=float, default=2**-7)

    # Reproducibility and initial-state sampling
    parser.add_argument("--seed", type=int, default=2003)
    parser.add_argument("--low", type=float, default=-5.0)
    parser.add_argument("--high", type=float, default=10.0)

    # Simulation interval and sampling
    parser.add_argument("--sample-dt", type=float, required=True, help="Spacing of saved full states.")
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

    # Check model dimensions and the initialization seed before generation
    if args.K < 4 or args.J < 3 or args.seed < 0:
        parser.error("K must be >= 4, J >= 3, and seed nonnegative")

    # Reject nonfinite values before checking bounds, since NaN can evade comparisons
    for name in ("hx", "hy", "forcing", "epsilon", "low", "high", "burn_in", "t0", "rtol", "atol"):
        if not math.isfinite(getattr(args, name)):
            parser.error(f"--{name.replace('_', '-')} must be finite")
    if args.epsilon <= 0 or args.rtol <= 0 or args.atol < 0 or args.burn_in < 0 or args.low >= args.high:
        parser.error("Require epsilon > 0, rtol > 0, atol >= 0, burn-in >= 0, and low < high")

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

    from src.data.lorenz96 import generate_multiscale_trajectory
    from src.data.lorenz96.metadata import build_multiscale_metadata

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

    parameters = dict(K=args.K, J=args.J, hx=args.hx, hy=args.hy, F=args.forcing, eps=args.epsilon)
    settings = dict(
        seed=args.seed, low=args.low, high=args.high, sample_dt=args.sample_dt, duration=args.duration, n_samples=args.n_samples,
        burn_in=args.burn_in, t0=args.t0, method=args.method,
        rtol=args.rtol, atol=args.atol, max_step=args.max_step,
    )

    started = perf_counter()
    trajectory = generate_multiscale_trajectory(
        **parameters, seed=args.seed, low=args.low, high=args.high,
        sample_dt=args.sample_dt, duration=args.duration, n_samples=args.n_samples,
        burn_in=args.burn_in, t0=args.t0, method=args.method,
        rtol=args.rtol, atol=args.atol, max_step=args.max_step,
    )

    generation_seconds = perf_counter() - started
    metadata = build_multiscale_metadata(
        filename=trajectory_path.name, parameters=parameters, settings=settings,
        trajectory=trajectory,
        generation_seconds=generation_seconds,
    )
    metadata_text = json.dumps(metadata, indent=2, allow_nan=False) + "\n"

    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{args.output_prefix.name}-", dir=output) as temporary:
        staging = Path(temporary)
        np.savez_compressed(
            staging / trajectory_path.name, time=trajectory.time,
            slow_states=trajectory.state[:args.K],
            fast_states=trajectory.state[args.K:].reshape(args.K, args.J, trajectory.time.size),
        )
        (staging / metadata_path.name).write_text(metadata_text)

        # Prepare both files before replacing either member of the output pair
        for name in (trajectory_path.name, metadata_path.name):
            (staging / name).replace(output / name)

    print(f"Saved multiscale trajectory {list(trajectory.state.shape)} in {generation_seconds:.2f}s: {trajectory_path}")


if __name__ == "__main__":
    main()
