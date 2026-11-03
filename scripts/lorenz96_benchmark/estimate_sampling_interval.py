r"""Estimate bandwidth frequencies, periods, and sampling intervals.

Single-scale example from the repository root::

    python scripts/lorenz96_benchmark/estimate_sampling_interval.py \
        --input-path data/calibration/singlescale_001.npz \
        --reduction median --samples-per-period 32 \
        --output-path data/calibration/singlescale_001_sampling.json

Slow and fast groups are detected from the named NPZ arrays. Each group is
estimated and reduced independently. Without fast_states, the fast estimate is null.
The JSON always contains estimate.slow and estimate.fast.
For multiscale input, sampling records a power-of-two subdivision of the slow
interval and the aligned fine interval. The original estimates are preserved.
For single-scale input, sampling is null.

The input NPZ contains time [N], slow_states [K, N], and optional fast_states
[K, J, N]. Each fast variable is analyzed as a separate channel. Each group's result
contains channel frequencies, a reduced frequency, its corresponding period,
and a sampling interval. --reduction defaults to median; --samples-per-period
is required. Frequencies are reduced before conversion to a period. An infinite
period cannot define a finite sampling interval and is rejected.
Rerunning replaces the specified output JSON. Relative paths use the working
directory. Use the lorenz96_benchmark environment.
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

    parser.add_argument("--input-path", type=Path, required=True, help="Trajectory NPZ containing time, slow_states, and optional fast_states.")
    parser.add_argument("--cumulative-energy-cutoff", type=float, default=0.99, help="Fraction of each channel's AC energy retained by its bandwidth; strictly between 0 and 1.")
    parser.add_argument("--reduction", choices=("max", "min", "mean", "median"), default="median", help="Reduction of channel frequencies within each group; default: median.")
    parser.add_argument("--samples-per-period", type=int, required=True, help="Positive number of intervals per reduced bandwidth period, used to derive the sampling interval.")
    parser.add_argument("--output-path", type=Path, required=True, help="Destination JSON path; existing output is replaced.")

    args = parser.parse_args()

    if args.output_path.suffix != ".json":
        parser.error("--output-path must name a .json file")

    if not math.isfinite(args.cumulative_energy_cutoff) or not 0 < args.cumulative_energy_cutoff < 1:
        parser.error("--cumulative-energy-cutoff must be finite and strictly between 0 and 1")
    if args.samples_per_period < 1:
        parser.error("--samples-per-period must be positive")

    return args


def main() -> None:
    args = parse_args()

    # Keep --help usable without scientific dependencies
    import numpy as np

    from src.data.lorenz96 import Trajectory
    from src.data.energy_bandwidth import align_multiscale_sampling, estimate
    from src.data.energy_bandwidth.metadata import build_estimation_metadata

    source = args.input_path.resolve()
    output = args.output_path.absolute()

    # The output must not replace the input, including through symbolic links
    if source == output.resolve():
        raise ValueError("Output must not overwrite the input trajectory")
    if output.is_dir():
        raise IsADirectoryError(output)
    if output.parent.exists() and not output.parent.is_dir():
        raise NotADirectoryError(output.parent)

    # Stream the checksum to avoid loading the entire compressed file into memory
    with source.open("rb") as file:
        source_hash = hashlib.file_digest(file, "sha256").hexdigest()
        file.seek(0)
        with np.load(file, allow_pickle=False) as data:
            time, slow = data["time"], data["slow_states"]
            fast = data["fast_states"] if "fast_states" in data else None

    if time.ndim != 1 or time.size < 2:
        raise ValueError("time must have shape (N,) with N >= 2")
    if slow.ndim != 2 or slow.shape[0] == 0 or slow.shape[1] != time.size:
        raise ValueError("slow_states must have nonempty shape (K, N) matching the timestamps")

    groups = {"slow": Trajectory(time=time, state=slow)}
    if fast is not None:
        # Preserve site/group associations in storage, then flatten sites and fast variables into FFT channels
        if fast.ndim != 3 or fast.shape[0] != slow.shape[0] or fast.shape[1] == 0 or fast.shape[2] != time.size:
            raise ValueError("fast_states must have shape (K, J, N) matching slow_states, with J > 0")
        groups["fast"] = Trajectory(time=time, state=fast.reshape(-1, time.size))

    settings = dict(
        cumulative_energy_cutoff=args.cumulative_energy_cutoff,
        reduction=args.reduction,
        samples_per_period=args.samples_per_period,
    )

    # Estimate and reduce each group separately before building one metadata record
    started = perf_counter()
    results = {group: estimate(selected, **settings) for group, selected in groups.items()}

    # Align the fine grid separately while preserving each group's estimated interval
    sampling = None
    if "fast" in results:
        interval_ratio, subdivision_factor, aligned_interval = align_multiscale_sampling(
            results["slow"].sampling_interval, results["fast"].sampling_interval,
        )
        sampling = dict(
            interval_ratio=interval_ratio, subdivision_factor=subdivision_factor,
            aligned_sampling_interval=aligned_interval,
        )

    estimation_seconds = perf_counter() - started
    metadata = build_estimation_metadata(
        source_trajectory=str(source),
        source_sha256=source_hash,
        settings=settings,
        slow_result=results["slow"], fast_result=results.get("fast"),
        sampling=sampling,
        estimation_seconds=estimation_seconds,
    )
    metadata_text = json.dumps(metadata, indent=2, allow_nan=False) + "\n"

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".sampling-estimate-", dir=output.parent) as temporary:
        staged = Path(temporary) / output.name
        staged.write_text(metadata_text, encoding="utf-8")
        staged.replace(output)

    n_channels = sum(selected.state.shape[0] for selected in groups.values())
    print(f"Estimated bandwidth and sampling results for {n_channels} channels in {estimation_seconds:.2f}s: {output}")


if __name__ == "__main__":
    main()
