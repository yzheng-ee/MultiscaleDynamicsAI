"""Estimate 99% AC-energy bandwidth periods for calibration configuration A.

Run using the panda_pilot environment:
    python scripts/lorenz96_benchmark/estimate_bandwidth_periods_a.py

Writes bandwidth_periods.json beside the dense calibration trajectory. Periods
are reciprocals of channel bandwidth frequencies, in simulation-time units.
Channel order follows the source metadata. Zero bandwidth implies an unbounded
period, recorded as JSON null. Slow/fast summaries are medians of the channel
periods, including unbounded periods. No sampling intervals are calculated.
"""

import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.lorenz96 import Trajectory
from src.data.sampling_interval import estimate_trajectory_bandwidths


def main():
    directory = ROOT / "outputs/lorenz96_benchmark/dense_calibration_trajectory_a"
    metadata = json.loads((directory / "metadata.json").read_text())
    with np.load(directory / metadata["filename"]) as data:
        trajectory = Trajectory(time=data["time"], state=data["state"])

    K = metadata["parameters"]["K"]
    J = metadata["parameters"]["J"]
    if trajectory.state.shape != (K * (1 + J), trajectory.time.size):
        raise ValueError("Trajectory shape does not match K, J, and time.")

    print("Estimating 99% AC-energy bandwidth periods...", flush=True)
    started = perf_counter()
    bandwidths = estimate_trajectory_bandwidths(
        trajectory,
        cumulative_energy_cutoff=0.99,
    )
    periods = np.full(bandwidths.shape, np.inf, dtype=np.float64)
    np.divide(1.0, bandwidths, out=periods, where=bandwidths > 0)
    slow_period = float(np.median(periods[:K]))
    fast_period = float(np.median(periods[K:]))
    elapsed = perf_counter() - started

    record = {
        "config_id": metadata["config_id"],
        "source_trajectory": metadata["filename"],
        "source_metadata": "metadata.json",
        "cumulative_energy_cutoff": 0.99,
        "energy_component": "AC (DC excluded)",
        "period_units": "simulation time",
        "channel_bandwidth_periods": [
            float(period) if np.isfinite(period) else None for period in periods
        ],
        "reduction": "median of channel bandwidth periods",
        "slow_bandwidth_period": slow_period if np.isfinite(slow_period) else None,
        "fast_bandwidth_period": fast_period if np.isfinite(fast_period) else None,
        "estimation_seconds": elapsed,
    }
    output_path = directory / "bandwidth_periods.json"
    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary_path.replace(output_path)
    print(f"Slow median bandwidth period: {slow_period:.8g}", flush=True)
    print(f"Fast median bandwidth period: {fast_period:.8g}", flush=True)
    print(f"Saved {periods.size} channel periods to {output_path}", flush=True)


if __name__ == "__main__":
    main()
