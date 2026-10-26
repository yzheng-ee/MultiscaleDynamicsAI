"""Estimate 99% AC-energy bandwidth periods for all six single-scale settings.

Run from the repository root using environments/lorenz96_benchmark.yml:
    python scripts/lorenz96_benchmark/estimate_singlescale_bandwidth_periods.py

Processes A and all five Test B configurations sequentially, writing
bandwidth_periods.json beside each single-scale dense calibration trajectory.
Periods are reciprocals of channel bandwidth frequencies in simulation-time
units. Channel order follows the source metadata. Zero bandwidth implies an
unbounded period, recorded as JSON null. The summary is the median of
all K channel periods, including unbounded periods. There are no fast channels
or fast-period summary. No sampling intervals are calculated. Running this
script replaces existing bandwidth-period JSON files.
"""

import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.lorenz96 import Trajectory
from src.data.sampling_interval import estimate_trajectory_bandwidths


def main():
    run_started = perf_counter()
    settings = ("a", "b_f", "b_hx", "b_hy", "b_j", "b_k")
    for subset in tqdm(settings, desc="Single-scale bandwidths", unit="configuration"):
        directory = ROOT / "data/lorenz96_benchmark/dense_calibration/singlescale" / f"singlescale_dense_calibration_trajectory_{subset}"
        metadata = json.loads((directory / "metadata.json").read_text())
        with np.load(directory / metadata["filename"], allow_pickle=False) as data:
            trajectory = Trajectory(time=data["time"], state=data["state"])

        K = metadata["parameters"]["K"]
        if trajectory.state.shape != (K, trajectory.time.size):
            raise ValueError("Single-scale trajectory shape does not match K and time.")

        started = perf_counter()
        bandwidths = estimate_trajectory_bandwidths(
            trajectory,
            cumulative_energy_cutoff=0.99,
        )
        periods = np.full(bandwidths.shape, np.inf, dtype=np.float64)
        np.divide(1.0, bandwidths, out=periods, where=bandwidths > 0)
        bandwidth_period = float(np.median(periods))
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
            "bandwidth_period": bandwidth_period if np.isfinite(bandwidth_period) else None,
            "estimation_seconds": elapsed,
        }
        output_path = directory / "bandwidth_periods.json"
        temporary_path = output_path.with_suffix(".json.tmp")
        temporary_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        temporary_path.replace(output_path)
        del trajectory, bandwidths, periods

    print(f"Single-scale bandwidth estimation finished: {len(settings)} records saved in {perf_counter() - run_started:.2f}s.", flush=True)


if __name__ == "__main__":
    main()
