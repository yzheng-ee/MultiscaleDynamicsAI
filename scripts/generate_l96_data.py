"""
Generate the Lorenz '96 datasets this project's experiments run on.

* l96_multiscale_eps<...>.npz - the full two-scale system at a sweep of scale-separation parameters. 
* closure_<name>.joblib - closures m fitted to the reference eps = 2^-7 run.
* l96_singlescale_<name>.npz - the averaged model driven by each closure.
* timescales.json - dominant period and Lyapunov exponents, needed to put every forecast horizon on a comparable axis.

Sampling:
The default --sample-dt is chosen so the *slow* variables are sampled at roughly PANDA's training density 
(~102 points per dominant period, where the Lorenz '96 slow period is ~1.54 time units). That is a modelling decision, not
a detail: PANDA is not scale-free in time, and feeding it a trajectory sampled 10x more finely is itself a distribution shift.


<PSEUDO_CODE>
CONSTANTS
    REFERENCE_EPS = 2^-7            # Example B.1 (Calvello/Reich/Stuart 2025)
    SLOW_PERIOD   = 1.54            # measured slow-variable dominant period
    default sample_dt = 1.54 / 102.4 ≈ 0.015
                                    # 102.4 = PANDA's training density (pts/period)

MAIN
    args ← parse CLI
        --eps             [2^-7, 2^-5, 2^-3, 2^-1]   scale-separation sweep
        --n-trajectories  32
        --t-total 60      --t-burn 20
        --sample-dt       0.015     storage interval (NOT the integrator step)
        --seed 0  --skip-lyapunov  --skip-existing  --out-dir data/
    mkdir out_dir
    print sample_dt and the implied points-per-period

    multiscale  ← STEP 1  generate_multiscale_sweep(args)
    closures    ← STEP 2  fit_closures(multiscale[min eps], args)
    singlescale ← STEP 3  generate_singlescale_runs(closures, args)
                  STEP 4  measure_timescales(multiscale, singlescale, closures, args)

STEP 1 — full two-scale system, one run per eps
    for eps in args.eps:
        path ← data/l96_multiscale_eps<eps>.npz
        if --skip-existing and path exists:
            load it, skip                     # the expensive stage; don't redo
        dt ← suggested_dt(eps) = 5e-3 · eps   # resolves the stiff fast ring
                                              # 3.9e-5 at eps=2^-7
        z0 ← 32 random ICs (slow ~ U[-5,10); each fast = its parent slow)
        z0 ← integrate 20 time units, discard   # burn-in onto the attractor
        traj ← integrate 60 more, storing every sample_dt
               # batched over all 32 trajectories at once
        save (32, ~4000, 81) + metadata{params, dt, sample_every, t_burn, seed}

STEP 2 — fit the closure m(x) on the eps = 2^-7 run
    for each trajectory: extract pairs (x_k, ybar_k)
        # x_k    = slow variable k
        # ybar_k = mean of its 8 fast variables  ← the term averaging replaces
    pool over time AND over ring position k     → ~1.15M pairs

    GP closure:
        fit GaussianProcessRegressor on 800 subsampled pairs (RBF, alpha=1)
        save closure_gp.joblib
        wrap in TabulatedClosure               # ~100x faster to evaluate;
                                               # a raw GP would make STEP 3 take hours
    cubic closure:
        least-squares degree-3 polyfit on all pairs
        also records residual_std ← what a memoryless closure cannot capture

    save 200k subsampled pairs → closure_pairs.npz

STEP 3 — single-scale (averaged) model, one run per closure
    params ← L96Params(eps = 2^-7)             # eps unused here; kept for metadata
    for name, closure in {gp, poly3}:
        # replaces h_x·ybar_k with h_x·m(x_k): 9-D, non-stiff, dt = 1e-3
        x0 ← 32 random ICs → burn in 20 → integrate 60, store every sample_dt
        save (32, ~4000, 9) → l96_singlescale_<name>.npz

STEP 4 — timescales, so horizons are comparable
    for each multiscale run:
        slow dominant period, points_per_period
        unless --skip-lyapunov:
            lambda_max via Benettin, t_renorm = 5·eps   # must track the FAST scale
            → this is the fast exponent, ~1/eps
    for each single-scale run:
        dominant period
        unless --skip-lyapunov: lambda_max via Benettin from the run's final state, t_renorm = 0.1
            → this is the SLOW exponent (~0.8), the real forecast yardstick
    write timescales.json
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
from msdyn.closures import (ClosurePairs,GaussianProcessClosure,PolynomialClosure,TabulatedClosure,gather_closure_pairs,)
from msdyn.data import generate_multiscale, generate_singlescale, load_dataset, save_dataset
from msdyn.data.generate import burn_in
from msdyn.diagnostics import dominant_period, max_lyapunov_exponent
from msdyn.diagnostics.spectra import PANDA_POINTS_PER_PERIOD
from msdyn.systems.integrate import suggested_dt
from msdyn.systems.l96 import L96Multiscale, L96Params, L96SingleScale

REFERENCE_EPS = 2.0**-7  # Example B.1 of Calvello, Reich & Stuart (2025)
SLOW_PERIOD = 1.54  # measured; see notebooks/01_l96_data_generation.ipynb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--eps",
        type=float,
        nargs="+",
        default=[2.0**-7, 2.0**-5, 2.0**-3, 2.0**-1],
        help="scale-separation sweep; 2^-7 is the reference value",
    )
    parser.add_argument("--n-trajectories", type=int, default=32)
    parser.add_argument("--t-total", type=float, default=60.0)
    parser.add_argument("--t-burn", type=float, default=20.0)
    parser.add_argument(
        "--sample-dt",
        type=float,
        default=SLOW_PERIOD / PANDA_POINTS_PER_PERIOD,
        help="storage interval; the default matches PANDA's training density",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-lyapunov", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="reuse multiscale datasets already on disk instead of regenerating them",
    )
    return parser.parse_args()

def generate_multiscale_sweep(args: argparse.Namespace) -> dict[float, object]:
    datasets = {}
    for eps in args.eps:
        params = L96Params(eps=eps)
        path = args.out_dir / f"l96_multiscale_eps{eps:.6g}.npz"
        if args.skip_existing and path.exists():
            datasets[eps] = load_dataset(path)
            print(f"[multiscale] eps=2^{np.log2(eps):.0f}  reusing {path}")
            continue
        started = time.perf_counter()
        print(f"[multiscale] eps=2^{np.log2(eps):.0f}  dt={suggested_dt(eps):.2e} ...", flush=True)
        dataset = generate_multiscale(
            params,
            n_trajectories=args.n_trajectories,
            t_total=args.t_total,
            t_burn=args.t_burn,
            sample_dt=args.sample_dt,
            seed=args.seed,
        )
        save_dataset(dataset, path)
        print(f"{dataset.describe()}")
        print(f"-> {path}  [{time.perf_counter() - started:.0f}s]", flush=True)
        datasets[eps] = dataset
    return datasets


def fit_closures(dataset, args: argparse.Namespace) -> dict[str, object]:
    """Fit m on pooled (x_k, ybar_k) pairs from the reference multiscale run."""

    pairs_per_traj = [gather_closure_pairs(dataset.trajectories[i], dataset.params) for i in range(dataset.n_trajectories)]
    pooled_x = np.concatenate([p.x for p in pairs_per_traj])
    pooled_y = np.concatenate([p.ybar for p in pairs_per_traj])
    pairs = ClosurePairs(pooled_x, pooled_y)
    print(f"[closure]  pooled {len(pairs):,} (x_k, ybar_k) pairs", flush=True)

    closures: dict[str, object] = {}

    gp = GaussianProcessClosure(max_pairs=800, seed=args.seed).fit(pairs)
    gp.save(args.out_dir / "closure_gp.joblib")
    print(f"[closure]  gp    {gp!r}", flush=True)
    # Integrate against a tabulated copy: a raw sklearn GP costs ~20 ms per call,
    # which at 4 calls per RK4 step makes a long run take hours. 
    # Tabulating agrees to ~1e-7 over the attractor and is ~100x faster. See closures/tabulated.py.
    gp_fast = TabulatedClosure.from_pairs(gp, pairs)
    print(f"[closure]  gp    tabulated as {gp_fast!r}", flush=True)
    closures["gp"] = gp_fast

    poly = PolynomialClosure(degree=3).fit(pairs)
    print(f"[closure]  poly3 {poly!r}", flush=True)
    closures["poly3"] = poly

    stored = pairs.subsample(200_000, seed=args.seed)
    np.savez_compressed(args.out_dir / "closure_pairs.npz", x=stored.x, ybar=stored.ybar)
    return closures


def generate_singlescale_runs(closures: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    params = L96Params(eps=REFERENCE_EPS)
    datasets = {}
    for name, closure in closures.items():
        started = time.perf_counter()
        dataset = generate_singlescale(
            params,
            closure,
            n_trajectories=args.n_trajectories,
            t_total=args.t_total,
            t_burn=args.t_burn,
            sample_dt=args.sample_dt,
            seed=args.seed,
            closure_name=name,
        )
        path = save_dataset(dataset, args.out_dir / f"l96_singlescale_{name}.npz")
        print(f"[single]   {name}: {dataset.describe()}")
        print(f"           -> {path}  [{time.perf_counter() - started:.0f}s]", flush=True)
        datasets[name] = dataset
    return datasets


def measure_timescales(multiscale, singlescale, closures, args: argparse.Namespace) -> dict:
    """Dominant periods and Lyapunov exponents, for both scales.

    The multiscale system's maximal Lyapunov exponent is set by the fast ring (roughly 1/eps), 
    so it says nothing about how far ahead the slow variables can be predicted. 
    The single-scale model's exponent is the right yardstick for slow-variable forecast horizons, and both are recorded here.
    """
    timescales: dict[str, dict] = {"multiscale": {}, "singlescale": {}}

    for eps, dataset in multiscale.items():
        entry = {
            "slow_dominant_period": dominant_period(dataset.slow[0], dataset.dt),
            "sample_dt": dataset.dt,
        }
        entry["points_per_period"] = entry["slow_dominant_period"] / dataset.dt
        if not args.skip_lyapunov:
            params = dataset.params
            system = L96Multiscale(params)
            dt = suggested_dt(eps)
            state = burn_in(system, system.default_initial_state(seed=args.seed), t_burn=2.0, dt=dt)
            exponent, _ = max_lyapunov_exponent(system, state, dt=dt, t_renorm=5 * eps, n_renorm=120, n_discard=20)
            entry["lambda_max"] = exponent
            entry["lyapunov_time"] = 1.0 / exponent
        timescales["multiscale"][f"{eps:.6g}"] = entry

    for name, dataset in singlescale.items():
        entry = {
            "dominant_period": dominant_period(dataset.trajectories[0], dataset.dt),
            "sample_dt": dataset.dt,
        }
        if not args.skip_lyapunov:
            system = L96SingleScale(dataset.params, closures[name])
            # Already on the attractor: start from the end of the generated run.
            state = dataset.trajectories[0, -1]
            exponent, _ = max_lyapunov_exponent(system, state, dt=1e-3, t_renorm=0.1, n_renorm=200, n_discard=20)
            entry["lambda_max"] = exponent
            entry["lyapunov_time"] = 1.0 / exponent
        timescales["singlescale"][name] = entry

    path = args.out_dir / "timescales.json"
    path.write_text(json.dumps(timescales, indent=2))
    print(f"[timescale] -> {path}")
    return timescales


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"sample_dt = {args.sample_dt:.5g} (~{SLOW_PERIOD / args.sample_dt:.0f} points per slow period)\n", flush=True,)

    multiscale = generate_multiscale_sweep(args)

    reference_eps = min(args.eps)
    closures = fit_closures(multiscale[reference_eps], args)
    singlescale = generate_singlescale_runs(closures, args)
    measure_timescales(multiscale, singlescale, closures, args)
    print("\ndone.")


if __name__ == "__main__":
    main()
