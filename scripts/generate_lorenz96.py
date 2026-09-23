#!/usr/bin/env python3
"""Command-line interface for Lorenz-96 data generation."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.lorenz96.generate import DEFAULT_DATASETS, generate_lorenz96_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--K", type=int, default=9, help="number of slow variables")
    parser.add_argument("--J", type=int, default=8, help="fast variables per slow variable")
    parser.add_argument("--hx", type=float, default=-0.8)
    parser.add_argument("--hy", type=float, default=1.0)
    parser.add_argument("--forcing", type=float, default=10.0)
    parser.add_argument("--epsilon", type=float, default=2**-7)
    parser.add_argument("--k0", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-min", type=float, default=-5.0)
    parser.add_argument("--initial-max", type=float, default=10.0)
    parser.add_argument("--spinup-duration", type=float, default=50.0)
    parser.add_argument("--learning-duration", type=float, default=300.0)
    parser.add_argument("--dynamics-duration", type=float, default=50.0)
    parser.add_argument("--inversion-duration", type=float, default=100.0)
    parser.add_argument("--tau", type=float, default=0.001)
    parser.add_argument("--dynamics-max-step", type=float, default=0.001)
    parser.add_argument("--spinup-max-step", type=float, default=0.01)
    parser.add_argument("--solver-method", default="RK45")
    parser.add_argument("--small-process-noise", type=float, default=0.001)
    parser.add_argument("--small-observation-noise", type=float, default=0.001)
    parser.add_argument("--large-process-noise", type=float, default=0.1)
    parser.add_argument("--large-observation-noise", type=float, default=0.1)
    parser.add_argument(
        "--observation-indices", type=int, nargs="+", default=[0, 1, 3, 4, 6, 7]
    )
    parser.add_argument("--stencil-left", type=int, default=0)
    parser.add_argument("--stencil-right", type=int, default=0)
    parser.add_argument("--closure-sample-size", type=int, default=800)
    parser.add_argument("--closure-kernel", choices=("rbf", "matern"), default="rbf")
    parser.add_argument("--closure-length-scale", type=float, default=3.0)
    parser.add_argument(
        "--closure-rbf-bounds", type=float, nargs=2, default=(1e-10, 1e6)
    )
    parser.add_argument("--closure-matern-nu", type=float, default=1.5)
    parser.add_argument("--closure-alpha", type=float, default=1.0)
    parser.add_argument("--closure-optimizer-restarts", type=int, default=15)
    parser.add_argument("--closure-random-state", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--closure-filename", default="closure.joblib")
    parser.add_argument(
        "--single-small-filename", default="simulation_data_singlescale_001.npz"
    )
    parser.add_argument(
        "--single-large-filename", default="simulation_data_singlescale_1.npz"
    )
    parser.add_argument(
        "--multiscale-filename", default="simulation_data_multiscale_001.npz"
    )
    parser.add_argument(
        "--inversion-filename", default="simulation_data_singlescale_inversion.npz"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DEFAULT_DATASETS,
        default=list(DEFAULT_DATASETS),
    )
    parser.add_argument("--skip-closure-training", action="store_true")
    parser.add_argument("--closure-path", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    written = generate_lorenz96_data(
        K=args.K,
        J=args.J,
        hx=args.hx,
        hy=args.hy,
        F=args.forcing,
        eps=args.epsilon,
        k0=args.k0,
        seed=args.seed,
        initial_min=args.initial_min,
        initial_max=args.initial_max,
        spinup_duration=args.spinup_duration,
        learning_duration=args.learning_duration,
        dynamics_duration=args.dynamics_duration,
        inversion_duration=args.inversion_duration,
        tau=args.tau,
        dynamics_max_step=args.dynamics_max_step,
        spinup_max_step=args.spinup_max_step,
        solver_method=args.solver_method,
        small_process_noise=args.small_process_noise,
        small_observation_noise=args.small_observation_noise,
        large_process_noise=args.large_process_noise,
        large_observation_noise=args.large_observation_noise,
        observation_indices=args.observation_indices,
        stencil_left=args.stencil_left,
        stencil_right=args.stencil_right,
        closure_sample_size=args.closure_sample_size,
        closure_kernel=args.closure_kernel,
        closure_length_scale=args.closure_length_scale,
        closure_rbf_bounds=tuple(args.closure_rbf_bounds),
        closure_matern_nu=args.closure_matern_nu,
        closure_alpha=args.closure_alpha,
        closure_optimizer_restarts=args.closure_optimizer_restarts,
        closure_random_state=args.closure_random_state,
        output_dir=args.output_dir,
        closure_filename=args.closure_filename,
        single_small_filename=args.single_small_filename,
        single_large_filename=args.single_large_filename,
        multiscale_filename=args.multiscale_filename,
        inversion_filename=args.inversion_filename,
        datasets=args.datasets,
        train_closure=not args.skip_closure_training,
        closure_path=args.closure_path,
    )
    for name, path in written.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
