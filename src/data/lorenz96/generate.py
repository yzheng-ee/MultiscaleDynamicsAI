"""Generate datasets from the multiscale and learned Lorenz-96 models."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .closure import fit_closure, load_closure
from .model import L96M

DEFAULT_DATASETS = ("single-small", "single-large", "multiscale", "inversion")


def _sample_count(duration: float, tau: float) -> int:
    if duration <= 0 or tau <= 0:
        raise ValueError("durations and tau must be positive")
    return np.arange(0, duration, tau).shape[0]


def _validate_observation_indices(indices: Sequence[int], K: int) -> None:
    if not indices:
        raise ValueError("at least one observation index is required")
    if any(index < 0 or index >= K for index in indices):
        raise ValueError(f"observation indices must be between 0 and {K - 1}")


def generate_single_scale_data(
    model: L96M,
    initial_state: NDArray[np.float64],
    duration: float = 50.0,
    tau: float = 0.001,
    process_noise: float = 0.001,
    observation_noise: float = 0.001,
    observation_indices: Sequence[int] = (0, 1, 3, 4, 6, 7),
    solver_method: str = "RK45",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate a noisy trajectory and observations from the reduced model."""
    _validate_observation_indices(observation_indices, model.K)
    n_samples = _sample_count(duration, tau)
    states = np.zeros((model.K, n_samples))
    observations = np.zeros((len(observation_indices), n_samples))
    states[:, 0] = initial_state[: model.K]

    for n in range(n_samples - 1):
        solution = solve_ivp(
            model.regressed,
            [0, tau],
            states[:, n],
            method=solver_method,
            max_step=tau,
        )
        states[:, n + 1] = solution.y[:, -1] + np.random.normal(
            0, process_noise, size=solution.y.shape[0]
        )
        observations[:, n + 1] = states[list(observation_indices), n + 1] + (
            np.random.normal(0, observation_noise, size=len(observation_indices))
        )
    return states, observations


def generate_multiscale_data(
    model: L96M,
    initial_state: NDArray[np.float64],
    duration: float = 50.0,
    tau: float = 0.001,
    process_noise: float = 0.001,
    observation_noise: float = 0.001,
    observation_indices: Sequence[int] = (0, 1, 3, 4, 6, 7),
    solver_method: str = "RK45",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate a noisy full multiscale trajectory and slow observations."""
    _validate_observation_indices(observation_indices, model.K)
    n_samples = _sample_count(duration, tau)
    state_dimension = model.K + model.K * model.J
    states = np.zeros((state_dimension, n_samples))
    observations = np.zeros((len(observation_indices), n_samples))
    states[:, 0] = initial_state

    for n in range(n_samples - 1):
        solution = solve_ivp(
            model,
            [0, tau],
            states[:, n],
            method=solver_method,
            max_step=tau,
        )
        states[:, n + 1] = solution.y[:, -1] + np.random.normal(
            0, process_noise, size=solution.y.shape[0]
        )
        observations[:, n + 1] = states[list(observation_indices), n + 1] + (
            np.random.normal(0, observation_noise, size=len(observation_indices))
        )
    return states, observations


def generate_inversion_data(
    model: L96M,
    initial_state: NDArray[np.float64],
    duration: float = 100.0,
    tau: float = 0.001,
    solver_method: str = "RK45",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate the deterministic reduced trajectory and its summary statistics."""
    n_samples = _sample_count(duration, tau)
    states = np.zeros((model.K, n_samples))
    states[:, 0] = initial_state[: model.K]

    for n in range(n_samples - 1):
        solution = solve_ivp(
            model.regressed,
            [0, tau],
            states[:, n],
            method=solver_method,
            max_step=tau,
        )
        states[:, n + 1] = solution.y[:, -1]

    mean = states.sum(axis=1).sum(axis=0) / (model.K * n_samples)
    variance = np.var(states, axis=1).sum(axis=0) / model.K
    return states, np.array([mean, variance])


def generate_lorenz96_data(
    *,
    K: int = 9,
    J: int = 8,
    hx: float = -0.8,
    hy: float = 1.0,
    F: float = 10.0,
    eps: float = 2**-7,
    k0: int = 0,
    seed: int = 42,
    initial_min: float = -5.0,
    initial_max: float = 10.0,
    spinup_duration: float = 50.0,
    learning_duration: float = 300.0,
    dynamics_duration: float = 50.0,
    inversion_duration: float = 100.0,
    tau: float = 0.001,
    dynamics_max_step: float = 0.001,
    spinup_max_step: float = 0.01,
    solver_method: str = "RK45",
    small_process_noise: float = 0.001,
    small_observation_noise: float = 0.001,
    large_process_noise: float = 0.1,
    large_observation_noise: float = 0.1,
    observation_indices: Sequence[int] = (0, 1, 3, 4, 6, 7),
    stencil_left: int = 0,
    stencil_right: int = 0,
    closure_sample_size: int = 800,
    closure_kernel: str = "rbf",
    closure_length_scale: float = 3.0,
    closure_rbf_bounds: tuple[float, float] = (1e-10, 1e6),
    closure_matern_nu: float = 1.5,
    closure_alpha: float = 1.0,
    closure_optimizer_restarts: int = 15,
    closure_random_state: int | None = None,
    output_dir: str | Path = ".",
    closure_filename: str = "closure.joblib",
    single_small_filename: str = "simulation_data_singlescale_001.npz",
    single_large_filename: str = "simulation_data_singlescale_1.npz",
    multiscale_filename: str = "simulation_data_multiscale_001.npz",
    inversion_filename: str = "simulation_data_singlescale_inversion.npz",
    datasets: Sequence[str] = DEFAULT_DATASETS,
    train_closure: bool = True,
    closure_path: str | Path | None = None,
) -> dict[str, Path]:
    """Run the Lorenz-96 data-generation pipeline and return written paths."""
    requested = tuple(datasets)
    unknown = set(requested) - set(DEFAULT_DATASETS)
    if unknown:
        raise ValueError(f"unknown datasets: {', '.join(sorted(unknown))}")
    if initial_max <= initial_min:
        raise ValueError("initial_max must be greater than initial_min")
    if dynamics_max_step <= 0 or spinup_max_step <= 0:
        raise ValueError("maximum solver steps must be positive")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = L96M(K=K, J=J, hx=hx, hy=hy, F=F, eps=eps, k0=k0)
    model.set_stencil(stencil_left, stencil_right)
    _validate_observation_indices(observation_indices, K)
    np.random.seed(seed)

    initial_state = np.empty(K + K * J)
    initial_state[:K] = np.random.rand(K) * (initial_max - initial_min) + initial_min
    for k in range(K):
        initial_state[K + k * J : K + (k + 1) * J] = initial_state[k]

    spinup = solve_ivp(
        model,
        [0, spinup_duration],
        initial_state,
        method=solver_method,
        max_step=spinup_max_step,
    )
    spun_up_state = spinup.y[:, -1]

    written: dict[str, Path] = {}
    destination_closure = output_dir / closure_filename
    if train_closure:
        learning = solve_ivp(
            model,
            [0, learning_duration],
            spun_up_state,
            method=solver_method,
            max_step=dynamics_max_step,
        )
        generation_initial_state = learning.y[:, -1]
        pairs = model.gather_pairs(learning.y)
        regressor = fit_closure(
            pairs,
            sample_size=closure_sample_size,
            kernel=closure_kernel,
            length_scale=closure_length_scale,
            rbf_length_scale_bounds=closure_rbf_bounds,
            matern_nu=closure_matern_nu,
            alpha=closure_alpha,
            optimizer_restarts=closure_optimizer_restarts,
            output_path=destination_closure,
            random_state=closure_random_state,
        )
        written["closure"] = destination_closure
    else:
        if closure_path is None:
            raise ValueError("closure_path is required when train_closure is false")
        regressor = load_closure(closure_path)
        generation_initial_state = spun_up_state

    model.set_predictor(regressor.predict)

    jobs: dict[str, tuple[str, Any]] = {
        "single-small": (
            single_small_filename,
            lambda: generate_single_scale_data(
                model,
                generation_initial_state,
                dynamics_duration,
                tau,
                small_process_noise,
                small_observation_noise,
                observation_indices,
                solver_method,
            ),
        ),
        "single-large": (
            single_large_filename,
            lambda: generate_single_scale_data(
                model,
                generation_initial_state,
                dynamics_duration,
                tau,
                large_process_noise,
                large_observation_noise,
                observation_indices,
                solver_method,
            ),
        ),
        "multiscale": (
            multiscale_filename,
            lambda: generate_multiscale_data(
                model,
                generation_initial_state,
                dynamics_duration,
                tau,
                small_process_noise,
                small_observation_noise,
                observation_indices,
                solver_method,
            ),
        ),
        "inversion": (
            inversion_filename,
            lambda: generate_inversion_data(
                model,
                generation_initial_state,
                inversion_duration,
                tau,
                solver_method,
            ),
        ),
    }

    for name in requested:
        filename, generate = jobs[name]
        output_path = output_dir / filename
        states, data = generate()
        if name == "inversion":
            np.savez(output_path, states=states, observation=data)
        else:
            np.savez(output_path, states=states, observations=data)
        written[name] = output_path

    return written
