"""Generate datasets from the multiscale and learned Lorenz-96 models."""

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .closure import fit_closure, load_closure
from .model import L96M


def build_lorenz96_metadata(
    *,
    K: int,
    J: int,
    hx: float,
    hy: float,
    F: float,
    eps: float,
    seed: int,
    initial_min: float,
    initial_max: float,
    spinup_duration: float,
    learning_duration: float,
    single_scale_dynamics_duration: float,
    multiscale_dynamics_duration: float,
    single_scale_sampling_interval: float,
    multiscale_sampling_interval: float,
    single_scale_dynamics_max_step: float | None,
    multiscale_dynamics_max_step: float | None,
    learning_max_step: float,
    spinup_max_step: float,
    solver_method: str,
    process_noise: float,
    stencil_left: int,
    stencil_right: int,
    closure_sample_size: int,
    closure_kernel: str,
    closure_length_scale: float,
    closure_rbf_bounds: tuple[float, float],
    closure_matern_nu: float,
    closure_alpha: float,
    closure_optimizer_restarts: int,
    closure_random_state: int | None,
    output_dir: str | Path,
    closure_filename: str,
    single_scale_filename: str,
    multiscale_filename: str,
    metadata_filename: str,
    write_metadata: bool,
    train_closure: bool,
    closure_path: str | Path | None,
) -> dict[str, object]:
    """Return the complete, grouped input configuration for one generation run."""
    return {
        "schema_version": 1,
        "model_parameters": {
            "K": K,
            "J": J,
            "hx": hx,
            "hy": hy,
            "forcing": F,
            "epsilon": eps,
        },
        "initialization_parameters": {
            "seed": seed,
            "initial_min": initial_min,
            "initial_max": initial_max,
        },
        "integration_and_sampling_parameters": {
            "spinup_duration": spinup_duration,
            "learning_duration": learning_duration,
            "single_scale_dynamics_duration": single_scale_dynamics_duration,
            "multiscale_dynamics_duration": multiscale_dynamics_duration,
            "single_scale_sampling_interval": single_scale_sampling_interval,
            "multiscale_sampling_interval": multiscale_sampling_interval,
            "single_scale_dynamics_max_step": single_scale_dynamics_max_step,
            "multiscale_dynamics_max_step": multiscale_dynamics_max_step,
            "learning_max_step": learning_max_step,
            "spinup_max_step": spinup_max_step,
            "solver_method": solver_method,
        },
        "noise_parameters": {"process_noise": process_noise},
        "closure_input_parameters": {
            "stencil_left": stencil_left,
            "stencil_right": stencil_right,
        },
        "gaussian_process_parameters": {
            "closure_sample_size": closure_sample_size,
            "closure_kernel": closure_kernel,
            "closure_length_scale": closure_length_scale,
            "closure_rbf_bounds": list(closure_rbf_bounds),
            "closure_matern_nu": closure_matern_nu,
            "closure_alpha": closure_alpha,
            "closure_optimizer_restarts": closure_optimizer_restarts,
            "closure_random_state": closure_random_state,
        },
        "output_parameters": {
            "output_dir": str(output_dir),
            "closure_filename": closure_filename,
            "single_scale_filename": single_scale_filename,
            "multiscale_filename": multiscale_filename,
            "metadata_filename": metadata_filename,
            "skip_metadata": not write_metadata,
        },
        "closure_reuse_parameters": {
            "skip_closure_training": not train_closure,
            "closure_path": str(closure_path) if closure_path is not None else None,
        },
    }


def _sample_count(duration: float, sampling_interval: float) -> int:
    if duration <= 0 or sampling_interval <= 0:
        raise ValueError("duration and sampling_interval must be positive")
    return np.arange(0, duration, sampling_interval).shape[0]


def _resolve_dynamics_max_step(
    sampling_interval: float, dynamics_max_step: float | None
) -> float:
    if dynamics_max_step is None:
        return sampling_interval
    if dynamics_max_step <= 0:
        raise ValueError("dynamics_max_step must be positive")
    return dynamics_max_step


def _validate_process_noise(process_noise: float) -> None:
    if process_noise < 0:
        raise ValueError("process_noise must be nonnegative")


def generate_single_scale_data(
    model: L96M,
    initial_state: NDArray[np.float64],
    duration: float = 50.0,
    sampling_interval: float = 0.001,
    dynamics_max_step: float | None = None,
    process_noise: float = 0.0,
    solver_method: str = "RK45",
) -> NDArray[np.float64]:
    """Generate reduced-model observations before the exclusive duration bound."""
    _validate_process_noise(process_noise)
    n_samples = _sample_count(duration, sampling_interval)
    dynamics_max_step = _resolve_dynamics_max_step(
        sampling_interval, dynamics_max_step
    )
    states = np.zeros((model.K, n_samples))
    states[:, 0] = initial_state[: model.K]

    for n in range(n_samples - 1):
        solution = solve_ivp(
            model.regressed,
            [0, sampling_interval],
            states[:, n],
            method=solver_method,
            max_step=dynamics_max_step,
        )
        states[:, n + 1] = solution.y[:, -1]
        if process_noise > 0:
            states[:, n + 1] += np.random.normal(
                0, process_noise, size=solution.y.shape[0]
            )
    return states


def generate_multiscale_data(
    model: L96M,
    initial_state: NDArray[np.float64],
    duration: float = 50.0,
    sampling_interval: float = 0.001,
    dynamics_max_step: float | None = None,
    process_noise: float = 0.0,
    solver_method: str = "RK45",
) -> NDArray[np.float64]:
    """Generate full-model observations before the exclusive duration bound."""
    _validate_process_noise(process_noise)
    n_samples = _sample_count(duration, sampling_interval)
    dynamics_max_step = _resolve_dynamics_max_step(
        sampling_interval, dynamics_max_step
    )
    state_dimension = model.K + model.K * model.J
    states = np.zeros((state_dimension, n_samples))
    states[:, 0] = initial_state

    for n in range(n_samples - 1):
        solution = solve_ivp(
            model,
            [0, sampling_interval],
            states[:, n],
            method=solver_method,
            max_step=dynamics_max_step,
        )
        states[:, n + 1] = solution.y[:, -1]
        if process_noise > 0:
            states[:, n + 1] += np.random.normal(
                0, process_noise, size=solution.y.shape[0]
            )
    return states


def generate_lorenz96_data(
    *,
    K: int = 9,
    J: int = 8,
    hx: float = -0.8,
    hy: float = 1.0,
    F: float = 10.0,
    eps: float = 2**-7,
    seed: int = 42,
    initial_min: float = -5.0,
    initial_max: float = 10.0,
    spinup_duration: float = 50.0,
    learning_duration: float = 300.0,
    single_scale_dynamics_duration: float = 50.0,
    multiscale_dynamics_duration: float = 50.0,
    single_scale_sampling_interval: float = 0.001,
    multiscale_sampling_interval: float = 0.001,
    single_scale_dynamics_max_step: float | None = None,
    multiscale_dynamics_max_step: float | None = None,
    learning_max_step: float = 0.001,
    spinup_max_step: float = 0.01,
    solver_method: str = "RK45",
    process_noise: float = 0.0,
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
    single_scale_filename: str = "simulation_data_singlescale.npz",
    multiscale_filename: str = "simulation_data_multiscale.npz",
    metadata_filename: str = "metadata.json",
    write_metadata: bool = True,
    train_closure: bool = True,
    closure_path: str | Path | None = None,
) -> dict[str, Path]:
    """Run the Lorenz-96 data-generation pipeline and return written paths."""
    if initial_max <= initial_min:
        raise ValueError("initial_max must be greater than initial_min")
    if learning_max_step <= 0 or spinup_max_step <= 0:
        raise ValueError("maximum solver steps must be positive")
    _sample_count(single_scale_dynamics_duration, single_scale_sampling_interval)
    _sample_count(multiscale_dynamics_duration, multiscale_sampling_interval)
    requested_single_scale_dynamics_max_step = single_scale_dynamics_max_step
    requested_multiscale_dynamics_max_step = multiscale_dynamics_max_step
    single_scale_dynamics_max_step = _resolve_dynamics_max_step(
        single_scale_sampling_interval, single_scale_dynamics_max_step
    )
    multiscale_dynamics_max_step = _resolve_dynamics_max_step(
        multiscale_sampling_interval, multiscale_dynamics_max_step
    )
    _validate_process_noise(process_noise)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = L96M(K=K, J=J, hx=hx, hy=hy, F=F, eps=eps)
    model.set_stencil(stencil_left, stencil_right)
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
            max_step=learning_max_step,
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
        written["closure"] = Path(closure_path)

    model.set_predictor(regressor.predict)

    single_states = generate_single_scale_data(
        model=model,
        initial_state=generation_initial_state,
        duration=single_scale_dynamics_duration,
        sampling_interval=single_scale_sampling_interval,
        dynamics_max_step=single_scale_dynamics_max_step,
        process_noise=process_noise,
        solver_method=solver_method,
    )
    single_scale_path = output_dir / single_scale_filename
    np.savez(single_scale_path, states=single_states)
    written["single-scale"] = single_scale_path

    multiscale_states = generate_multiscale_data(
        model=model,
        initial_state=generation_initial_state,
        duration=multiscale_dynamics_duration,
        sampling_interval=multiscale_sampling_interval,
        dynamics_max_step=multiscale_dynamics_max_step,
        process_noise=process_noise,
        solver_method=solver_method,
    )
    multiscale_path = output_dir / multiscale_filename
    np.savez(multiscale_path, states=multiscale_states)
    written["multiscale"] = multiscale_path

    if write_metadata:
        metadata = build_lorenz96_metadata(
            K=K,
            J=J,
            hx=hx,
            hy=hy,
            F=F,
            eps=eps,
            seed=seed,
            initial_min=initial_min,
            initial_max=initial_max,
            spinup_duration=spinup_duration,
            learning_duration=learning_duration,
            single_scale_dynamics_duration=single_scale_dynamics_duration,
            multiscale_dynamics_duration=multiscale_dynamics_duration,
            single_scale_sampling_interval=single_scale_sampling_interval,
            multiscale_sampling_interval=multiscale_sampling_interval,
            single_scale_dynamics_max_step=requested_single_scale_dynamics_max_step,
            multiscale_dynamics_max_step=requested_multiscale_dynamics_max_step,
            learning_max_step=learning_max_step,
            spinup_max_step=spinup_max_step,
            solver_method=solver_method,
            process_noise=process_noise,
            stencil_left=stencil_left,
            stencil_right=stencil_right,
            closure_sample_size=closure_sample_size,
            closure_kernel=closure_kernel,
            closure_length_scale=closure_length_scale,
            closure_rbf_bounds=closure_rbf_bounds,
            closure_matern_nu=closure_matern_nu,
            closure_alpha=closure_alpha,
            closure_optimizer_restarts=closure_optimizer_restarts,
            closure_random_state=closure_random_state,
            output_dir=output_dir,
            closure_filename=closure_filename,
            single_scale_filename=single_scale_filename,
            multiscale_filename=multiscale_filename,
            metadata_filename=metadata_filename,
            write_metadata=write_metadata,
            train_closure=train_closure,
            closure_path=closure_path,
        )
        metadata_path = output_dir / metadata_filename
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        written["metadata"] = metadata_path

    return written
