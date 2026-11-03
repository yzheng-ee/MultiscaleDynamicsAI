"""Lorenz–96 models, trajectory generation, simulation, and local closure fitting.

Example (the caller chooses and composes the operations)::

    model = MultiscaleLorenz96()
    initial_state = sample_initial_state(model, seed=42)
    training = simulate_ode(
        model.rhs,
        initial_state=initial_state,
        duration=300,
        sample_dt=0.01,
        burn_in=50,
        max_step=model.eps / 4,
    )
    slow, fast = model.split_state(training.state)
    closure = fit_closure(slow, fast, seed=42)
    reduced_model = SingleScaleLorenz96(closure=closure, K=model.K,
                                        hx=model.hx, F=model.F)
    reduced = simulate_ode(
        reduced_model.rhs,
        initial_state=sample_initial_state(reduced_model, seed=43),
        max_step=0.05,
        duration=50,
        sample_dt=0.001,
    )

Samples always include both endpoints.
Specify sample_dt and exactly one of duration or n_samples; N samples span
(N - 1) * sample_dt units of time.
For either system, continue from the last saved state with
initial_state=trajectory.state[:, -1] and t0=trajectory.time[-1].
simulate_ode also accepts a general rhs(t, state).
"""

from .model import MultiscaleLorenz96, SingleScaleLorenz96, sample_initial_state
from .simulate import Trajectory, simulate_ode
from .closure import GPClosure, fit_closure
from .generate import generate_multiscale_trajectory, generate_singlescale_trajectory
from .downsample import downsample_trajectory

__all__ = ["MultiscaleLorenz96", "SingleScaleLorenz96", "Trajectory", "simulate_ode", "sample_initial_state", "GPClosure", "fit_closure", "generate_multiscale_trajectory", "generate_singlescale_trajectory", "downsample_trajectory"]
