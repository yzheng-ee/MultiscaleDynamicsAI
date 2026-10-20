"""Independent deterministic simulations; no closure training or file I/O."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .model import Array


@dataclass(frozen=True)
class Trajectory:
    """Samples after burn-in, at their actual integration times.

    time has shape (N,) and state has shape (D, N). The model defines the
    state layout. state[:, -1] is the state at the last saved timestamp,
    time[-1]. It is the integration endpoint only when endpoint=True.
    """

    time: Array
    state: Array


def simulate_ode(
    rhs: Callable[[float, Array], Array],
    initial_state: Array,
    *,
    # Simulation interval and sampling
    duration: float,
    sample_dt: float,
    burn_in: float = 0.0,
    t0: float = 0.0,
    endpoint: bool = True,

    # Numerical solver settings
    method: str = "DOP853",
    rtol: float = 1e-8,
    atol: float | Array = 1e-10,
    max_step: float | None = None,
) -> Trajectory:
    """Integrate a real-valued ODE rhs(t, state) and return uniform samples.

    initial_state is a nonempty vector at t0. Burn-in runs from t0 to
    t0 + burn_in; the recorded interval then lasts duration. Returned times
    are absolute. endpoint=True (the default) includes both endpoints;
    endpoint=False excludes the final endpoint from returned samples. The solver
    still integrates over the full duration. duration must be an integer
    multiple of sample_dt. Internal steps remain adaptive and need not align
    with sampling times. max_step=None sets the step-size cap to sample_dt;
    an explicit value overrides it. Solver controls apply to both intervals.
    """
    state = np.asarray(initial_state, dtype=float).copy()
    if state.ndim != 1 or state.size == 0 or not np.isfinite(state).all():
        raise ValueError("initial_state must be a nonempty finite vector")
    dimension = state.size
    if max_step is None:
        max_step = sample_dt
    if np.isnan(max_step) or max_step <= 0:
        raise ValueError("max_step must be positive (np.inf is allowed)")
    if not np.isfinite(rtol) or rtol <= 0:
        raise ValueError("rtol must be finite and positive")
    atol = np.asarray(atol, dtype=float)
    if (
        atol.shape not in ((), (dimension,))
        or not np.isfinite(atol).all()
        or np.any(atol < 0)
    ):
        raise ValueError(f"atol must be finite and nonnegative, either scalar or shape ({dimension},)")

    for name, value in (("duration", duration), ("sample_dt", sample_dt)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(burn_in) or burn_in < 0:
        raise ValueError("burn_in must be finite and nonnegative")
    if not np.isfinite(t0):
        raise ValueError("t0 must be finite")
    if not isinstance(endpoint, (bool, np.bool_)):
        raise ValueError("endpoint must be a boolean")
    interval_count = duration / sample_dt
    if not np.isfinite(interval_count):
        raise ValueError("duration / sample_dt must be finite")
    n_intervals = round(interval_count)
    if n_intervals < 1 or not np.isclose(interval_count, n_intervals, rtol=1e-12, atol=0):
        raise ValueError("duration must be an integer multiple of sample_dt")
    start = t0 + burn_in
    end = start + duration
    if not np.isfinite(start) or not np.isfinite(end):
        raise ValueError("integration times must be finite")
    if end <= start:
        raise ValueError("duration is too small to resolve at the specified start time")
    times = np.linspace(start, end, n_intervals + int(endpoint), endpoint=endpoint)
    if np.any(np.diff(times) <= 0) or (not endpoint and times[-1] >= end):
        raise ValueError("sample_dt is too small to resolve at the specified start time")

    def solve(state, start, end, times):
        result = solve_ivp(
            rhs,
            (start, end),
            state,
            t_eval=times,
            method=method,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )
        if not result.success or not np.isfinite(result.y).all():
            raise RuntimeError(f"ODE integration failed: {result.message}")
        return result.y

    if burn_in:
        if start <= t0:
            raise ValueError("burn_in is too small to resolve at t0")
        state = solve(state, t0, start, [start])[:, -1]
    states = solve(state, start, end, times)
    return Trajectory(times, states)
