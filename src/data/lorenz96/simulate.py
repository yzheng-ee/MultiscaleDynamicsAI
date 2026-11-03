"""Numerical integration of deterministic ODEs with uniform trajectory sampling."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .._utils.validation import _validate_finite, _validate_integer, _validate_nonnegative, _validate_positive, _validate_real
from .model import Array


@dataclass(frozen=True)
class Trajectory:
    """Samples after burn-in, at their actual integration times.

    time has shape (N,) and state has shape (D, N). The model defines the
    state layout. state[:, -1] is the state at the last saved timestamp,
    time[-1], which is always the integration endpoint.
    """

    time: Array
    state: Array


def simulate_ode(
    rhs: Callable[[float, Array], Array],
    initial_state: Array,
    *,
    # Simulation interval and sampling
    sample_dt: float,
    duration: float | None = None,
    n_samples: int | None = None,
    burn_in: float = 0.0,
    t0: float = 0.0,

    # Numerical solver settings
    method: str = "DOP853",
    rtol: float = 1e-8,
    atol: float | Array = 1e-10,
    max_step: float | None = None,
) -> Trajectory:
    """Integrate a real-valued ODE rhs(t, state) and return uniform samples.

    initial_state is a nonempty vector at t0. Burn-in runs from t0 to
    t0 + burn_in. Returned times are absolute and always include both endpoints.
    Supply exactly one of duration or n_samples. duration must be a positive
    integer multiple of sample_dt; n_samples must be an integer >= 2 and implies
    duration = (n_samples - 1) * sample_dt. Internal steps remain adaptive and
    need not align with sampling times. max_step=None sets the step-size cap
    to sample_dt; an explicit value overrides it. Solver controls apply to
    both intervals. Sampling grids must remain uniform and match sample_dt
    within a relative tolerance of 1e-7 at the specified absolute times.
    """
    # Validate the initial state and copy it to preserve the caller's array
    _validate_real([("initial_state", initial_state)])
    state = np.asarray(initial_state, dtype=float).copy()
    if state.ndim != 1 or state.size == 0:
        raise ValueError("initial_state must be a nonempty vector")
    _validate_finite([("initial_state", state)])
    dimension = state.size

    # Require positive sampling spacing and exactly one length specification
    _validate_finite([("sample_dt", sample_dt)])
    _validate_positive([("sample_dt", sample_dt)])
    if (duration is None) == (n_samples is None):
        raise ValueError("Supply exactly one of duration or n_samples")
    if n_samples is not None:
        _validate_integer([("n_samples", n_samples)])
        if n_samples < 2:
            raise ValueError("n_samples must be >= 2")
        # Including both endpoints gives N - 1 intervals for N samples
        n_intervals = int(n_samples) - 1
        duration = n_intervals * sample_dt
    else:
        _validate_finite([("duration", duration)])
        _validate_positive([("duration", duration)])
        interval_count = duration / sample_dt
        _validate_finite([("duration / sample_dt", interval_count)])
        # Allow floating-point roundoff without accepting a fractional interval
        n_intervals = round(interval_count)
        if n_intervals < 1 or not np.isclose(interval_count, n_intervals, rtol=1e-12, atol=0):
            raise ValueError("duration must be an integer multiple of sample_dt")
    # Count-derived durations must also remain finite and positive
    _validate_finite([("duration", duration)])
    _validate_positive([("duration", duration)])

    # Validate solver controls; infinity explicitly disables the step-size cap
    if max_step is None:
        max_step = sample_dt  # By default, cap the integration step at the sampling interval
    _validate_positive([("max_step", max_step)])
    _validate_finite([("rtol", rtol)])
    _validate_positive([("rtol", rtol)])
    # Absolute tolerances may be scalar or one value per state component
    _validate_real([("atol", atol)])
    atol = np.asarray(atol, dtype=float)
    if atol.shape not in ((), (dimension,)):
        raise ValueError(f"atol must be scalar or have shape ({dimension},)")
    _validate_finite([("atol", atol)])
    _validate_nonnegative([("atol", atol)])

    # Validate the time origin and optional burn-in duration
    _validate_finite([("burn_in", burn_in), ("t0", t0)])
    _validate_nonnegative([("burn_in", burn_in)])

    # Finite inputs can still overflow or lose small durations at large times
    start = t0 + burn_in
    end = start + duration
    _validate_finite([("integration start", start), ("integration end", end)])
    if end <= start:
        raise ValueError("duration is too small to resolve at the specified start time")

    # Reject grids whose adjacent timestamps collapse at floating-point precision
    times = np.linspace(start, end, n_intervals + 1)
    intervals = np.diff(times)
    if np.any(intervals <= 0):
        raise ValueError("sample_dt is too small to resolve at the specified start time")

    # Distinct timestamps can still lose uniform spacing or shift the requested interval
    recorded_dt = (end - start) / n_intervals
    if not np.isclose(recorded_dt, sample_dt, rtol=1e-7, atol=0) or not np.allclose(intervals, recorded_dt, rtol=1e-7, atol=0):
        raise ValueError("sample_dt cannot be represented uniformly at the specified start time")

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
        # Reject failed integration and nonfinite states in either simulation phase
        if not result.success or not np.isfinite(result.y).all():
            raise RuntimeError(f"ODE integration failed: {result.message}")
        return result.y

    if burn_in:
        # A positive burn-in must advance the represented absolute time
        if start <= t0:
            raise ValueError("burn_in is too small to resolve at t0")
        state = solve(state, t0, start, [start])[:, -1]
    states = solve(state, start, end, times)
    return Trajectory(times, states)
