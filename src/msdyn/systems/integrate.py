"""Fixed-step and adaptive integrators for the Lorenz '96 test-bed.

Trajectories are time-first
This differs from the reference implementation of Burov / Calvello et al. (EnsembleKalmanMethods), which stores (state_dim, n_times). 
Time-first is what PANDA and every other sequence model expects, so we normalise here and transpose only at the boundary with the reference code.
"""

from __future__ import annotations
from collections.abc import Callable
import numpy as np
RHS = Callable[[float, np.ndarray], np.ndarray]

# Empirically calibrated step size for the Lorenz '96 fast ring, measured in the fast time variable tau = t / eps. 
# A naive linear-stability estimate (dt < 2.78 * eps) is far too optimistic: the fast subsystem's quadratic terms make |dy/dtau| ~ 60 on the attractor, 
# and a 4th-order scheme needs to resolve that, not the -y/eps term.
# At this value RK4 shows clean order-4 self-convergence with ~2e-6 error per 0.01 time units at eps = 2^-7. notebooks/01_l96_data_generation.ipynb

_DT_PER_EPS = 5e-3

def rk4_step(f: RHS, t: float, y: np.ndarray, dt: float) -> np.ndarray:
    """One classical Runge-Kutta 4 step.  y is untouched."""
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def suggested_dt(eps: float, safety: float = 1.0) -> float:
    """Calibrated RK4 step size for Lorenz '96 at scale separation eps.

    Returns 5e-3 * eps (~3.9e-5 at eps = 2^-7), which resolves the fast ring rather than merely keeping the linear term stable. 
    Lower safety if you raise h_y or F substantially, since both increase the amplitude, hence the speed of the fast variables.
    Always confirm with `self_convergence_error` before generating data: this is a calibration, not a guarantee.
    """
    return safety * _DT_PER_EPS * eps


def integrate_rk4(
    f: RHS,
    y0: np.ndarray,
    t_end: float,
    dt: float,
    *,
    t_start: float = 0.0,
    sample_every: int = 1,
    progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate y' = f(t, y) with fixed-step RK4, storing every sample_every step.

    f: Right-hand side, signature f(t, y) -> dy/dt. May be batch-aware.
    y0: Initial state, shape (state_dim,) or (n_ensemble, state_dim).
    t_end: Final time (integration runs over [t_start, t_end]).
    dt: Integration step. See `suggested_dt`.
    sample_every: Store the state every this many steps. The effective sampling interval of the returned trajectory is sample_every * dt.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if sample_every < 1:
        raise ValueError(f"sample_every must be >= 1, got {sample_every}")

    n_steps = int(round((t_end - t_start) / dt))
    if n_steps < 1:
        raise ValueError(f"t_end - t_start = {t_end - t_start} is shorter than dt = {dt}")

    n_samples = n_steps // sample_every + 1
    traj = np.empty((n_samples,) + y0.shape, dtype=np.float64)
    times = np.empty(n_samples, dtype=np.float64)

    y = np.asarray(y0, dtype=np.float64).copy()
    t = float(t_start)
    traj[0] = y
    times[0] = t

    store_idx = 1
    for step in range(1, n_steps + 1):
        y = rk4_step(f, t, y, dt)
        t = t_start + step * dt
        if not np.all(np.isfinite(y)):
            raise FloatingPointError(f"integration blew up at t={t:.4f} (step {step}); reduce dt (currently {dt})")
        if step % sample_every == 0 and store_idx < n_samples:
            traj[store_idx] = y
            times[store_idx] = t
            store_idx += 1
        if progress and step % max(1, n_steps // 20) == 0:
            print(f"  ... {100 * step / n_steps:5.1f}%  t={t:8.3f}", flush=True)

    return times[:store_idx], traj[:store_idx]


def integrate_scipy(
    f: RHS,
    y0: np.ndarray,
    t_end: float,
    *,
    t_start: float = 0.0,
    t_eval: np.ndarray | None = None,
    method: str = "RK45",
    max_step: float = np.inf,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Adaptive reference integration via scipy.integrate.solve_ivp.
    Slower than `integrate_rk4` but error-controlled; used to validate the fixed-step path and to reproduce the reference EnsembleKalmanMethods runs.
    """
    from scipy.integrate import solve_ivp

    sol = solve_ivp(f,(t_start, t_end),np.asarray(y0, dtype=np.float64),method=method,t_eval=t_eval,max_step=max_step,rtol=rtol,atol=atol,)
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    return sol.t, sol.y.T


def self_convergence_error(f: RHS, y0: np.ndarray, t_end: float, dt: float) -> float:
    """Richardson self-convergence: max |RK4(dt) - RK4(dt/2)| at t_end.

    This is the right well-posed check for a stiff chaotic system. Comparing against an adaptive solver over a long window is meaningless here: 
    the Lorenz '96 fast subsystem has a Lyapunov exponent of order 1/eps (roughly 250 at eps = 2^-7), so two correct-but-not-identical solutions
    separate exponentially within a fraction of a time unit. Keep t_end below about 0.02 at eps = 2^-7.

    For a 4th-order method the error should fall by ~16x when dt is halved.
    """
    _, coarse = integrate_rk4(f, y0, t_end, dt)
    _, fine = integrate_rk4(f, y0, t_end, dt / 2)
    return float(np.max(np.abs(coarse[-1] - fine[-1])))


def check_integrator_agreement(f: RHS, y0: np.ndarray, t_end: float, dt: float) -> float:
    """Max abs deviation between fixed-step RK4 and an error-controlled solve.

    Same caveat as :func:`self_convergence_error`: only meaningful on windows short compared with the *fastest* Lyapunov time.
    """
    _, traj_fixed = integrate_rk4(f, y0, t_end, dt)
    _, traj_ref = integrate_scipy(f, y0, t_end, t_eval=np.array([t_end]), max_step=dt)
    return float(np.max(np.abs(traj_fixed[-1] - traj_ref[-1])))
