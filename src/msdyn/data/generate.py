"""Generate Lorenz '96 trajectory ensembles.

*****
Initial-condition convention only.
*****

Ensembles, not one long run. Forecast evaluation needs many independent context windows. 
Because the RHS is tiny (81 numbers), NumPy call overhead dominates and integrating 64 trajectories costs barely more wall-clock
than one so everything here is batched over an ensemble axis.

Burn-in is mandatory. The reference initial condition (fast variables set equal to their parent slow variable) 
is far off the attractor and produces a violent transient; statistics gathered from it are meaningless.

The integrator step must resolve the fast ring (~4e-5 at eps = 2^-7), but nothing downstream needs samples that dense.
sample_every decouples the two.
"""

from __future__ import annotations
from collections.abc import Callable
import numpy as np
from msdyn.data.io import TrajectoryDataset
from msdyn.systems.integrate import integrate_rk4, suggested_dt
from msdyn.systems.l96 import L96Multiscale, L96Params, L96SingleScale

DEFAULT_BURN_IN = 20.0  # time units; ~30 slow Lyapunov times, ample to reach the attractor

def burn_in(rhs: Callable[[float, np.ndarray], np.ndarray],z0: np.ndarray,t_burn: float,dt: float,*,progress: bool = False,) -> np.ndarray:
    """Integrate forward and return only the final state(s), discarding the transient."""
    _, traj = integrate_rk4(rhs, z0, t_end=t_burn, dt=dt, sample_every=max(1, int(t_burn / dt)), progress=progress)
    return traj[-1]


def generate_multiscale(
    params: L96Params | None = None,
    *,
    n_trajectories: int = 16,
    t_total: float = 50.0,
    t_burn: float = DEFAULT_BURN_IN,
    dt: float | None = None,
    sample_dt: float = 0.01,
    seed: int = 0,
    progress: bool = False,
) -> TrajectoryDataset:
    """Integrate the full two-scale system from independent, burnt-in initial conditions.
    sample_dt: Interval at which states are stored. Rounded to a multiple of dt.
    dt: Integrator step; defaults to :func:`~msdyn.systems.integrate.suggested_dt`.

    Returns TrajectoryDataset: kind="multiscale", channels ordered [x_0..x_{K-1}, y_0..y_{KJ-1}]
    """
    params = params or L96Params()
    system = L96Multiscale(params)
    dt = dt if dt is not None else suggested_dt(params.eps)
    sample_every = max(1, int(round(sample_dt / dt)))

    z0 = system.default_initial_state(seed=seed, n_ensemble=n_trajectories)
    if progress:
        print(f"burn-in: t={t_burn} dt={dt:g} ({int(t_burn / dt)} steps)", flush=True)
    z_attractor = burn_in(system, z0, t_burn, dt, progress=progress)

    if progress:
        print(f"sampling: t={t_total} store every {sample_every} steps", flush=True)
    times, traj = integrate_rk4(system, z_attractor, t_end=t_total, dt=dt, sample_every=sample_every, progress=progress)

    return TrajectoryDataset(
        trajectories=np.transpose(traj, (1, 0, 2)),  # (T, n_traj, D) -> (n_traj, T, D)
        times=times,
        params=params,
        kind="multiscale",
        metadata={
            "integrator": "rk4",
            "dt": dt,
            "sample_every": sample_every,
            "t_burn": t_burn,
            "seed": seed,
        },
    )


def generate_singlescale(
    params: L96Params,
    closure: Callable[[np.ndarray], np.ndarray],
    *,
    n_trajectories: int = 16,
    t_total: float = 50.0,
    t_burn: float = DEFAULT_BURN_IN,
    dt: float = 1e-3,
    sample_dt: float = 0.01,
    seed: int = 0,
    closure_name: str = "unknown",
    progress: bool = False,
) -> TrajectoryDataset:
    """Integrate the single-scale closure model.

    The fast ring is gone, so this is a non-stiff K-dimensional system and
    dt = 1e-3 is ample, two orders of magnitude cheaper per unit time than
    the multiscale model.

    closure: Fitted m; see :mod:`msdyn.closures`.
    closure_name: Recorded in the dataset metadata so runs stay traceable.
    """
    system = L96SingleScale(params, closure)
    sample_every = max(1, int(round(sample_dt / dt)))

    rng = np.random.default_rng(seed)
    x0 = rng.random((n_trajectories, params.K)) * 15.0 - 5.0
    x_attractor = burn_in(system, x0, t_burn, dt, progress=progress)

    times, traj = integrate_rk4(system, x_attractor, t_end=t_total, dt=dt, sample_every=sample_every, progress=progress)

    return TrajectoryDataset(
        trajectories=np.transpose(traj, (1, 0, 2)),
        times=times,
        params=params,
        kind="singlescale",
        metadata={
            "integrator": "rk4",
            "dt": dt,
            "sample_every": sample_every,
            "t_burn": t_burn,
            "seed": seed,
            "closure": closure_name,
        },
    )
