"""Generate single-scale and multiscale Lorenz–96 trajectories from explicit parameters."""

from .model import Array, Closure, MultiscaleLorenz96, SingleScaleLorenz96, sample_initial_state
from .simulate import Trajectory, simulate_ode


def generate_multiscale_trajectory(
    *,
    # Model parameters
    K: int = 9,
    J: int = 8,
    hx: float = -0.8,
    hy: float = 1.0,
    F: float = 10.0,
    eps: float = 2**-7,

    # Initial-state sampling
    low: float = -5.0,
    high: float = 10.0,
    seed: int | None = 2003,

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
    """Construct a model, sample its initial state, and integrate it.

    Slow initial values are uniform on [low, high); each fast group starts
    at its associated slow value. seed controls only initialization. For an
    existing initial state or continuation, call simulate_ode directly.

    The returned state has shape (K + K*J, N), with slow entries first and
    fast entries flattened in C order. Times sample the interval
    [t0 + burn_in, t0 + burn_in + duration], always including both endpoints.
    Supply exactly one of duration or n_samples. duration must be an integer
    multiple of sample_dt; n_samples must be an integer >= 2 and implies
    duration = (n_samples - 1) * sample_dt. Solver settings pass through
    unchanged; max_step=None sets the step-size cap to sample_dt. An explicit
    value overrides it for both burn-in and the recorded simulation.

    Validation is delegated to the model, sampler, and integrator.
    """
    model = MultiscaleLorenz96(
        K=K,
        J=J,
        hx=hx,
        hy=hy,
        F=F,
        eps=eps,
    )
    initial_state = sample_initial_state(
        model,
        low=low,
        high=high,
        seed=seed,
    )
    return simulate_ode(
        model.rhs,
        initial_state,
        duration=duration,
        n_samples=n_samples,
        sample_dt=sample_dt,
        burn_in=burn_in,
        t0=t0,
        method=method,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )


def generate_singlescale_trajectory(
    *,
    # Model parameters and frozen closure
    closure: Closure,
    K: int = 9,
    hx: float = -0.8,
    F: float = 10.0,

    # Initial-state sampling
    low: float = -5.0,
    high: float = 10.0,
    seed: int | None = 2003,

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
    """Construct a reduced model, sample its initial state, and integrate it.

    closure is a fixed callable mapping slow values to associated fast means,
    before multiplication by hx.
    Parameters of the eliminated fast dynamics belong to the supplied closure.

    Initial values are uniform on [low, high). seed controls only initialization;
    matching K, an explicit seed, and sampling bounds gives the same initial slow values as
    generate_multiscale_trajectory, before either system's burn-in. For an
    existing initial state or continuation, call simulate_ode directly.

    The returned state has shape (K, N). Times sample the interval
    [t0 + burn_in, t0 + burn_in + duration], always including both endpoints.
    Supply exactly one of duration or n_samples. duration must be an integer
    multiple of sample_dt; n_samples must be an integer >= 2 and implies
    duration = (n_samples - 1) * sample_dt. Solver settings pass through
    unchanged; max_step=None sets the step-size cap to sample_dt. An explicit
    value overrides it for both burn-in and the recorded simulation.

    Validation is delegated to the model, sampler, and integrator.
    """
    model = SingleScaleLorenz96(closure=closure, K=K, hx=hx, F=F)
    initial_state = sample_initial_state(model, low=low, high=high, seed=seed)
    return simulate_ode(
        model.rhs,
        initial_state,
        duration=duration,
        n_samples=n_samples,
        sample_dt=sample_dt,
        burn_in=burn_in,
        t0=t0,
        method=method,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
