"""Generate multiscale Lorenz '96 trajectories from explicit parameters."""

from .model import Array, MultiscaleLorenz96, sample_initial_state
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
    seed: int | None = 42,

    # Simulation interval and sampling
    duration: float,
    sample_dt: float, # 0.001
    burn_in: float = 0.0,
    t0: float = 0.0,
    endpoint: bool = True,

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
    [t0 + burn_in, t0 + burn_in + duration], including both endpoints by
    default. endpoint=False excludes the final endpoint. The solver still integrates
    the full duration. duration must be an integer
    multiple of sample_dt. Solver settings are passed through unchanged;
    max_step=None sets the step-size cap to sample_dt; an explicit value
    overrides it for both burn-in and the recorded simulation.

    Validation is delegated to the model, sampler, and integrator. This
    function returns only the trajectory and performs no fitting or file I/O.
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
        sample_dt=sample_dt,
        burn_in=burn_in,
        t0=t0,
        endpoint=endpoint,
        method=method,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
