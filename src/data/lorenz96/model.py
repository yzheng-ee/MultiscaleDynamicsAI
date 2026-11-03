"""Single-scale and multiscale Lorenz–96 models and initial-state sampling."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .._utils.validation import _validate_finite, _validate_integer, _validate_positive, _validate_real

Array = NDArray[np.float64]
Closure = Callable[[Array], Array]


def _slow_rhs(slow: Array, forcing: float) -> Array:
    """Compute slow-variable advection, damping, and constant forcing."""
    # f_k(x) = -x_{k-1} * (x_{k-2} - x_{k+1}) - x_k + F, with indices modulo K
    return -np.roll(slow, 1) * (np.roll(slow, 2) - np.roll(slow, -1)) - slow + forcing


@dataclass(frozen=True)
class MultiscaleLorenz96:
    """Coupled dynamics of K slow variables and J fast variables per slow site.

    hx scales fast-to-slow feedback, hy scales slow-to-fast forcing, and F
    is the constant slow-variable forcing. Positive eps controls the fast
    timescale relative to the slow dynamics.

    A full state has shape (K + K*J,): K slow values followed by fast values
    grouped by slow site. Fast values form one periodic ring across all groups.
    """

    K: int = 9
    J: int = 8
    hx: float = -0.8
    hy: float = 1.0
    F: float = 10.0
    eps: float = 2**-7

    def __post_init__(self):
        _validate_integer([("K", self.K), ("J", self.J)])
        # The slow advection stencil requires four distinct periodic sites
        if self.K < 4:
            raise ValueError("K must be >= 4")
        if self.J < 3:
            raise ValueError("J must be >= 3")
        _validate_finite([("hx", self.hx), ("hy", self.hy), ("F", self.F), ("eps", self.eps)])
        _validate_positive([("eps", self.eps)])

    @property
    def state_dimension(self) -> int:
        """Total number of slow and fast variables in a full state."""
        return self.K * (1 + self.J)

    def split_state(self, state: Array) -> tuple[Array, Array]:
        """Separate a full state or trajectory into slow and fast arrays.

        Input shape is (K + K*J,) or (K + K*J, N), where N is the sample count.
        Returns shapes (K,) and (K, J) for a state, or (K, N) and (K, J, N)
        for a trajectory. Fast-array axes are slow site, fast index, and time.
        """
        _validate_real([("state", state)])
        state = np.asarray(state, dtype=float)
        # Check the full channel count before reshaping fast values into site groups
        if state.ndim not in (1, 2) or state.shape[0] != self.state_dimension:
            raise ValueError(f"state must have shape ({self.state_dimension},) or ({self.state_dimension}, N)")

        return state[: self.K], state[self.K :].reshape(self.K, self.J, *state.shape[1:])

    def rhs(self, t: float, state: Array) -> Array:
        """Return time derivatives for a full state of shape (K + K*J,).

        The output preserves the input's slow-first, grouped-fast layout.
        t is accepted for ODE solver compatibility; the dynamics are autonomous.
        """
        slow, fast = state[: self.K], state[self.K :]
        # dx_k/dt = f_k(x) + (hx / J) * sum_{j=0}^{J-1} y_{k,j}
        ds = _slow_rhs(slow, self.F) + self.hx * fast.reshape(self.K, self.J).mean(axis=1)
        # dy_q/dt = [-y_{q+1} * (y_{q+2} - y_{q-1}) - y_q + hy * x_{floor(q/J)}] / eps, with fast indices modulo K*J
        df = (-np.roll(fast, -1) * (np.roll(fast, -2) - np.roll(fast, 1)) - fast + self.hy * np.repeat(slow, self.J)) / self.eps
        return np.concatenate((ds, df))


@dataclass(frozen=True)
class SingleScaleLorenz96:
    """Reduced Lorenz–96 dynamics for K slow variables using a fixed closure.

    closure maps slow values of shape (K,) to predicted fast-group means of
    the same shape. hx scales this feedback, and F is the constant forcing.
    Parameters of the eliminated fast dynamics belong to the supplied closure.
    """

    closure: Closure
    K: int = 9
    hx: float = -0.8
    F: float = 10.0

    def __post_init__(self):
        _validate_integer([("K", self.K)])
        # The reduced model uses the same four-site slow advection stencil
        if self.K < 4:
            raise ValueError("K must be >= 4")
        _validate_finite([("hx", self.hx), ("F", self.F)])
        if not callable(self.closure):
            raise TypeError("closure must be callable")

    @property
    def state_dimension(self) -> int:
        """Number of slow variables in a reduced state."""
        return self.K

    def rhs(self, t: float, state: Array) -> Array:
        """Return time derivatives for a slow state of shape (K,).

        Closure predictions are fast-group means, multiplied by hx here.
        t is accepted for ODE solver compatibility; the dynamics are autonomous.
        """
        means = self.closure(state)
        _validate_real([("closure output", means)])
        means = np.asarray(means, dtype=float)
        # Require one prediction per slow variable to prevent unintended broadcasting
        if means.shape != state.shape:
            raise ValueError("closure must return values with the same shape as slow")
        _validate_finite([("closure output", means)])

        # dx_k/dt = f_k(x) + hx * m_hat_k
        return _slow_rhs(state, self.F) + self.hx * means


def sample_initial_state(
    model: MultiscaleLorenz96 | SingleScaleLorenz96,
    *,
    low: float = -5.0,
    high: float = 10.0,
    seed: int | None = 2003,
) -> Array:
    """Sample an initial state using a NumPy random seed.

    Slow entries are uniform on [low, high). For a multiscale model, each fast
    group starts at its associated slow value. Returns shape (K + K*J,) for
    a multiscale model or (K,) for a single-scale model.

    Matching K, bounds, and an explicit seed gives identical slow entries
    for both model types. The default seed is 2003; seed=None uses nondeterministic initialization.
    """
    if not isinstance(model, (MultiscaleLorenz96, SingleScaleLorenz96)):
        raise TypeError("model must be a MultiscaleLorenz96 or SingleScaleLorenz96")
    _validate_finite([("low", low), ("high", high)])
    if low >= high:
        raise ValueError("low must be less than high")

    slow = np.random.default_rng(seed).uniform(low, high, model.K)
    if isinstance(model, MultiscaleLorenz96):
        return np.concatenate((slow, np.repeat(slow, model.J)))
    return slow
