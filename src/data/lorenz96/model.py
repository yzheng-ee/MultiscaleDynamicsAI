"""Appendix B Lorenz '96 equations, independent of numerical integration."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Closure = Callable[[Array], Array]


def _slow_rhs(slow: Array, forcing: float) -> Array:
    return -np.roll(slow, 1) * (np.roll(slow, 2) - np.roll(slow, -1)) - slow + forcing


def _validate_dimension(name: str, value: int, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class MultiscaleLorenz96:
    """Physical parameters (K, hx, hy correspond to L, h_v, h_w).

    Full states contain K slow entries followed by K*J fast entries in
    C order. The fast entries form one periodic ring across all groups.
    """

    K: int = 9
    J: int = 8
    hx: float = -0.8
    hy: float = 1.0
    F: float = 10.0
    eps: float = 2**-7

    def __post_init__(self):
        for name, minimum in (("K", 4), ("J", 3)):
            _validate_dimension(name, getattr(self, name), minimum)
        for name in ("hx", "hy", "F", "eps"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.eps <= 0:
            raise ValueError("eps must be positive")

    @property
    def state_dimension(self) -> int:
        return self.K * (1 + self.J)

    def split_state(self, state: Array) -> tuple[Array, Array]:
        """Unpack a state (D,) or trajectory (D, N) into slow and fast arrays.

        Returns shapes (K,), (K, J) for a vector, or (K, N), (K, J, N)
        for a trajectory. Fast entries use the same C order as rhs.
        """
        state = np.asarray(state, dtype=float)
        if state.ndim not in (1, 2) or state.shape[0] != self.state_dimension:
            raise ValueError(f"state must have shape ({self.state_dimension},) or ({self.state_dimension}, N)")
        return state[:self.K], state[self.K:].reshape(self.K, self.J, *state.shape[1:])

    def rhs(self, t: float, state: Array) -> Array:
        """Evaluate B.1–B.3 on a flat full state."""
        slow, fast = state[:self.K], state[self.K:]
        ds = _slow_rhs(slow, self.F) + self.hx * fast.reshape(self.K, self.J).mean(axis=1)
        df = (-np.roll(fast, -1) * (np.roll(fast, -2) - np.roll(fast, 1))
              - fast + self.hy * np.repeat(slow, self.J)) / self.eps
        return np.concatenate((ds, df))


@dataclass(frozen=True)
class SingleScaleLorenz96:
    """Reduced dynamics with a required, fixed callable fast-mean closure.

    The closure maps a K-vector of slow values to a K-vector of fast means.
    Parameters of the eliminated fast dynamics belong to the supplied closure.
    """

    closure: Closure
    K: int = 9
    hx: float = -0.8
    F: float = 10.0

    def __post_init__(self):
        _validate_dimension("K", self.K, 4)
        for name in ("hx", "F"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not callable(self.closure):
            raise TypeError("closure must be callable")

    @property
    def state_dimension(self) -> int:
        return self.K

    def rhs(self, t: float, state: Array) -> Array:
        """Evaluate (2.9); closure predicts fast means, before multiplying by hx."""
        means = np.asarray(self.closure(state), dtype=float)
        if means.shape != state.shape or not np.isfinite(means).all():
            raise ValueError("closure must return finite values with the same shape as slow")
        return _slow_rhs(state, self.F) + self.hx * means


def sample_initial_state(
    model: MultiscaleLorenz96 | SingleScaleLorenz96,
    *,
    low: float = -5.0,
    high: float = 10.0,
    seed: int | None = None,
) -> Array:
    """Sample a state without integrating or evaluating a closure.

    Slow entries are uniform on [low, high). For a multiscale model, each fast
    group starts at its associated slow value, packed in C order. With the
    same K and seed, both model types receive identical slow entries.
    """
    if not isinstance(model, (MultiscaleLorenz96, SingleScaleLorenz96)):
        raise TypeError("model must be a MultiscaleLorenz96 or SingleScaleLorenz96")
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        raise ValueError("low and high must be finite, with low < high")
    slow = np.random.default_rng(seed).uniform(low, high, model.K)
    if isinstance(model, MultiscaleLorenz96):
        return np.concatenate((slow, np.repeat(slow, model.J)))
    return slow
