"""Lorenz '96 multiscale model and its single-scale (closure) approximation.

*****
L96 equations + Example B.1 params (EnsembleKalmanMethods)
Vectorised rewrite, test-verified to 1e-12.
*****

Follows the Fatkullin-Vanden-Eijnden form used in Appendix B of Calvello, Reich & Stuart, Ensemble Kalman methods: 
a mean-field perspective (Acta Numerica 2025), and implemented by D. Burov for Burov, Giannakis, Manohar & Stuart (2021). 

a vectorised, allocation-light rewrite of that reference code; :mod:`tests.test_l96` asserts bit-comparable agreement with it.

Slow variables x_k, :math:`k = 0 \\dots K-1` (periodic):
    dx_k/dt = -x_{k-1} (x_{k-2} - x_{k+1}) - x_k + F + h_x * ybar_k

Fast variables y_{k,j}, :math:`j = 0 \\dots J-1`, coupled to slow variable k, with the (k, j) lattice wrapping as a single ring of length K*J
(i.e. y_{k,J} == y_{k+1,0}):
    dy_{k,j}/dt = ( -y_{k,j+1} (y_{k,j+2} - y_{k,j-1}) - y_{k,j} + h_y x_k ) / eps 
    and ybar_k = mean_j y_{k,j}.

The key structural observation behind the vectorisation is that the fast subsystem is a plain cyclic L96 ring in the flattened index n = k*J + j,
so the whole right-hand side reduces to a handful of cyclic index gathers.

Single-scale reduction: 
In the scale-separated limit eps -> 0 the averaging principle replaces ybar_k by a closure m(x_k) (see :mod:`msdyn.closures`), giving:
    dx_k/dt = -x_{k-1} (x_{k-2} - x_{k+1}) - x_k + F + h_x * m(x_k)
"""

from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

__all__ = [
    "L96Params",
    "L96Multiscale",
    "L96SingleScale",
    "slow_tendency",
    "average_fast",
    "split_state",
    "join_state",
    "slow_block",
    "fast_block",
]


@dataclass(frozen=True)
class L96Params:
    """Parameters of the Lorenz '96 multiscale system.
    Defaults reproduce the chaotic regime of Example B.1 in Calvello et al. (2025).

    K: Number of slow variables.
    J: Number of fast variables attached to each slow variable.
    h_x: Coupling of the fast-variable average into the slow equation (h_v in the Acta Numerica notation).
    h_y: Coupling of the slow variables into the fast equations (h_w).
    F: Constant forcing on the slow variables.
    eps: Scale-separation parameter; the fast variables evolve 1/eps faster.
    """

    K: int = 9
    J: int = 8
    h_x: float = -0.8
    h_y: float = 1.0
    F: float = 10.0
    eps: float = 2.0**-7

    @property
    def n_slow(self) -> int:
        return self.K

    @property
    def n_fast(self) -> int:
        return self.K * self.J

    @property
    def dim(self) -> int:
        """Dimension of the full multiscale state vector."""
        return self.K + self.K * self.J

    def with_(self, **changes: float | int) -> L96Params:
        """Return a copy with fields replaced (parameters are immutable)."""
        return replace(self, **changes)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "K": self.K,
            "J": self.J,
            "h_x": self.h_x,
            "h_y": self.h_y,
            "F": self.F,
            "eps": self.eps,
        }


####### State layout helpers                                                          
def split_state(z: np.ndarray, params: L96Params) -> tuple[np.ndarray, np.ndarray]:
    """Split a full state (dim,) into (slow (K,), fast (K*J,)) views."""
    return z[: params.K], z[params.K :]


def join_state(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Concatenate slow and fast blocks into a full state vector."""
    return np.concatenate([x, y])


def average_fast(y_flat: np.ndarray, params: L96Params) -> np.ndarray:
    """ybar_k = mean_j y_{k,j} for one state or a batch of states.
    Accepts (K*J,) or (..., K*J) and returns (K,) / (..., K).
    """
    shape = y_flat.shape[:-1] + (params.K, params.J)
    return y_flat.reshape(shape).mean(axis=-1)


def slow_block(traj: np.ndarray, params: L96Params) -> np.ndarray:
    """Extract the slow variables from a time-first trajectory (T, dim)."""
    return traj[..., : params.K]


def fast_block(traj: np.ndarray, params: L96Params) -> np.ndarray:
    """Extract the fast variables from a time-first trajectory (T, dim)."""
    return traj[..., params.K :]


# ### Right-hand sides                                                              

@lru_cache(maxsize=64)
def _ring_indices(n: int, offset: int) -> np.ndarray:
    """Gather indices realising shifted[i] = a[i + offset] on a cyclic ring.
    Cached and marked read-only: these are rebuilt on every RHS evaluation of the single-scale model, which runs for millions of steps.
    """
    indices = (np.arange(n) + offset) % n
    indices.flags.writeable = False
    return indices


def slow_tendency(x: np.ndarray, F: float) -> np.ndarray:
    """Uncoupled slow-variable tendency -x_{k-1}(x_{k-2} - x_{k+1}) - x_k + F.

    Shared by the multiscale and single-scale models so the two can never drift apart. 
    Batch-aware: the ring index is the *last* axis, so x may be (K,) or (..., K).
    """
    return -_shift(x, -1) * (_shift(x, -2) - _shift(x, 1)) - x + F


def _shift(a: np.ndarray, offset: int) -> np.ndarray:
    """out[..., i] = a[..., (i + offset) % n] -- a cyclic gather on the last axis."""
    return a[..., _ring_indices(a.shape[-1], offset)]


class L96Multiscale:
    """Full two-scale Lorenz '96 system, callable as f(t, z) for integrators.

    Batch-aware: z may be a single state (dim,) or an ensemble (n_ensemble, dim). 
    Batching is the main lever for throughput, the cost is dominated by NumPy call overhead on these small rings, 
    so 64 trajectories integrate in roughly twice the wall-clock of one.

    Examples: 
    >>> params = L96Params()
    >>> system = L96Multiscale(params)
    >>> z0 = system.default_initial_state(seed=0)
    >>> system(0.0, z0).shape == (params.dim,)
    True
    >>> system(0.0, np.stack([z0, z0])).shape == (2, params.dim)
    True
    """

    def __init__(self, params: L96Params | None = None) -> None:
        self.params = params if params is not None else L96Params()
        p = self.params
        # Precomputed cyclic gathers: measurably faster than np.roll, which
        # allocates and concatenates on every call.
        self._ix_km1 = _ring_indices(p.K, -1)
        self._ix_km2 = _ring_indices(p.K, -2)
        self._ix_kp1 = _ring_indices(p.K, 1)
        n_fast = p.n_fast
        self._iy_np1 = _ring_indices(n_fast, 1)
        self._iy_np2 = _ring_indices(n_fast, 2)
        self._iy_nm1 = _ring_indices(n_fast, -1)
        self._i_parent = np.repeat(np.arange(p.K), p.J)

    def __call__(self, t: float, z: np.ndarray) -> np.ndarray:  # noqa: ARG002 - ODE signature
        p = self.params
        x = z[..., : p.K]
        y = z[..., p.K :]

        ybar = y.reshape(*y.shape[:-1], p.K, p.J).mean(axis=-1)
        dx = -x[..., self._ix_km1] * (x[..., self._ix_km2] - x[..., self._ix_kp1]) - x + p.F
        dx = dx + p.h_x * ybar

        # The fast lattice is a single cyclic ring of length K*J in the flat
        # index n = k*J + j, running in the opposite direction to the slow ring.
        dy = -y[..., self._iy_np1] * (y[..., self._iy_np2] - y[..., self._iy_nm1]) - y
        dy = dy + p.h_y * x[..., self._i_parent]
        dy = dy / p.eps

        return np.concatenate([dx, dy], axis=-1)

    def default_initial_state(self, seed: int | None = None, n_ensemble: int | None = None) -> np.ndarray:
        """Initial condition matching the reference data_gen.py.

        Slow variables uniform on [-5, 10); each fast variable initialised to its parent slow variable. 
        This is far off the attractor, so always burn in before using the trajectory (:mod:`msdyn.data.generate`).

        Returns (dim,) when n_ensemble is None, else (n_ensemble, dim).
        """
        rng = np.random.default_rng(seed)
        p = self.params
        shape = (p.K,) if n_ensemble is None else (n_ensemble, p.K)
        x0 = rng.random(shape) * 15.0 - 5.0
        y0 = np.repeat(x0, p.J, axis=-1)
        return np.concatenate([x0, y0], axis=-1)


class L96SingleScale:
    """Single-scale Lorenz '96 closure model, callable as f(t, x).

    params: Shares K, F and h_x with the multiscale system; J, h_y and eps are unused here but retained so that a single
        :class:`L96Params` describes the matched pair of models.
    closure: Map m from the slow state (K,) to the predicted fast average (K,). 
        Under the averaging principle the exact object is M(x) = int ybar mu^x(dy) the standard further approximation M_k(x) ~ m(x_k) makes it a scalar function applied componentwise.
    :mod:`msdyn.closures`
    """

    def __init__(self, params: L96Params, closure: Callable[[np.ndarray], np.ndarray]) -> None:
        self.params = params
        self.closure = closure

    def __call__(self, t: float, x: np.ndarray) -> np.ndarray:  # noqa: ARG002 - ODE signature
        p = self.params
        m = np.asarray(self.closure(x))
        if m.shape != x.shape:
            raise ValueError(f"closure returned shape {m.shape}, expected {x.shape}")
        return slow_tendency(x, p.F) + p.h_x * m

    @property
    def dim(self) -> int:
        return self.params.K
