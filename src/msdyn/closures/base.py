"""Interface and training data for Lorenz '96 closures.

The averaging principle replaces the fast-variable average ybar_k in the slow equation by M_k(x) = int ybar_k mu^x(dy). 
Following Appendix B of Calvello, Reich & Stuart (2025) we use the further approximation M_k(x) ~ m(x_k), a single *scalar* function applied componentwise, 
which is empirically accurate for large J. Every closure here therefore fits m: R -> R from (x_k, ybar_k) pairs pooled over k and over time, exploiting the ring symmetry of the system.

Closures deliberately expose a plain __call__(x) -> m(x) so they can be dropped straight into :class:`msdyn.systems.l96.L96SingleScale`.
"""

from __future__ import annotations
from typing import NamedTuple, Protocol, runtime_checkable
import numpy as np
from msdyn.systems.l96 import L96Params, average_fast

@runtime_checkable
class Closure(Protocol):
    """A fitted map from the slow state to the predicted fast-variable average."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Map slow state (..., K) to predicted ybar of the same shape."""
        ...

    def fit(self, pairs: ClosurePairs) -> Closure:
        """Fit on pooled (x_k, ybar_k) pairs and return self."""
        ...


class ClosurePairs(NamedTuple):
    """Pooled training pairs for a componentwise closure.
    x: Slow-variable values, shape (n_pairs,).
    ybar: Corresponding fast-variable averages, shape (n_pairs,).
    """

    x: np.ndarray
    ybar: np.ndarray

    def __len__(self) -> int:
        return self.x.size

    def subsample(self, n: int, seed: int | None = None) -> ClosurePairs:
        """Draw n pairs without replacement (no-op if n >= len(self)).

        Gaussian process regression is O(n^3), so the reference implementation fits on a few hundred points; this makes that explicit.
        """
        if n >= len(self):
            return self
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self), size=n, replace=False)
        return ClosurePairs(self.x[idx], self.ybar[idx])


def gather_closure_pairs(traj: np.ndarray, params: L96Params) -> ClosurePairs:
    """Pool (x_k, ybar_k) pairs from a multiscale trajectory.
    traj: Time-first full multiscale trajectory, shape (T, dim).
    params: System parameters (supplies K and J).

    Returns: ClosurePairs: T * K pairs, flattened over both time and ring position.
    """
    if traj.ndim != 2 or traj.shape[-1] != params.dim:
        raise ValueError(f"expected trajectory of shape (T, {params.dim}), got {traj.shape}")
    x = traj[:, : params.K]
    ybar = average_fast(traj[:, params.K :], params)
    return ClosurePairs(x.reshape(-1), ybar.reshape(-1))
