"""Gaussian process closure - the method used in the reference experiments.

*****
GP hyperparameters (RBF, length_scale=3, alpha=1, 800 pts)
*****

Calvello, Reich & Stuart (2025, Appendix B.2) fit m by Gaussian process regression, 
as does the accompanying EnsembleKalmanMethods code (an RBF kernel with alpha=1, fitted on a few hundred subsampled pairs). 
We keep those defaults so results are comparable, but expose them as arguments.

Unlike :class:`~msdyn.closures.polynomial.PolynomialClosure`, a GP also returns a predictive standard deviation, which is the natural handle for a stochastic
closure and for quantifying where the single-scale reduction is least trustworthy.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
from msdyn.closures.base import ClosurePairs

_DEFAULT_MAX_PAIRS = 800  # GP regression is O(n^3); the reference uses 500-800.


class GaussianProcessClosure:
    """GP regression closure m(x) with an RBF kernel.

    max_pairs: Subsample size used for fitting (GP cost is cubic in this).
    length_scale, alpha, n_restarts:
        Passed to scikit-learn. alpha=1 matches the reference code and acts as a strong nugget: the (x_k, ybar_k) scatter is genuinely thick, so the GP is being asked for a conditional mean, not an interpolant.
    seed: Controls the subsample.
    """

    def __init__(
        self,
        max_pairs: int = _DEFAULT_MAX_PAIRS,
        length_scale: float = 3.0,
        alpha: float = 1.0,
        n_restarts: int = 15,
        seed: int | None = 0,
    ) -> None:
        self.max_pairs = max_pairs
        self.length_scale = length_scale
        self.alpha = alpha
        self.n_restarts = n_restarts
        self.seed = seed
        self.regressor = None

    def fit(self, pairs: ClosurePairs) -> GaussianProcessClosure:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF

        sample = pairs.subsample(self.max_pairs, seed=self.seed)
        kernel = 1.0 * RBF(self.length_scale, (1e-10, 1e6))
        self.regressor = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=self.n_restarts,
            alpha=self.alpha,
            random_state=self.seed,
        )
        self.regressor.fit(sample.x[:, None], sample.ybar)
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.regressor is None:
            raise RuntimeError("closure is not fitted; call .fit(pairs) first")
        flat = np.asarray(x).reshape(-1, 1)
        return self.regressor.predict(flat).reshape(np.shape(x))

    def predict_with_std(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and standard deviation of m at x."""
        if self.regressor is None:
            raise RuntimeError("closure is not fitted; call .fit(pairs) first")
        flat = np.asarray(x).reshape(-1, 1)
        mean, std = self.regressor.predict(flat, return_std=True)
        return mean.reshape(np.shape(x)), std.reshape(np.shape(x))

    def save(self, path: str | Path) -> None:
        from joblib import dump
        dump(self.regressor, path)

    @classmethod
    def load(cls, path: str | Path, **kwargs: object) -> GaussianProcessClosure:
        """Load a regressor saved by :meth:`save` (or the reference closure.joblib)."""
        from joblib import load
        closure = cls(**kwargs)  # type: ignore[arg-type]
        closure.regressor = load(path)
        return closure

    def __repr__(self) -> str:
        state = "unfitted" if self.regressor is None else f"kernel={self.regressor.kernel_}"
        return f"GaussianProcessClosure(max_pairs={self.max_pairs}, {state})"
