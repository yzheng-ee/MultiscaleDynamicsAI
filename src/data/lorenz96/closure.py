"""GP closure fitting and separate training-pair sampling utilities."""

from dataclasses import dataclass

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Kernel

from .model import Array


@dataclass(frozen=True)
class GPClosure:
    """Adapt a fitted single-feature, scalar-output GP to a local closure.

    Each input value is predicted independently; the input shape is preserved.
    """

    regressor: GaussianProcessRegressor

    def __call__(self, slow: Array) -> Array:
        values = np.asarray(slow, dtype=float)
        return self.regressor.predict(values.reshape(-1, 1)).reshape(values.shape)


def sample_closure_training_data(
    slow_states: Array,  # (K, N)
    fast_states: Array,  # (K, J, N)
    *,
    sample_size: int = 800,
    seed: int | None = 42,
) -> tuple[Array, Array]:
    """Sample local (slow value, associated fast mean) training pairs.

    slow_states has shape (K, N) and fast_states has shape (K, J, N).
    fast_states[k, :, n] belongs to slow_states[k, n]. Every stored site/time
    pair has equal weight; callers must supply uniformly spaced samples if
    equal weighting in physical time is desired. No timestamps are inspected.
    Burn-in must be removed by the caller or simulation. Selects up to
    sample_size pairs without replacement; seed controls only selection.
    The default seed is 42; pass None for nondeterministic sampling.
    Returns features (n, 1) and targets (n,), where n is the smaller of
    sample_size and the available number of site/time pairs. Targets are
    instantaneous fast-group means, without multiplication by hx.
    """
    if isinstance(sample_size, bool) or not isinstance(sample_size, (int, np.integer)) or sample_size < 1:
        raise ValueError("sample_size must be a positive integer")
    slow = np.asarray(slow_states, dtype=float)
    fast = np.asarray(fast_states, dtype=float)
    if slow.ndim != 2 or slow.size == 0:
        raise ValueError("slow_states must have nonempty shape (K, N)")
    if (
        fast.ndim != 3
        or fast.shape[0] != slow.shape[0]
        or fast.shape[1] == 0
        or fast.shape[2] != slow.shape[1]
    ):
        raise ValueError("fast_states must have shape (K, J, N) matching slow_states, with J > 0")
    if not np.isfinite(slow).all() or not np.isfinite(fast).all():
        raise ValueError("training states must be finite")

    rng = np.random.default_rng(seed)
    indices = rng.choice(
        slow.size,
        size=min(sample_size, slow.size),
        replace=False,
    )
    features = slow.reshape(-1)[indices, None]
    targets = fast.mean(axis=1).reshape(-1)[indices]
    return features, targets


def fit_closure(
    features: Array,  # (n_samples, 1)
    targets: Array,  # (n_samples,)
    *,
    seed: int | None = None,
    kernel: Kernel | None = None,
    alpha: float = 1.0,
    n_restarts_optimizer: int = 2,
) -> GPClosure:
    """Fit a callable GP closure using all supplied samples, in order.

    features has shape (n_samples, 1) and targets (n_samples,).
    No sampling or simulation occurs. seed controls only optimizer randomness.
    Returns a GPClosure that preserves the shape of its input values.
    The fitted scikit-learn regressor is available as closure.regressor.

    kernel=None selects a constant-times-RBF kernel with initial length scale
    3 and length-scale bounds (1e-3, 1e3). A supplied scikit-learn kernel
    specifies its own initial parameters and optimization bounds.
    alpha is a scalar, nonnegative diagonal regularization/noise variance.
    n_restarts_optimizer adds optimization runs beyond the initial fit.
    """
    features = np.asarray(features, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if features.ndim != 2 or features.size == 0 or features.shape[1] != 1:
        raise ValueError("features must have nonempty shape (n_samples, 1)")
    if targets.shape != (features.shape[0],):
        raise ValueError("targets must have shape (n,) matching features")
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError("training pairs must be finite")
    if np.ndim(alpha) != 0 or not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be a finite, nonnegative scalar")

    if kernel is None:
        kernel = 1.0 * RBF(
            length_scale=3.0,
            length_scale_bounds=(1e-3, 1e3),
        )
    regressor = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=seed,
    )
    regressor.fit(features, targets)
    return GPClosure(regressor)
