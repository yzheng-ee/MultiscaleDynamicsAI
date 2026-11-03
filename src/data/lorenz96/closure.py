"""Local GP closure fitting from slow and fast states."""

from dataclasses import dataclass

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Kernel

from .._utils.validation import _validate_finite, _validate_integer, _validate_nonnegative, _validate_positive, _validate_real
from .model import Array


@dataclass(frozen=True)
class GPClosure:
    """Predict the associated fast-group mean from each slow value.

    Each input value is predicted independently; the input shape is preserved.
    """

    regressor: GaussianProcessRegressor

    def __call__(self, slow: Array) -> Array:
        _validate_real([("slow", slow)])
        values = np.asarray(slow, dtype=float)
        return self.regressor.predict(values.reshape(-1, 1)).reshape(values.shape)


def _sample_closure_training_data(
    slow_states: Array,  # (K, N)
    fast_states: Array,  # (K, J, N)
    *,
    sample_size: int = 800,
    seed: int | np.random.Generator | None = 2003,
) -> tuple[Array, Array]:
    """Sample local (slow value, associated fast mean) training pairs.

    slow_states has shape (K, N) and fast_states has shape (K, J, N).
    fast_states[k, :, n] belongs to slow_states[k, n]. Every stored site/time
    pair has equal selection probability. Selects up to sample_size pairs
    without replacement; seed accepts an integer, an existing generator,
    or None for nondeterministic sampling.
    Returns features (n, 1) and targets (n,), where n is the smaller of
    sample_size and the available number of site/time pairs. Targets are
    instantaneous fast-group means, without multiplication by hx.
    """
    _validate_integer([("sample_size", sample_size)])
    _validate_positive([("sample_size", sample_size)])

    # Reject complex states before float conversion can discard imaginary components
    _validate_real([("slow_states", slow_states), ("fast_states", fast_states)])
    slow = np.asarray(slow_states, dtype=float)
    fast = np.asarray(fast_states, dtype=float)

    # Match slow sites and timestamps so each target averages its associated fast group
    if slow.ndim != 2 or slow.size == 0:
        raise ValueError("slow_states must have nonempty shape (K, N)")
    if (
        fast.ndim != 3
        or fast.shape[0] != slow.shape[0]
        or fast.shape[1] == 0
        or fast.shape[2] != slow.shape[1]
    ):
        raise ValueError("fast_states must have shape (K, J, N) matching slow_states, with J > 0")
    _validate_finite([("slow_states", slow), ("fast_states", fast)])

    rng = np.random.default_rng(seed)
    indices = rng.choice(slow.size, size=min(sample_size, slow.size), replace=False)

    # Flatten both arrays in the same site/time order to preserve pair alignment
    features = slow.reshape(-1)[indices, None]
    targets = fast.mean(axis=1).reshape(-1)[indices]
    return features, targets


def fit_closure(
    slow_states: Array,  # (K, N)
    fast_states: Array,  # (K, J, N)
    *,
    # Training-pair sampling and reproducibility
    sample_size: int = 800,
    seed: int | None = 2003,

    # Gaussian-process fitting
    kernel: Kernel | None = None,
    alpha: float = 1.0,
    n_restarts_optimizer: int = 2,
) -> GPClosure:
    """Sample local training pairs and fit a callable GP closure.

    slow_states has shape (K, N) and fast_states has shape (K, J, N).
    Selects up to sample_size site/time pairs uniformly without replacement.
    Supply uniformly spaced samples for equal weighting in physical time,
    and remove burn-in before calling. Targets are instantaneous fast-group
    means, without multiplication by hx.

    seed controls independent random streams for pair selection and GP
    optimization. The default is 2003; pass None for nondeterministic results.

    Returns a GPClosure that preserves the shape of its input values.
    The fitted scikit-learn regressor is available as closure.regressor;
    its X_train_ and y_train_ attributes contain the sampled training pairs.

    kernel=None selects a constant-times-RBF kernel with initial length scale
    3 and length-scale bounds (1e-3, 1e3). A supplied scikit-learn kernel
    specifies its own initial parameters and optimization bounds.
    alpha is a scalar, nonnegative diagonal regularization/noise variance.
    n_restarts_optimizer adds optimization runs beyond the initial fit.
    """
    # Separate streams keep optimizer randomness independent of pair selection
    sampling_sequence, fit_sequence = np.random.SeedSequence(seed).spawn(2)
    sampling_rng = np.random.default_rng(sampling_sequence)
    fit_seed = int(fit_sequence.generate_state(1, dtype=np.uint32)[0])  # scikit-learn accepts a 32-bit integer seed

    features, targets = _sample_closure_training_data(slow_states, fast_states, sample_size=sample_size, seed=sampling_rng)

    # Averaging finite fast states can still overflow when constructing targets
    _validate_finite([("features", features), ("targets", targets)])

    # Use one shared regularization value rather than a separate value per training pair
    if np.ndim(alpha) != 0:
        raise ValueError("alpha must be a scalar")
    _validate_finite([("alpha", alpha)])
    _validate_nonnegative([("alpha", alpha)])

    if kernel is None:
        kernel = 1.0 * RBF(length_scale=3.0, length_scale_bounds=(1e-3, 1e3))

    regressor = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=fit_seed,
    )
    regressor.fit(features, targets)

    return GPClosure(regressor)
