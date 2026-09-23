"""Gaussian-process closure for the single-scale Lorenz-96 model."""

from pathlib import Path
from typing import Any
import warnings

from joblib import dump, load
import numpy as np
from numpy.typing import NDArray
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF


def fit_closure(
    pairs: NDArray[np.float64],
    sample_size: int = 800,
    kernel: str = "rbf",
    length_scale: float = 3.0,
    rbf_length_scale_bounds: tuple[float, float] = (1e-10, 1e6),
    matern_nu: float = 1.5,
    alpha: float = 1.0,
    optimizer_restarts: int = 15,
    output_path: str | Path = "closure.joblib",
    random_state: int | None = None,
) -> GaussianProcessRegressor:
    """Fit and save the closure mapping from slow variables to fast means."""
    if pairs.ndim != 2 or pairs.shape[1] < 2:
        raise ValueError("pairs must be a 2D array with input and target columns")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    n_pairs = pairs.shape[0]
    if n_pairs > sample_size:
        indices = np.random.choice(n_pairs, size=sample_size, replace=False)
    else:
        indices = np.s_[1:n_pairs]
    sampled_pairs = pairs[indices]

    if kernel == "matern":
        gp_kernel = 1.0 * Matern(length_scale=length_scale, nu=matern_nu)
    else:
        if kernel != "rbf":
            warnings.warn(
                f"Kernel {kernel!r} is not supported; falling back to RBF",
                stacklevel=2,
            )
        gp_kernel = 1.0 * RBF(
            length_scale=length_scale,
            length_scale_bounds=rbf_length_scale_bounds,
        )

    regressor = GaussianProcessRegressor(
        kernel=gp_kernel,
        n_restarts_optimizer=optimizer_restarts,
        alpha=alpha,
        random_state=random_state,
    )
    regressor.fit(sampled_pairs[:, :-1], sampled_pairs[:, -1])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dump(regressor, output_path)
    return regressor


def load_closure(path: str | Path = "closure.joblib") -> Any:
    """Load a previously fitted closure."""
    return load(Path(path))
