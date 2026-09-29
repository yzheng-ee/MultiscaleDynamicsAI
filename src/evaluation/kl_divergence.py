"""Kullback-Leibler divergence estimators."""

from collections.abc import Callable
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.special import logsumexp
from sklearn.neighbors import NearestNeighbors

LogDensity = Callable[[NDArray[np.float64]], ArrayLike]


def _as_state_samples(values: ArrayLike, name: str) -> NDArray[np.float64]:
    """Convert trajectory values to finite state samples."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional.")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one state sample.")
    if array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one state dimension.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


class Distribution(Protocol):
    """Interface required for Monte Carlo distribution comparisons."""

    def sample(
        self,
        n_samples: int,
        random_state: int | np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        """Draw independent samples from the distribution.

        Args:
            n_samples: Number of samples to draw.
            random_state: Seed or NumPy random generator used for sampling.

        Returns:
            Samples whose first axis has length ``n_samples``.
        """
        ...

    def log_density(self, samples: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the log density at each sample.

        Args:
            samples: Values whose first axis indexes samples.

        Returns:
            One log-density value per sample.
        """
        ...


class DistributionFitter(Protocol):
    """Interface for fitting a probability distribution from state samples."""

    def fit(self, samples: ArrayLike) -> Distribution:
        """Fit a probability distribution to observed state samples.

        Args:
            samples: State samples whose first axis indexes observations.

        Returns:
            A fitted distribution that supports sampling and log-density
            evaluation.
        """
        ...


class GaussianMixtureDistribution:
    """Finite mixture of multivariate Gaussian distributions.

    Args:
        means: Component means shaped ``(components, dimensions)``.
        covariances: Positive-definite component covariance matrices shaped
            ``(components, dimensions, dimensions)``.
        weights: Nonnegative component weights shaped ``(components,)``. The
            weights are normalized to sum to one. Defaults to equal weights.

    Raises:
        ValueError: If a parameter has an invalid shape or value, or a
            covariance matrix is not positive definite.
    """

    def __init__(
        self,
        means: ArrayLike,
        covariances: ArrayLike,
        weights: ArrayLike | None = None,
    ) -> None:
        means_array = np.asarray(means, dtype=np.float64)
        covariances_array = np.asarray(covariances, dtype=np.float64)

        if means_array.ndim != 2 or means_array.shape[0] == 0:
            raise ValueError(
                "means must have shape (components, dimensions) with at least "
                "one component."
            )
        if means_array.shape[1] == 0:
            raise ValueError("means must contain at least one dimension.")
        if not np.all(np.isfinite(means_array)):
            raise ValueError("means must contain only finite values.")

        n_components, n_dimensions = means_array.shape
        expected_covariance_shape = (
            n_components,
            n_dimensions,
            n_dimensions,
        )
        if covariances_array.shape != expected_covariance_shape:
            raise ValueError(
                "covariances must have shape "
                f"{expected_covariance_shape}; got {covariances_array.shape}."
            )
        if not np.all(np.isfinite(covariances_array)):
            raise ValueError("covariances must contain only finite values.")
        if not np.allclose(
            covariances_array,
            np.swapaxes(covariances_array, 1, 2),
        ):
            raise ValueError("covariance matrices must be symmetric.")

        try:
            cholesky_factors = np.linalg.cholesky(covariances_array)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "covariance matrices must be positive definite."
            ) from error

        if weights is None:
            weights_array = np.full(n_components, 1.0 / n_components)
        else:
            weights_array = np.asarray(weights, dtype=np.float64)
            if weights_array.shape != (n_components,):
                raise ValueError(
                    "weights must have shape "
                    f"({n_components},); got {weights_array.shape}."
                )
            if not np.all(np.isfinite(weights_array)):
                raise ValueError("weights must contain only finite values.")
            if np.any(weights_array < 0.0):
                raise ValueError("weights must be nonnegative.")
            weight_sum = np.sum(weights_array)
            if weight_sum <= 0.0:
                raise ValueError("at least one weight must be positive.")
            weights_array = weights_array / weight_sum

        self.means = means_array
        self.covariances = covariances_array
        self.weights = weights_array
        self._cholesky_factors = cholesky_factors
        self._log_weights = np.full(n_components, -np.inf, dtype=np.float64)
        np.log(
            weights_array,
            out=self._log_weights,
            where=weights_array > 0.0,
        )
        self._log_determinants = 2.0 * np.sum(
            np.log(np.diagonal(cholesky_factors, axis1=1, axis2=2)),
            axis=1,
        )

    @property
    def n_components(self) -> int:
        """Return the number of Gaussian components."""
        return self.means.shape[0]

    @property
    def n_dimensions(self) -> int:
        """Return the dimension of each sample."""
        return self.means.shape[1]

    def sample(
        self,
        n_samples: int,
        random_state: int | np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        """Draw independent samples from the Gaussian mixture.

        Args:
            n_samples: Number of samples to draw.
            random_state: Seed or NumPy random generator used for sampling.

        Returns:
            Samples shaped ``(n_samples, dimensions)``.

        Raises:
            ValueError: If ``n_samples`` is not a positive integer.
        """
        if not isinstance(n_samples, int) or isinstance(n_samples, bool):
            raise ValueError("n_samples must be a positive integer.")
        if n_samples <= 0:
            raise ValueError("n_samples must be a positive integer.")

        generator = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        component_indices = generator.choice(
            self.n_components,
            size=n_samples,
            p=self.weights,
        )
        standard_normal = generator.normal(
            size=(n_samples, self.n_dimensions),
        )
        transformed_noise = np.einsum(
            "nij,nj->ni",
            self._cholesky_factors[component_indices],
            standard_normal,
        )
        return self.means[component_indices] + transformed_noise

    def log_density(self, samples: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the Gaussian-mixture log density at each sample.

        Args:
            samples: Values shaped ``(samples, dimensions)``. For a
                one-dimensional distribution, a one-dimensional array is also
                accepted.

        Returns:
            One log-density value per sample.

        Raises:
            ValueError: If the samples have an invalid shape or contain
                nonfinite values.
        """
        sample_array = np.asarray(samples, dtype=np.float64)
        if sample_array.ndim == 1 and self.n_dimensions == 1:
            sample_array = sample_array[:, None]
        expected_suffix = (self.n_dimensions,)
        if sample_array.ndim != 2 or sample_array.shape[1:] != expected_suffix:
            raise ValueError(
                "samples must have shape (samples, dimensions) with "
                f"dimensions={self.n_dimensions}; got {sample_array.shape}."
            )
        if sample_array.shape[0] == 0:
            raise ValueError("samples must contain at least one sample.")
        if not np.all(np.isfinite(sample_array)):
            raise ValueError("samples must contain only finite values.")

        component_log_densities = np.empty(
            (sample_array.shape[0], self.n_components),
            dtype=np.float64,
        )
        normalization_constant = self.n_dimensions * np.log(2.0 * np.pi)
        for component in range(self.n_components):
            differences = sample_array - self.means[component]
            whitened = np.linalg.solve(
                self._cholesky_factors[component],
                differences.T,
            ).T
            squared_mahalanobis = np.sum(np.square(whitened), axis=1)
            component_log_densities[:, component] = (
                self._log_weights[component]
                - 0.5
                * (
                    normalization_constant
                    + self._log_determinants[component]
                    + squared_mahalanobis
                )
            )

        return logsumexp(component_log_densities, axis=1)


def monte_carlo_kl_divergence(
    samples: ArrayLike,
    reference_log_density: LogDensity,
    comparison_log_density: LogDensity,
) -> float:
    """Estimate forward KL divergence using reference-distribution samples.

    The samples must be independently drawn from the reference distribution.
    This function only performs Monte Carlo integration; it does not estimate
    either distribution or generate samples.

    Args:
        samples: Samples drawn from the reference distribution. The first axis
            indexes samples, and any remaining axes describe one sample.
        reference_log_density: Function returning the reference log density for
            every sample.
        comparison_log_density: Function returning the comparison log density
            for every sample.

    Returns:
        The estimated forward KL divergence from the reference distribution to
        the comparison distribution, in the logarithm's units.

    Raises:
        ValueError: If no samples are provided or a log-density function does
            not return one value per sample.
    """
    sample_array = np.asarray(samples, dtype=np.float64)
    if sample_array.ndim == 0 or sample_array.shape[0] == 0:
        raise ValueError("samples must contain at least one sample.")

    reference_values = np.asarray(
        reference_log_density(sample_array),
        dtype=np.float64,
    )
    comparison_values = np.asarray(
        comparison_log_density(sample_array),
        dtype=np.float64,
    )
    expected_shape = (sample_array.shape[0],)
    if reference_values.shape != expected_shape:
        raise ValueError(
            "reference_log_density must return one value per sample; "
            f"expected shape {expected_shape}, got {reference_values.shape}."
        )
    if comparison_values.shape != expected_shape:
        raise ValueError(
            "comparison_log_density must return one value per sample; "
            f"expected shape {expected_shape}, got {comparison_values.shape}."
        )

    return float(np.mean(reference_values - comparison_values))


# =============================================================================
# PANDA-specific Gaussian-mixture fitting and KL-divergence protocol
# =============================================================================


def _estimate_simplex_scale(
    neighbor_distances: NDArray[np.float64],
    tolerance: float,
) -> float:
    """Estimate one local scale using PANDA's simplex-neighbor equation."""
    nearest_distance = float(np.min(neighbor_distances))
    shifted_distances = np.maximum(
        neighbor_distances - nearest_distance,
        0.0,
    )
    target_sum = np.log2(neighbor_distances.size)

    def equation(scale: float) -> float:
        return float(
            np.sum(
                np.exp(
                    -shifted_distances / (scale + tolerance)
                )
            )
            - target_sum
        )

    lower_bound = 0.0
    lower_value = equation(lower_bound)
    if lower_value >= 0.0:
        return tolerance

    upper_bound = max(float(np.max(neighbor_distances)), tolerance)
    upper_value = equation(upper_bound)
    while upper_value < 0.0:
        upper_bound *= 2.0
        upper_value = equation(upper_bound)
        if not np.isfinite(upper_bound):
            raise ValueError("Could not bracket the simplex-scale root.")

    scale = brentq(
        equation,
        lower_bound,
        upper_bound,
        xtol=tolerance,
        rtol=4.0 * np.finfo(np.float64).eps,
    )

    regularized_scale = float(scale + tolerance)
    if not np.isfinite(regularized_scale) or regularized_scale <= 0.0:
        raise ValueError("simplex-neighbor scale estimation did not converge.")
    return regularized_scale


def estimate_simplex_neighbor_scales(
    samples: ArrayLike,
    *,
    n_neighbors: int = 10,
    tolerance: float = 1e-6,
) -> NDArray[np.float64]:
    """Estimate PANDA's local isotropic covariance scales.

    Args:
        samples: State samples shaped ``(samples, dimensions)``. A
            one-dimensional input is treated as a scalar trajectory.
        n_neighbors: Number of nearest neighbors used for each local estimate.
        tolerance: Positive numerical tolerance used in scale estimation.

    Returns:
        One positive local covariance scale per state sample.

    Raises:
        ValueError: If the samples or estimator settings are invalid, or scale
            estimation fails.
    """
    sample_array = _as_state_samples(samples, "samples")
    if not isinstance(n_neighbors, int) or isinstance(n_neighbors, bool):
        raise ValueError("n_neighbors must be a positive integer.")
    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be a positive integer.")
    if sample_array.shape[0] <= n_neighbors:
        raise ValueError("samples must contain more states than n_neighbors.")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite.")

    neighbor_search = NearestNeighbors(
        n_neighbors=n_neighbors + 1,
        algorithm="auto",
        metric="euclidean",
    )
    neighbor_search.fit(sample_array)
    distances, _ = neighbor_search.kneighbors(sample_array)
    neighbor_distances = distances[:, 1:]

    return np.array(
        [
            _estimate_simplex_scale(distances_for_sample, tolerance)
            for distances_for_sample in neighbor_distances
        ],
        dtype=np.float64,
    )


class PandaGaussianMixtureFitter:
    """Fit PANDA's adaptive sample-centered Gaussian mixture.

    Args:
        n_neighbors: Number of nearest neighbors used for each local scale.
        tolerance: Positive numerical tolerance used in scale estimation.

    Raises:
        ValueError: If an estimator setting is invalid.
    """

    def __init__(
        self,
        *,
        n_neighbors: int = 10,
        tolerance: float = 1e-6,
    ) -> None:
        if not isinstance(n_neighbors, int) or isinstance(n_neighbors, bool):
            raise ValueError("n_neighbors must be a positive integer.")
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be a positive integer.")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be positive and finite.")

        self.n_neighbors = n_neighbors
        self.tolerance = tolerance

    def fit(self, samples: ArrayLike) -> GaussianMixtureDistribution:
        """Fit a PANDA Gaussian mixture to observed state samples.

        Every state sample becomes an equally weighted component mean. Each
        component receives an isotropic covariance whose scale is estimated
        from its simplex neighbors.

        Args:
            samples: State samples shaped ``(samples, dimensions)``. A
                one-dimensional input is treated as a scalar trajectory.

        Returns:
            A fully specified Gaussian-mixture distribution.

        Raises:
            ValueError: If the samples are invalid or scale estimation fails.
        """
        sample_array = _as_state_samples(samples, "samples")
        scales = estimate_simplex_neighbor_scales(
            sample_array,
            n_neighbors=self.n_neighbors,
            tolerance=self.tolerance,
        )
        identity = np.eye(sample_array.shape[1], dtype=np.float64)
        covariances = scales[:, None, None] * identity[None, :, :]
        return GaussianMixtureDistribution(
            means=sample_array,
            covariances=covariances,
        )


def panda_kl_divergence(
    target: ArrayLike,
    prediction: ArrayLike,
    *,
    n_samples: int = 1_000,
    n_neighbors: int = 10,
    tolerance: float = 1e-6,
    density_floor: float | None = 1e-300,
    random_state: int | np.random.Generator | None = None,
) -> float:
    """Estimate PANDA's forward KL divergence between two trajectories.

    Args:
        target: Ground-truth state samples shaped ``(samples, dimensions)``.
        prediction: Predicted state samples with the same state dimension.
        n_samples: Number of target-mixture samples used for Monte Carlo
            integration.
        n_neighbors: Number of nearest neighbors used for local scales.
        tolerance: Positive numerical tolerance used in scale estimation.
        density_floor: Optional minimum comparison density. Pass ``None`` to
            disable flooring.
        random_state: Seed or NumPy random generator used for sampling.

    Returns:
        The estimated forward KL divergence from the target trajectory
        distribution to the prediction trajectory distribution, in nats.

    Raises:
        ValueError: If an input or estimator setting is invalid.
    """
    target_array = _as_state_samples(target, "target")
    prediction_array = _as_state_samples(prediction, "prediction")
    if target_array.shape[1] != prediction_array.shape[1]:
        raise ValueError(
            "target and prediction must have the same state dimension; "
            f"got {target_array.shape[1]} and {prediction_array.shape[1]}."
        )
    if not isinstance(n_samples, int) or isinstance(n_samples, bool):
        raise ValueError("n_samples must be a positive integer.")
    if n_samples <= 0:
        raise ValueError("n_samples must be a positive integer.")
    if density_floor is not None and (
        not np.isfinite(density_floor) or density_floor <= 0.0
    ):
        raise ValueError("density_floor must be positive and finite or None.")

    fitter = PandaGaussianMixtureFitter(
        n_neighbors=n_neighbors,
        tolerance=tolerance,
    )
    target_distribution = fitter.fit(target_array)
    prediction_distribution = fitter.fit(prediction_array)
    monte_carlo_samples = target_distribution.sample(
        n_samples,
        random_state=random_state,
    )

    if density_floor is None:
        comparison_log_density = prediction_distribution.log_density
    else:
        log_density_floor = float(np.log(density_floor))

        def comparison_log_density(
            values: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            return np.maximum(
                prediction_distribution.log_density(values),
                log_density_floor,
            )

    return monte_carlo_kl_divergence(
        monte_carlo_samples,
        target_distribution.log_density,
        comparison_log_density,
    )
