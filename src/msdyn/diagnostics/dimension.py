"""Correlation dimension of an attractor (Grassberger-Procaccia).

*****
Grassberger-Procaccia recipe from PANDA appendix)
*****

The PANDA paper characterises its training corpus by correlation dimension:
2.09 +/- 0.27 for the 129 founder ("base") systems and 2.11 +/- 0.23 for the evolved skew systems.

Citing:  Table 2 in the ICLR 2026 PDF held in Literatures/, but it is numbered Table 4 in at least one OpenReview revision (where Table 4 in our copy is the model-architecture table). 

The paper's recipe, structure:
1. Pairwise Euclidean distances r_ij between trajectory points (i != j).
2. Keep the scaling region r(5%) < r < r(50%) (empirical percentiles).
3. Fit p(r) = Z r^-alpha for r >= r_min by the Clauset-Shalizi-Newman maximum-likelihood estimator alpha_hat = 1 + n / sum ln(r / r_min).
4. Report D2 ~ alpha_hat.

Calibration: estimator is monotonic but compressive. On uniform d-cubes of known dimension it returns:

true d      1     2     3     4     5     6
estimate     1.61  2.14  2.60  3.02  3.39  3.75

It is near-unbiased around D2 ~ 2 precisely where PANDA's corpus sits, so their reported figure is meaningful at face value,
but it increasingly understates higher dimensions. An estimate of 3.4 corresponds to a true D2 nearer 5.

The practical consequence: when comparing Lorenz '96 against PANDA's corpus, the raw gap in these units is a lower bound on the true gap. 
Quote the numbers as "same estimator, same units", never as absolute fractal dimensions.

The compression comes from step 3. The Hill estimator is written for a decaying power-law tail, whereas the correlation integral grows as C(r) ~ r^D2, 
and the fixed 5-50th percentile window is not the r -> 0 limit the theory needs.
We deliberately reproduce the published procedure rather than a corrected one, because the entire purpose is comparability with their reported figure, 
and it does reproduce it: 2.12 +/- 0.20 over ten founder systems against their 2.09 +/- 0.27.
"""

from __future__ import annotations
import numpy as np

PANDA_BASE_SYSTEMS_GP_DIM = (2.09, 0.27)  # founder systems, mean +/- std across systems
PANDA_SKEW_SYSTEMS_GP_DIM = (2.11, 0.23)  # evolved skew systems
_DEFAULT_MAX_POINTS = 2000  # pairwise distances are O(n^2)


def correlation_dimension(trajectory: np.ndarray,*,max_points: int = _DEFAULT_MAX_POINTS,lower_percentile: float = 5.0,upper_percentile: float = 50.0,standardize: bool = True,seed: int = 0,) -> float:
    """Grassberger-Procaccia correlation dimension of a time-first trajectory.
    trajectory: Transients should already be removed.
    max_points: Points subsampled before forming pairwise distances. Cost is quadratic, and the estimate is stable well below the full record.
    standardize: Normalise each channel to zero mean and unit variance first. 
        PANDA instance-normalises every trajectory, so this keeps the comparison fairand stops one large-amplitude channel dominating the metric.

    Returns the estimated D2.
    """
    from scipy.spatial.distance import pdist

    points = np.asarray(trajectory, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError(f"expected (n_times, n_channels), got {points.shape}")
    if points.shape[0] < 100:
        raise ValueError(f"need at least 100 points, got {points.shape[0]}")

    if standardize:
        scale = points.std(axis=0)
        scale[scale == 0] = 1.0
        points = (points - points.mean(axis=0)) / scale

    if points.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        points = points[np.sort(rng.choice(points.shape[0], max_points, replace=False))]

    distances = pdist(points)
    distances = distances[distances > 0]
    lower = np.percentile(distances, lower_percentile)
    upper = np.percentile(distances, upper_percentile)
    scaling_region = distances[(distances > lower) & (distances < upper)]
    if scaling_region.size < 50:
        raise ValueError(f"scaling region has only {scaling_region.size} distances; widen the percentile window or supply a longer trajectory")

    r_min = scaling_region.min()
    return float(1.0 + scaling_region.size / np.sum(np.log(scaling_region / r_min)))


def correlation_dimension_ensemble(trajectories: np.ndarray, **kwargs: float | int | bool) -> tuple[float, float]:
    """Mean and standard deviation of D2 over an ensemble (n, T, C)."""
    values = np.array([correlation_dimension(trajectories[i], **kwargs) for i in range(trajectories.shape[0])])  # type: ignore[arg-type]
    return float(values.mean()), float(values.std())
