"""Forecast metrics for chaotic systems.

Two families, and the distinction matters more here than in ordinary forecasting. 
PANDA is explicitly a weather model, optimised for short-term pointwise accuracy, not a climate model, 
so it is expected to do well on the pointwise metrics and to eventually regress to the mean on the distributional ones. 
Reporting only one family would flatter or unfairly penalise it.

Pointwise (short horizons): sMAPE, MAE, RMSE, and valid prediction time.
Distributional (long horizons): agreement of the invariant measure and of the power spectrum between forecast and truth.

All functions take time-first arrays (..., horizon, n_channels).
"""

from __future__ import annotations
import numpy as np

_EPS = 1e-12


def _check_shapes(prediction: np.ndarray, target: np.ndarray) -> None:
    if prediction.shape != target.shape:
        raise ValueError(f"shape mismatch: prediction {prediction.shape} vs target {target.shape}")


def smape(prediction: np.ndarray, target: np.ndarray, axis: tuple[int, ...] | int | None = None) -> np.ndarray:
    """Symmetric mean absolute percentage error, in percent (PANDA's headline metric).
    Uses the 0-200% convention 200 * |p - t| / (|p| + |t|), matching the numbers reported in the PANDA paper.
    """
    _check_shapes(prediction, target)
    denominator = np.abs(prediction) + np.abs(target)
    ratio = np.where(denominator < _EPS, 0.0, 200.0 * np.abs(prediction - target) / (denominator + _EPS))
    return np.mean(ratio, axis=axis)


def mae(prediction: np.ndarray, target: np.ndarray, axis: tuple[int, ...] | int | None = None) -> np.ndarray:
    _check_shapes(prediction, target)
    return np.mean(np.abs(prediction - target), axis=axis)


def rmse(prediction: np.ndarray, target: np.ndarray, axis: tuple[int, ...] | int | None = None) -> np.ndarray:
    _check_shapes(prediction, target)
    return np.sqrt(np.mean((prediction - target) ** 2, axis=axis))


def error_vs_lead_time(prediction: np.ndarray, target: np.ndarray, metric: str = "rmse") -> np.ndarray:
    """Error as a function of lead time, averaged over ensemble and channels. """
    _check_shapes(prediction, target)
    func = {"rmse": rmse, "mae": mae, "smape": smape}[metric]
    return func(prediction, target, axis=(0, 2))


def valid_prediction_time(prediction: np.ndarray,target: np.ndarray,dt: float,*,threshold: float = 0.3,climatology_std: float | np.ndarray | None = None,) -> np.ndarray:
    """Time until normalised RMSE first exceeds threshold.

    The standard horizon summary for chaotic forecasting. Errors are normalised by the climatological standard deviation, 
    so a value of 1 means "no better than guessing the long-run distribution" threshold = 0.3 is the usual convention.

    dt: Sampling interval, so the result is in time units. Divide by the Lyapunov time to make it comparable across systems.
    climatology_std: Per-channel standard deviation of the attractor. Defaults to the standard deviation of target, 
        which is a reasonable proxy only when the evaluation set spans the attractor, pass it otherwise.
    """
    _check_shapes(prediction, target)
    if climatology_std is None:
        climatology_std = np.std(target, axis=(0, 1))
    normalised = np.sqrt(np.mean(((prediction - target) / (np.asarray(climatology_std) + _EPS)) ** 2, axis=2))
    horizon = normalised.shape[1]
    exceeded = normalised > threshold
    first = np.where(exceeded.any(axis=1), exceeded.argmax(axis=1), horizon)
    return first * dt


def invariant_measure_kl(prediction: np.ndarray,target: np.ndarray,*,n_bins: int = 50,bin_range: tuple[float, float] | None = None,) -> float:
    """KL divergence between the forecast and true one-point marginal densities.

    A climate metric: it ignores timing entirely and asks whether the forecast visits the right part of state space with the right frequency. 
    A model that has regressed to the mean scores badly here even if its short-horizon RMSE was fine.
    """
    flat_pred = np.asarray(prediction).reshape(-1)
    flat_true = np.asarray(target).reshape(-1)
    if bin_range is None:
        lo = min(flat_pred.min(), flat_true.min())
        hi = max(flat_pred.max(), flat_true.max())
        bin_range = (float(lo), float(hi))

    p_hist, _ = np.histogram(flat_pred, bins=n_bins, range=bin_range, density=True)
    q_hist, _ = np.histogram(flat_true, bins=n_bins, range=bin_range, density=True)
    width = (bin_range[1] - bin_range[0]) / n_bins

    p = p_hist * width + _EPS
    q = q_hist * width + _EPS
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def spectral_hellinger(prediction: np.ndarray, target: np.ndarray, dt: float) -> float:
    """Hellinger distance between normalised power spectra, averaged over channels.

    Captures whether a forecast reproduces the *timescales* of the system,
    a trajectory can have the right marginal distribution and still oscillate at the wrong frequency.
    """
    from msdyn.diagnostics.spectra import power_spectrum

    _check_shapes(prediction, target)
    distances = []
    for window in range(prediction.shape[0]):
        _, p_power = power_spectrum(prediction[window], dt)
        _, q_power = power_spectrum(target[window], dt)
        p = p_power / (p_power.sum() + _EPS)
        q = q_power / (q_power.sum() + _EPS)
        distances.append(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))
    return float(np.mean(distances))
