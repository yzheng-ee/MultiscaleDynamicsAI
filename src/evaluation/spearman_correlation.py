"""Spearman rank correlation metric."""

import operator

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import rankdata


def spearman_correlation(
    target: ArrayLike,
    prediction: ArrayLike,
    *,
    axis: int = 0,
) -> float | NDArray[np.float64]:
    """Compute Spearman rank correlation along an observation axis.

    The inputs are ranked independently along ``axis``, and the Pearson
    correlation of those ranks is returned. For arrays shaped ``(time,
    channels)``, the default ``axis=0`` returns one correlation per channel.
    Constant or NaN-containing slices produce NaN because their rank
    correlation is undefined.

    Args:
        target: Ground-truth values.
        prediction: Predicted values with the same shape as ``target``.
        axis: Observation axis along which to compute correlation. Defaults to
            the first axis.

    Returns:
        The Spearman correlation. A scalar is returned for one-dimensional
        inputs; otherwise an array containing one correlation for every
        remaining index combination is returned.

    Raises:
        ValueError: If the inputs have different shapes, are empty, are
            scalar, or contain fewer than two observations along ``axis``.
    """
    target_array = np.asarray(target, dtype=np.float64)
    prediction_array = np.asarray(prediction, dtype=np.float64)

    if target_array.shape != prediction_array.shape:
        raise ValueError(
            "target and prediction must have the same shape; "
            f"got {target_array.shape} and {prediction_array.shape}."
        )
    if target_array.size == 0:
        raise ValueError("target and prediction must not be empty.")
    if target_array.ndim == 0:
        raise ValueError("target and prediction must have at least one dimension.")

    try:
        normalized_axis = operator.index(axis)
    except TypeError as error:
        raise TypeError("axis must be an integer.") from error
    if not -target_array.ndim <= normalized_axis < target_array.ndim:
        raise ValueError(
            f"axis {normalized_axis} is out of bounds for an array with "
            f"{target_array.ndim} dimensions."
        )
    normalized_axis %= target_array.ndim
    if target_array.shape[normalized_axis] < 2:
        raise ValueError("correlation requires at least two observations.")

    target_ranks = rankdata(
        target_array,
        axis=normalized_axis,
        nan_policy="propagate",
    )
    prediction_ranks = rankdata(
        prediction_array,
        axis=normalized_axis,
        nan_policy="propagate",
    )

    target_centered = target_ranks - np.mean(
        target_ranks,
        axis=normalized_axis,
        keepdims=True,
    )
    prediction_centered = prediction_ranks - np.mean(
        prediction_ranks,
        axis=normalized_axis,
        keepdims=True,
    )
    numerator = np.sum(
        target_centered * prediction_centered,
        axis=normalized_axis,
    )
    denominator = np.sqrt(
        np.sum(np.square(target_centered), axis=normalized_axis)
        * np.sum(np.square(prediction_centered), axis=normalized_axis)
    )

    result = np.full_like(denominator, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator != 0.0)

    if np.ndim(result) == 0:
        return float(result)
    return result
