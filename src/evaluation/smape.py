"""Symmetric mean absolute percentage error metric."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def symmetric_mean_absolute_percentage_error(
    target: ArrayLike,
    prediction: ArrayLike,
    *,
    axis: int | tuple[int, ...] | None = None,
) -> float | NDArray[np.float64]:
    """Compute the symmetric mean absolute percentage error.

    The result ranges from 0 to 2 for finite inputs. Multiply the result by 100
    to express it in percentage points. An element for which both the target
    and prediction are zero contributes zero error. By default, the average is
    taken over every element. Pass ``axis`` to retain the remaining dimensions.
    Inputs are converted to double precision before the calculation. NaN and
    infinite values propagate according to NumPy's standard arithmetic rules.

    Args:
        target: Ground-truth values.
        prediction: Predicted values with the same shape as ``target``.
        axis: Axis or axes over which to average. Defaults to all axes.

    Returns:
        The symmetric mean absolute percentage error. A scalar is returned
        when the reduction removes every dimension; otherwise an array is
        returned.

    Raises:
        ValueError: If the inputs have different shapes or are empty.
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

    denominator = np.abs(target_array) + np.abs(prediction_array)
    elementwise_error = np.zeros_like(denominator)
    np.divide(
        2.0 * np.abs(target_array - prediction_array),
        denominator,
        out=elementwise_error,
        where=denominator != 0.0,
    )

    result = np.mean(elementwise_error, axis=axis)
    if np.ndim(result) == 0:
        return float(result)
    return result
