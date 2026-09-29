"""Mean squared error metric."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def mean_squared_error(
    target: ArrayLike,
    prediction: ArrayLike,
    *,
    axis: int | tuple[int, ...] | None = None,
) -> float | NDArray[np.float64]:
    """Compute the average squared difference between target and prediction.

    By default, the average is taken over every element. Pass ``axis`` to
    retain the remaining dimensions, for example ``axis=0`` to obtain one
    error per channel for arrays shaped ``(time, channels)``. Inputs are
    converted to double precision before subtraction. NaN and infinite values
    propagate according to NumPy's standard arithmetic rules.

    Args:
        target: Ground-truth values.
        prediction: Predicted values with the same shape as ``target``.
        axis: Axis or axes over which to average. Defaults to all axes.

    Returns:
        The mean squared error. A scalar is returned when the reduction removes
        every dimension; otherwise an array is returned.

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

    result = np.mean(np.square(target_array - prediction_array), axis=axis)
    if np.ndim(result) == 0:
        return float(result)
    return result
