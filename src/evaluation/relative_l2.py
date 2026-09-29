"""Relative L2 error metric."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def relative_l2_error(
    target: ArrayLike,
    prediction: ArrayLike,
    *,
    axis: int | tuple[int, ...] | None = None,
) -> float | NDArray[np.float64]:
    """Compute the L2 norm of the error relative to the target L2 norm.

    By default, both norms are computed over every element. Pass ``axis`` to
    compute the norms over selected dimensions while retaining the remaining
    dimensions. Inputs are converted to double precision before subtraction.
    If a target norm is zero, the corresponding result is zero when the error
    norm is also zero and infinity otherwise. NaN values propagate according
    to NumPy's standard arithmetic rules.

    Args:
        target: Ground-truth values.
        prediction: Predicted values with the same shape as ``target``.
        axis: Axis or axes over which to compute each norm. Defaults to all
            axes.

    Returns:
        The relative L2 error. A scalar is returned when the reduction removes
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

    error_norm = np.sqrt(np.sum(np.square(target_array - prediction_array), axis=axis))
    target_norm = np.sqrt(np.sum(np.square(target_array), axis=axis))

    result = np.full_like(target_norm, np.nan, dtype=np.float64)
    nonzero_target = target_norm != 0.0
    np.divide(
        error_norm,
        target_norm,
        out=result,
        where=nonzero_target,
    )
    result = np.where(
        (target_norm == 0.0) & (error_norm == 0.0),
        0.0,
        result,
    )
    result = np.where(
        (target_norm == 0.0) & (error_norm > 0.0),
        np.inf,
        result,
    )

    if np.ndim(result) == 0:
        return float(result)
    return result
