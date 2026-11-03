"""Internal validation helpers for named numerical parameters and arrays."""

import numpy as np
from numpy.typing import ArrayLike


def _validate_integer(parameters: list[tuple[str, int]]) -> None:
    """Require integer scalars, excluding Python and NumPy booleans."""
    for name, value in parameters:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")


def _validate_real(parameters: list[tuple[str, ArrayLike]]) -> None:
    """Reject complex scalars and arrays before real-valued conversion."""
    for name, value in parameters:
        if np.iscomplexobj(value):
            raise ValueError(f"{name} must be real")


def _validate_finite(parameters: list[tuple[str, ArrayLike]]) -> None:
    """Require finite scalars or arrays with all finite elements."""
    for name, value in parameters:
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be finite")


def _validate_positive(parameters: list[tuple[str, ArrayLike]]) -> None:
    """Require values greater than zero; positive infinity is permitted."""
    for name, value in parameters:
        if not np.all(np.asarray(value) > 0):
            raise ValueError(f"{name} must be positive")


def _validate_nonnegative(parameters: list[tuple[str, ArrayLike]]) -> None:
    """Require values greater than or equal to zero; positive infinity is permitted."""
    for name, value in parameters:
        if not np.all(np.asarray(value) >= 0):
            raise ValueError(f"{name} must be nonnegative")
