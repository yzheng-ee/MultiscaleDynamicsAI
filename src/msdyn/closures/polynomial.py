"""Polynomial closures for the Lorenz '96 single-scale model.

A least-squares polynomial fit of m is the workhorse closure in the stochastic-parameterisation literature (Wilks 2005; Arnold, Moroz & Palmer 2013), 
where a cubic in x_k is the standard choice. It is cheap, smooth, and unlike a Gaussian process fitted on a few hundred points,
costs nothing to evaluate inside an ODE right-hand side, which matters when the single-scale model is integrated for millions of steps.
"""

from __future__ import annotations
import numpy as np
from msdyn.closures.base import ClosurePairs

class PolynomialClosure:
    """Least-squares polynomial m(x) = sum_d c_d x^d.
    degree: Polynomial degree. 3 reproduces the classical Wilks parameterisation.
    """

    def __init__(self, degree: int = 3) -> None:
        if degree < 0:
            raise ValueError(f"degree must be non-negative, got {degree}")
        self.degree = degree
        self.coefficients: np.ndarray | None = None
        self.residual_std: float | None = None

    def fit(self, pairs: ClosurePairs) -> PolynomialClosure:
        """Fit by least squares. Returns 'self' for chaining."""
        coeffs = np.polynomial.polynomial.polyfit(pairs.x, pairs.ybar, self.degree)
        self.coefficients = coeffs
        residuals = pairs.ybar - np.polynomial.polynomial.polyval(pairs.x, coeffs)
        # The residual spread is the part of the fast forcing that a deterministic
        # closure cannot represent; it sets the noise level for a stochastic one.
        self.residual_std = float(np.std(residuals))
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("closure is not fitted; call .fit(pairs) first")
        return np.polynomial.polynomial.polyval(x, self.coefficients)

    def __repr__(self) -> str:
        if self.coefficients is None:
            return f"PolynomialClosure(degree={self.degree}, unfitted)"
        terms = " + ".join(f"{c:+.4g}x^{d}" for d, c in enumerate(self.coefficients))
        return f"PolynomialClosure(m(x) = {terms}, residual_std={self.residual_std:.4g})"


class LinearClosure(PolynomialClosure):
    """The m(x) = h_y * x baseline (set_G0_predictor in the reference code).

    Fitting a degree-1 polynomial is the data-driven version
    passing slope=h_y gives the analytic 'G0' predictor with no fitting at all.
    """

    def __init__(self, slope: float | None = None, intercept: float = 0.0) -> None:
        super().__init__(degree=1)
        if slope is not None:
            self.coefficients = np.array([intercept, slope])
            self.residual_std = None
