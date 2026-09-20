"""Precomputed lookup tables for expensive scalar closures.

Motivation: A Gaussian process closure costs a kernel evaluation against its whole training set on EVERY call, 
and the single-scale right-hand side calls it four times per RK4 step. 
Integrating 32 trajectories for 80 time units at dt = 1e-3 means ~3x10^5 GP predictions, minutes of wall-clock, versus fractions of a second for a polynomial.

But the closure is a scalar function of a scalar. Once fitted, tabulating it on a fine grid and interpolating is numerically indistinguishable  from calling it directly, and roughly two orders of magnitude faster.

Extrapolation: A GP posterior mean reverts to the prior (zero) far from its training data,  so an excursion off the attractor would see the fast forcing switch off entirely. 
physically wrong, and a plausible route to a blown-up integration. Linear extrapolation from the endpoint slopes is the safer default and is what `TabulatedClosure` does.
"""

from __future__ import annotations
from collections.abc import Callable
import numpy as np
from msdyn.closures.base import ClosurePairs

class TabulatedClosure:
    """A closure evaluated by interpolation on a precomputed grid.

    closure: Any fitted scalar closure, e.g. msdyn.closures.gp.GaussianProcessClosure
    x_min, x_max: Grid bounds. Choose them from the data the closure was fitted on, widened by 'margin'.
    n_grid: Grid points. 2001 over a range of ~25 gives a spacing of ~0.012, far finer than the GP's fitted length scale (~6), so interpolation error is negligible.
    margin: Fraction of the range added to each side before tabulating.
    """

    def __init__(self,closure: Callable[[np.ndarray], np.ndarray],x_min: float,x_max: float,*,n_grid: int = 2001,margin: float = 0.2,) -> None:
        if x_max <= x_min:
            raise ValueError(f"x_max ({x_max}) must exceed x_min ({x_min})")
        if n_grid < 4:
            raise ValueError(f"n_grid must be at least 4, got {n_grid}")

        span = x_max - x_min
        self.x_min = x_min - margin * span
        self.x_max = x_max + margin * span
        self.grid = np.linspace(self.x_min, self.x_max, n_grid)
        self.values = np.asarray(closure(self.grid)).reshape(-1)

        # Endpoint slopes, used for linear extrapolation outside the grid.
        self._slope_low = (self.values[1] - self.values[0]) / (self.grid[1] - self.grid[0])
        self._slope_high = (self.values[-1] - self.values[-2]) / (self.grid[-1] - self.grid[-2])
        self.source = repr(closure)

    @classmethod
    def from_pairs(cls,closure: Callable[[np.ndarray], np.ndarray],pairs: ClosurePairs,**kwargs: float | int,) -> TabulatedClosure:
        """Tabulate over the range actually visited by the training data."""
        return cls(closure, float(pairs.x.min()), float(pairs.x.max()), **kwargs)  # type: ignore[arg-type]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64)
        out = np.interp(values, self.grid, self.values)

        below = values < self.x_min
        above = values > self.x_max
        if below.any():
            out = np.where(below, self.values[0] + self._slope_low * (values - self.x_min), out)
        if above.any():
            out = np.where(above, self.values[-1] + self._slope_high * (values - self.x_max), out)
        return out.reshape(np.shape(x))

    def fraction_outside(self, x: np.ndarray) -> float:
        """Share of x that falls outside the tabulated range.
        Report this after any long run: a non-negligible value means results depend on the extrapolation rule rather than on the fitted closure.
        """
        values = np.asarray(x)
        return float(np.mean((values < self.x_min) | (values > self.x_max)))

    def __repr__(self) -> str:
        return (f"TabulatedClosure({self.source}, range=[{self.x_min:.2f}, {self.x_max:.2f}], n_grid={self.grid.size})")
