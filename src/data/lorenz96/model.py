"""Multiscale Lorenz-96 model and its learned single-scale reduction."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


class L96M:
    """Lorenz-96 model with ``K`` slow and ``K * J`` fast variables."""

    def __init__(
        self,
        K: int = 9,
        J: int = 8,
        hx: float = -0.8,
        hy: float = 1.0,
        F: float = 10.0,
        eps: float = 2**-7,
        k0: int = 0,
    ) -> None:
        if K < 4:
            raise ValueError("K must be at least 4")
        if J < 3:
            raise ValueError("J must be at least 3")
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.K = K
        self.J = J
        self.hx = hx
        self.hy = hy
        self.F = F
        self.eps = eps
        self.k0 = k0
        self.predictor: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None
        self.stencil = np.array([0], dtype=int)

        self.xk_star = np.zeros(K)
        self.xk_star[0] = 5
        self.xk_star[1] = 5
        self.xk_star[-1] = 5

    def __call__(
        self, t: float, z: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Evaluate the full multiscale Lorenz-96 right-hand side."""
        del t
        K = self.K
        J = self.J
        rhs = np.empty(K + K * J)

        Yk = self.compute_Yk(z)

        rhs[0] = -z[K - 1] * (z[K - 2] - z[1]) - z[0]
        rhs[1] = -z[0] * (z[K - 1] - z[2]) - z[1]
        rhs[K - 1] = -z[K - 2] * (z[K - 3] - z[0]) - z[K - 1]
        for k in range(2, K - 1):
            rhs[k] = -z[k - 1] * (z[k - 2] - z[k + 1]) - z[k]

        rhs[:K] += self.F
        rhs[:K] += self.hx * Yk

        rhs[self.fidx(0, 0)] = (
            -z[self.fidx(1, 0)]
            * (z[self.fidx(2, 0)] - z[self.fidx(J - 1, K - 1)])
            - z[self.fidx(0, 0)]
        )
        rhs[self.fidx(J - 2, 0)] = (
            -z[self.fidx(J - 1, 0)]
            * (z[self.fidx(0, 1)] - z[self.fidx(J - 3, 0)])
            - z[self.fidx(J - 2, 0)]
        )
        rhs[self.fidx(J - 1, 0)] = (
            -z[self.fidx(0, 1)]
            * (z[self.fidx(1, 1)] - z[self.fidx(J - 2, 0)])
            - z[self.fidx(J - 1, 0)]
        )
        for j in range(1, J - 2):
            rhs[self.fidx(j, 0)] = (
                -z[self.fidx(j + 1, 0)]
                * (z[self.fidx(j + 2, 0)] - z[self.fidx(j - 1, 0)])
                - z[self.fidx(j, 0)]
            )

        rhs[self.fidx(0, K - 1)] = (
            -z[self.fidx(1, K - 1)]
            * (z[self.fidx(2, K - 1)] - z[self.fidx(J - 1, K - 2)])
            - z[self.fidx(0, K - 1)]
        )
        rhs[self.fidx(J - 2, K - 1)] = (
            -z[self.fidx(J - 1, K - 1)]
            * (z[self.fidx(0, 0)] - z[self.fidx(J - 3, K - 1)])
            - z[self.fidx(J - 2, K - 1)]
        )
        rhs[self.fidx(J - 1, K - 1)] = (
            -z[self.fidx(0, 0)]
            * (z[self.fidx(1, 0)] - z[self.fidx(J - 2, K - 1)])
            - z[self.fidx(J - 1, K - 1)]
        )
        for j in range(1, J - 2):
            rhs[self.fidx(j, K - 1)] = (
                -z[self.fidx(j + 1, K - 1)]
                * (z[self.fidx(j + 2, K - 1)] - z[self.fidx(j - 1, K - 1)])
                - z[self.fidx(j, K - 1)]
            )

        for k in range(1, K - 1):
            rhs[self.fidx(0, k)] = (
                -z[self.fidx(1, k)]
                * (z[self.fidx(2, k)] - z[self.fidx(J - 1, k - 1)])
                - z[self.fidx(0, k)]
            )
            rhs[self.fidx(J - 2, k)] = (
                -z[self.fidx(J - 1, k)]
                * (z[self.fidx(0, k + 1)] - z[self.fidx(J - 3, k)])
                - z[self.fidx(J - 2, k)]
            )
            rhs[self.fidx(J - 1, k)] = (
                -z[self.fidx(0, k + 1)]
                * (z[self.fidx(1, k + 1)] - z[self.fidx(J - 2, k)])
                - z[self.fidx(J - 1, k)]
            )
            for j in range(1, J - 2):
                rhs[self.fidx(j, k)] = (
                    -z[self.fidx(j + 1, k)]
                    * (z[self.fidx(j + 2, k)] - z[self.fidx(j - 1, k)])
                    - z[self.fidx(j, k)]
                )

        for k in range(K):
            rhs[K + k * J : K + (k + 1) * J] += self.hy * z[k]
        rhs[K:] /= self.eps
        return rhs

    def regressed(
        self, t: float, x: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Evaluate the learned single-scale Lorenz-96 right-hand side."""
        del t
        K = self.K
        rhs = np.empty(K)

        rhs[0] = -x[K - 1] * (x[K - 2] - x[1]) - x[0]
        rhs[1] = -x[0] * (x[K - 1] - x[2]) - x[1]
        rhs[K - 1] = -x[K - 2] * (x[K - 3] - x[0]) - x[K - 1]
        for k in range(2, K - 1):
            rhs[k] = -x[k - 1] * (x[k - 2] - x[k + 1]) - x[k]

        rhs += self.F
        rhs += self.hx * self.simulate(x)
        return rhs

    def set_predictor(
        self, predictor: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    ) -> None:
        self.predictor = predictor

    def set_G0_predictor(self) -> None:
        self.predictor = lambda x: self.hy * x

    def set_stencil(self, left: int = 0, right: int = 0) -> None:
        if left > right:
            raise ValueError("stencil left must not exceed stencil right")
        self.stencil = np.arange(left, right + 1)

    def set_F(self, value: float) -> None:
        self.F = value

    def set_hx(self, value: float) -> None:
        self.hx = value

    def fidx(self, j: int, k: int) -> int:
        """Return a fast variable's index in the full state vector."""
        return self.K + k * self.J + j

    def fidx_dec(self, j: int, k: int) -> int:
        """Return a fast variable's index in a fast-only state vector."""
        return k * self.J + j

    def simulate(self, slow: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.predictor is None:
            raise RuntimeError("a closure predictor must be set before simulation")
        return np.reshape(self.predictor(self.apply_stencil(slow)), (-1,))

    def compute_Yk(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        return z[self.K :].reshape((self.J, self.K), order="F").sum(axis=0) / self.J

    def gather_pairs(self, time_series: NDArray[np.float64]) -> NDArray[np.float64]:
        n_times = time_series.shape[1]
        pairs = np.empty((self.K * n_times, self.stencil.size + 1))
        for j in range(n_times):
            start = self.K * j
            stop = self.K * (j + 1)
            pairs[start:stop, :-1] = self.apply_stencil(time_series[: self.K, j])
            pairs[start:stop, -1] = self.compute_Yk(time_series[:, j])
        return pairs

    def gather_pairs_k0(
        self, time_series: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        n_times = time_series.shape[1]
        pairs = np.empty((n_times, 2))
        for j in range(n_times):
            pairs[j, 0] = time_series[self.k0, j]
            pairs[j, 1] = time_series[self.K :, j].sum() / self.J
        return pairs

    def apply_stencil(self, slow: NDArray[np.float64]) -> NDArray[np.float64]:
        indices = np.add.outer(np.arange(self.K), self.stencil) % self.K
        return slow[indices]
