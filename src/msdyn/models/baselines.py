"""Baseline forecasters.

The interesting one is :class:`ClosureModelForecaster`. The others exist to calibrate the scale of the problem,
a chaotic-system benchmark where a pretrained transformer fails to beat persistence is telling you about the sampling rate, not about the model.
"""

from __future__ import annotations
from collections.abc import Callable
import numpy as np
from msdyn.models.base import validate_forecast
from msdyn.systems.integrate import integrate_rk4
from msdyn.systems.l96 import L96Params, L96SingleScale


class PersistenceForecaster:
    """Hold the last observed value. The floor any model must clear."""
    name = "persistence"

    def forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        prediction = np.repeat(context[:, -1:, :], horizon, axis=1)
        validate_forecast(context, prediction, horizon)
        return prediction

class ClimatologyForecaster:
    """Predict the per-channel mean of the context window.
    The natural long-horizon asymptote: any forecaster that has lost all skill should converge to roughly this.
    """

    name = "climatology"
    def forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        mean = context.mean(axis=1, keepdims=True)
        prediction = np.repeat(mean, horizon, axis=1)
        validate_forecast(context, prediction, horizon)
        return prediction


class LinearAutoregressiveForecaster:
    """Vector autoregression fitted per evaluation, rolled out autoregressively.

    Fitted on the context window itself, so it stays within the same info budget as every other forecaster here. 
    "local linear model" reference point: it measures how much of the forecast skill is available without any nonlinear structure at all.
    """

    name = "linear-ar"

    def __init__(self, order: int = 4, ridge: float = 1e-6) -> None:
        self.order = order
        self.ridge = ridge

    def _fit_one(self, window: np.ndarray) -> np.ndarray:
        """Least-squares VAR coefficients for one window (T, C)."""
        n_times, n_channels = window.shape
        lags = self.order
        n_rows = n_times - lags
        design = np.empty((n_rows, lags * n_channels + 1))
        for lag in range(lags):
            design[:, lag * n_channels : (lag + 1) * n_channels] = window[lags - lag - 1 : -lag - 1 or None]
        design[:, -1] = 1.0
        targets = window[lags:]

        gram = design.T @ design + self.ridge * np.eye(design.shape[1])
        return np.linalg.solve(gram, design.T @ targets)

    def forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        n_windows, _, n_channels = context.shape
        predictions = np.empty((n_windows, horizon, n_channels))

        for i in range(n_windows):
            coefficients = self._fit_one(context[i])
            history = list(context[i][-self.order :])
            for step in range(horizon):
                features = np.concatenate([*reversed(history[-self.order :]), [1.0]])
                next_state = features @ coefficients
                predictions[i, step] = next_state
                history.append(next_state)

        validate_forecast(context, predictions, horizon)
        return predictions


class ClosureModelForecaster:
    """Integrate the single-scale closure model forward from the last context state.

    physics baseline: reduced model that the averaging principle licenses. 
    Comparing a pretrained sequence model against it asks whether general-purpose knowledge of chaos beats a purpose-built, 
    theory-derived reduction of this specific system.

    Note: asymmetry. this forecaster knows F, h_x and the ring topology exactly, and needs only the last state
    PANDA knows none of that and sees only the trajectory.
    """

    name = "closure-model"

    def __init__(
        self,
        params: L96Params,
        closure: Callable[[np.ndarray], np.ndarray],
        sample_dt: float,
        *,
        dt: float = 1e-3,
        name: str | None = None,
    ) -> None:
        self.params = params
        self.system = L96SingleScale(params, closure)
        self.sample_dt = sample_dt
        self.dt = min(dt, sample_dt)
        if name is not None:
            self.name = name

    def forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        if context.shape[2] != self.params.K:
            raise ValueError(f"closure model forecasts the {self.params.K} slow variables, got {context.shape[2]} channels")
        sample_every = max(1, int(round(self.sample_dt / self.dt)))
        dt = self.sample_dt / sample_every

        initial = context[:, -1, :]
        _, traj = integrate_rk4(self.system, initial, t_end=horizon * self.sample_dt, dt=dt, sample_every=sample_every)

        # traj is (horizon + 1, n_windows, K) including the initial state.
        prediction = np.transpose(traj[1 : horizon + 1], (1, 0, 2))
        validate_forecast(context, prediction, horizon)
        return prediction


class PerfectModelForecaster:
    """Integrate the full multiscale system forward. skill ceiling.

    Requires the complete state (slow AND fast), so it is not a fair competitor; 
    it quantifies how much of the forecast error is irreducible given the sampling rate and how much comes from 
    not observing the fast variables.
    """

    name = "perfect-model"

    def __init__(self, params: L96Params, sample_dt: float, *, dt: float | None = None) -> None:
        from msdyn.systems.integrate import suggested_dt
        from msdyn.systems.l96 import L96Multiscale

        self.params = params
        self.system = L96Multiscale(params)
        self.sample_dt = sample_dt
        self.dt = dt if dt is not None else suggested_dt(params.eps)

    def forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        if context.shape[2] != self.params.dim:
            raise ValueError(f"perfect model needs the full {self.params.dim}-dimensional state, got {context.shape[2]} channels")
        sample_every = max(1, int(round(self.sample_dt / self.dt)))
        dt = self.sample_dt / sample_every
        _, traj = integrate_rk4(self.system, context[:, -1, :], t_end=horizon * self.sample_dt, dt=dt, sample_every=sample_every,)
        prediction = np.transpose(traj[1 : horizon + 1], (1, 0, 2))
        validate_forecast(context, prediction, horizon)
        return prediction
