"""Metrics, windowing, and the forecaster contract."""

from __future__ import annotations

import numpy as np
import pytest

from msdyn.evaluation import (error_vs_lead_time,invariant_measure_kl,mae,make_windows,relative_l2,rmse,smape,spectral_hellinger,valid_prediction_time,)
from msdyn.evaluation.protocol import compare_forecasters, evaluate_forecaster
from msdyn.models import (ClimatologyForecaster,ClosureModelForecaster,LinearAutoregressiveForecaster,PersistenceForecaster,)
from msdyn.systems.l96 import L96Params

# Metrics                                                                       

def test_metrics_are_zero_for_a_perfect_forecast() -> None:
    rng = np.random.default_rng(0)
    target = rng.normal(size=(4, 16, 3))
    assert smape(target, target) == pytest.approx(0.0)
    assert mae(target, target) == pytest.approx(0.0)
    assert rmse(target, target) == pytest.approx(0.0)


def test_smape_uses_the_0_to_200_convention() -> None:
    """Opposite signs of equal magnitude is the worst case: 200%."""
    target = np.ones((1, 1, 1))
    assert smape(-target, target) == pytest.approx(200.0)
    assert smape(3 * target, target) == pytest.approx(100.0)


def test_relative_l2_is_zero_for_a_perfect_forecast_and_one_for_a_null_one() -> None:
    target = np.random.default_rng(0).normal(size=(4, 16, 3)) + 5.0
    assert relative_l2(target, target) == pytest.approx(0.0)
    assert relative_l2(np.zeros_like(target), target) == pytest.approx(1.0)


def test_relative_l2_is_scale_free() -> None:
    """Multiplying the system's units by 100 must not change the error."""
    rng = np.random.default_rng(1)
    target = rng.normal(size=(2, 8, 3))
    prediction = target + rng.normal(scale=0.1, size=target.shape)
    assert relative_l2(100 * prediction, 100 * target) == pytest.approx(relative_l2(prediction, target))


def test_relative_l2_reduces_per_window() -> None:
    """axis=(1, 2) gives one number per window -- what an error bar is computed from."""
    target = np.ones((3, 4, 2))
    prediction = target.copy()
    prediction[1] *= 2.0  # this window is wrong by exactly 100% of its own norm
    per_window = relative_l2(prediction, target, axis=(1, 2))
    assert per_window.shape == (3,)
    np.testing.assert_allclose(per_window, [0.0, 1.0, 0.0], atol=1e-9)


def test_metrics_reject_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        rmse(np.zeros((2, 3, 1)), np.zeros((2, 4, 1)))


def test_error_vs_lead_time_grows_for_a_drifting_forecast() -> None:
    target = np.zeros((5, 10, 2))
    drift = np.arange(10)[None, :, None] * np.ones((5, 10, 2))
    curve = error_vs_lead_time(drift, target, "rmse")
    assert curve.shape == (10,)
    assert np.all(np.diff(curve) > 0)


def test_valid_prediction_time_detects_the_crossing() -> None:
    horizon, dt = 20, 0.5
    target = np.zeros((1, horizon, 1))
    prediction = np.zeros((1, horizon, 1))
    prediction[0, 10:, 0] = 5.0  # error jumps well past threshold at step 10
    vpt = valid_prediction_time(prediction, target, dt, threshold=0.3, climatology_std=np.array([1.0]))
    assert vpt[0] == pytest.approx(10 * dt)


def test_valid_prediction_time_returns_full_horizon_when_never_exceeded() -> None:
    horizon, dt = 12, 0.25
    target = np.zeros((1, horizon, 1))
    vpt = valid_prediction_time(target, target, dt, climatology_std=np.array([1.0]))
    assert vpt[0] == pytest.approx(horizon * dt)


def test_invariant_measure_kl_is_zero_for_identical_samples() -> None:
    rng = np.random.default_rng(1)
    sample = rng.normal(size=(3, 500, 2))
    assert invariant_measure_kl(sample, sample) == pytest.approx(0.0, abs=1e-9)


def test_invariant_measure_kl_detects_a_shifted_distribution() -> None:
    rng = np.random.default_rng(2)
    target = rng.normal(size=(3, 2000, 1))
    shifted = target + 3.0
    assert invariant_measure_kl(shifted, target) > invariant_measure_kl(target + 0.05, target)


def test_spectral_hellinger_is_zero_for_identical_spectra() -> None:
    t = np.linspace(0, 20, 256)
    signal = np.sin(t)[None, :, None]
    assert spectral_hellinger(signal, signal, dt=t[1] - t[0]) == pytest.approx(0.0, abs=1e-9)


def test_spectral_hellinger_separates_different_frequencies() -> None:
    t = np.linspace(0, 20, 256)
    dt = t[1] - t[0]
    slow = np.sin(t)[None, :, None]
    fast = np.sin(9 * t)[None, :, None]
    assert spectral_hellinger(fast, slow, dt) > 0.5


# Windowing                                                                     

def test_make_windows_shapes_and_alignment() -> None:
    n_traj, n_times, n_channels = 2, 100, 3
    context_length, horizon, stride = 10, 5, 5
    trajectories = np.arange(n_traj * n_times * n_channels, dtype=float).reshape(n_traj, n_times, n_channels)
    windows = make_windows(trajectories, context_length, horizon, dt=0.1, stride=stride)

    starts = np.arange(0, n_times - (context_length + horizon) + 1, stride)
    assert windows.n_windows == n_traj * starts.size
    assert windows.context.shape[1:] == (context_length, n_channels)
    assert windows.target.shape[1:] == (horizon, n_channels)

    # Windows are laid out trajectory-major, and the target must begin exactly
    # where the context ends -- no gap, no overlap, no crossing between trajectories.
    for i in range(windows.n_windows):
        traj = i // starts.size
        start = starts[i % starts.size]
        source = trajectories[traj]
        np.testing.assert_array_equal(windows.context[i], source[start : start + context_length])
        np.testing.assert_array_equal(windows.target[i], source[start + context_length : start + context_length + horizon],)


def test_make_windows_respects_max_windows_and_is_reproducible() -> None:
    trajectories = np.random.default_rng(0).normal(size=(4, 200, 2))
    a = make_windows(trajectories, 20, 10, 0.1, stride=5, max_windows=7, seed=3)
    b = make_windows(trajectories, 20, 10, 0.1, stride=5, max_windows=7, seed=3)
    assert a.n_windows == 7
    np.testing.assert_array_equal(a.context, b.context)


def test_make_windows_rejects_too_short_trajectories() -> None:
    with pytest.raises(ValueError, match="need at least"):
        make_windows(np.zeros((1, 10, 2)), context_length=20, horizon=5, dt=0.1)


def test_lead_times_in_lyapunov_units() -> None:
    windows = make_windows(np.zeros((1, 50, 2)), 10, 5, dt=0.2, lyapunov_time=2.0)
    np.testing.assert_allclose(windows.lead_times_lyapunov, np.arange(1, 6) * 0.1)


def test_select_channels_keeps_everything_aligned() -> None:
    trajectories = np.random.default_rng(0).normal(size=(1, 60, 5))
    windows = make_windows(trajectories, 20, 5, 0.1)
    subset = windows.select_channels([0, 2])
    assert subset.context.shape[2] == 2
    assert subset.climatology_std.shape == (2,)
    np.testing.assert_array_equal(subset.context[..., 1], windows.context[..., 2])


# Forecasters                                                                   

def test_baselines_honour_the_forecaster_contract() -> None:
    context = np.random.default_rng(0).normal(size=(3, 40, 4))
    for forecaster in [
        PersistenceForecaster(),
        ClimatologyForecaster(),
        LinearAutoregressiveForecaster(order=3),
    ]:
        prediction = forecaster.forecast(context, horizon=7)
        assert prediction.shape == (3, 7, 4), forecaster.name
        assert np.all(np.isfinite(prediction)), forecaster.name


def test_persistence_repeats_the_last_observation() -> None:
    context = np.random.default_rng(0).normal(size=(2, 10, 3))
    prediction = PersistenceForecaster().forecast(context, horizon=4)
    for step in range(4):
        np.testing.assert_allclose(prediction[:, step], context[:, -1])


def test_linear_ar_is_exact_on_a_linear_recursion() -> None:
    """x_{n+1} = 0.5 x_n is in the VAR hypothesis class, so it must be recovered."""
    series = 0.5 ** np.arange(60, dtype=float)
    context = series[None, :50, None]
    prediction = LinearAutoregressiveForecaster(order=2).forecast(context, horizon=5)
    np.testing.assert_allclose(prediction[0, :, 0], series[50:55], rtol=1e-4, atol=1e-8)


def test_closure_forecaster_rejects_the_wrong_channel_count() -> None:
    params = L96Params()
    forecaster = ClosureModelForecaster(params, lambda x: np.zeros_like(x), sample_dt=0.01)
    with pytest.raises(ValueError, match="slow variables"):
        forecaster.forecast(np.zeros((1, 10, 3)), horizon=4)


def test_closure_forecaster_integrates_the_reduced_model() -> None:
    """With h_x = 0 and F = 0 the slow system decays; the forecast must decay too."""
    params = L96Params(h_x=0.0, F=0.0)
    forecaster = ClosureModelForecaster(params, lambda x: np.zeros_like(x), sample_dt=0.05)
    context = np.full((2, 10, params.K), 0.01)  # small, so the quadratic term is negligible
    prediction = forecaster.forecast(context, horizon=10)
    assert np.all(np.abs(prediction[:, -1]) < np.abs(context[:, -1]))


def test_evaluate_forecaster_returns_the_expected_record() -> None:
    trajectories = np.random.default_rng(0).normal(size=(2, 120, 3))
    windows = make_windows(trajectories, 20, 8, dt=0.1, lyapunov_time=1.5)
    record = evaluate_forecaster(PersistenceForecaster(), windows)

    for key in ["model", "smape", "mae", "rmse", "vpt", "vpt_lyapunov", "rmse_vs_lead"]:
        assert key in record
    assert record["rmse_vs_lead"].shape == (8,)


def test_compare_forecasters_scores_every_model() -> None:
    trajectories = np.random.default_rng(0).normal(size=(2, 120, 3))
    windows = make_windows(trajectories, 20, 8, dt=0.1)
    records = compare_forecasters([PersistenceForecaster(), ClimatologyForecaster()], windows)
    assert [r["model"] for r in records] == ["persistence", "climatology"]


def test_make_windows_accepts_a_single_trajectory() -> None:
    """A 2-D input is (n_times, n_channels), not (n_trajectories, n_times)."""
    single = np.random.default_rng(0).normal(size=(80, 4))
    windows = make_windows(single, context_length=20, horizon=10, dt=0.1)
    assert windows.context.shape[1:] == (20, 4)
    np.testing.assert_array_equal(windows.context[0], single[:20])


def test_make_windows_rejects_wrong_dimensionality() -> None:
    with pytest.raises(ValueError, match="expected"):
        make_windows(np.zeros((2, 3, 4, 5)), context_length=2, horizon=1, dt=0.1)
