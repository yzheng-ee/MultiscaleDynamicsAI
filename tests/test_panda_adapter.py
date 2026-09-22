"""PANDA adapter contract, and calibration of the sampling-density estimator."""

from __future__ import annotations
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("panda.patchtst.patchtst")

from msdyn.models import PandaForecaster  # noqa: E402
pytestmark = pytest.mark.panda


@pytest.fixture(scope="module")
def panda() -> PandaForecaster:
    return PandaForecaster(device="cpu", batch_size=4)


def _smooth_context(n_windows: int, n_channels: int, length: int = 512) -> np.ndarray:
    t = np.linspace(0, 12 * np.pi, length)
    return np.stack(
        [np.stack([np.sin((1 + 0.2 * c) * t + w) for c in range(n_channels)], axis=-1)for w in range(n_windows)])


def test_forecast_shape_and_finiteness(panda: PandaForecaster) -> None:
    context = _smooth_context(2, 3)
    prediction = panda.forecast(context, panda.prediction_length)
    assert prediction.shape == (2, panda.prediction_length, 3)
    assert np.all(np.isfinite(prediction))


def test_accepts_more_channels_than_it_was_trained_on(panda: PandaForecaster) -> None:
    """PANDA was trained at d=3; channel attention is meant to generalise."""
    prediction = panda.forecast(_smooth_context(1, 9), panda.prediction_length)
    assert prediction.shape == (1, panda.prediction_length, 9)


def test_rollout_extends_beyond_the_native_horizon(panda: PandaForecaster) -> None:
    horizon = 3 * panda.prediction_length
    prediction = panda.forecast(_smooth_context(1, 3), horizon)
    assert prediction.shape[1] == horizon
    assert panda.rollout_chunks(horizon) == 3


def test_horizon_not_a_multiple_of_the_chunk_is_truncated(panda: PandaForecaster) -> None:
    prediction = panda.forecast(_smooth_context(1, 3), 200)
    assert prediction.shape[1] == 200


def test_short_context_fails_loudly_rather_than_padding(panda: PandaForecaster) -> None:
    short = _smooth_context(1, 3, length=256)
    with pytest.raises(ValueError, match="exactly 512"):
        panda.forecast(short, panda.prediction_length)


def test_long_context_is_trimmed_to_the_most_recent_window(panda: PandaForecaster) -> None:
    long_context = _smooth_context(1, 3, length=700)
    trimmed = panda._prepare_context(long_context)
    np.testing.assert_allclose(trimmed, long_context[:, -512:, :])


def test_batching_does_not_change_the_result(panda: PandaForecaster) -> None:
    context = _smooth_context(6, 3)
    batched = panda.forecast(context, panda.prediction_length)
    one_at_a_time = np.concatenate(
        [panda.forecast(context[i : i + 1], panda.prediction_length) for i in range(6)]
    )
    np.testing.assert_allclose(batched, one_at_a_time, rtol=1e-4, atol=1e-5)


def test_growing_context_is_refused(panda: PandaForecaster) -> None:
    with pytest.raises(NotImplementedError, match="fixed patch count"):
        PandaForecaster(device="cpu", sliding_context=False)


@pytest.mark.slow
def test_the_seed_pins_the_patch_indices_the_checkpoint_does_not_carry() -> None:
    """The released weights do not fully determine the model.

    PANDA's patch embedding draws `patch_indices` with torch.randint and never registers them, so
    they are absent from the state dict and redrawn on every construction. Equal seeds must give
    identical forecasts; different seeds must not, because the gap between them is the noise floor
    under every PANDA number in this project.
    """
    context = _smooth_context(2, 3)
    horizon = 8

    first = PandaForecaster(device="cpu", batch_size=4, seed=0).forecast(context, horizon)
    again = PandaForecaster(device="cpu", batch_size=4, seed=0).forecast(context, horizon)
    other = PandaForecaster(device="cpu", batch_size=4, seed=1).forecast(context, horizon)

    np.testing.assert_allclose(first, again, rtol=1e-6, atol=1e-6)
    assert not np.allclose(first, other, rtol=1e-3, atol=1e-3), (
        "different seeds gave the same forecast; either the upstream patch_indices are now "
        "registered in the checkpoint (good -- drop this test) or the seed is not reaching them"
    )


@pytest.mark.slow
def test_period_estimator_recovers_the_density_dysts_generated_at() -> None:
    """Calibrate `dominant_period` against trajectories of known sampling density.

    dysts generates at exactly 102.4 samples per its own stored period. 
    We cannot reproduce that estimate without its metadata, so this test pins how far our independent estimator drifts,
    and asserts the centroid method is the more stable of the two, which is why it is the default.
    """
    flows = pytest.importorskip("dysts.flows")
    from msdyn.diagnostics import points_per_period

    names = ["Rossler", "Halvorsen", "SprottA", "Chua", "Lorenz", "QiChen"]
    centroid, peak = [], []
    for name in names:
        system = getattr(flows, name)()
        _, traj = system.make_trajectory(4096, pts_per_period=4096 // 40, return_times=True)
        centroid.append(points_per_period(traj, 1.0, method="centroid"))
        peak.append(points_per_period(traj, 1.0, method="peak"))

    centroid, peak = np.array(centroid), np.array(peak)
    assert np.std(centroid) < np.std(peak), (f"centroid should be the more stable estimator: centroid {centroid.round(0)}, peak {peak.round(0)}")
    # Every centroid estimate should land within a factor of two of the truth.
    assert np.all((centroid > 51.2) & (centroid < 204.8)), centroid.round(0)
