"""Dataset round-trip, generation invariants, and dynamical diagnostics."""

from __future__ import annotations
import numpy as np
import pytest
from msdyn.data import (TrajectoryDataset,generate_multiscale,generate_singlescale,load_dataset,save_dataset,)
from msdyn.data.generate import burn_in
from msdyn.diagnostics import (dominant_period,lyapunov_time,max_lyapunov_exponent,points_per_period,power_spectrum,resample_to_points_per_period,subsample_for_points_per_period,)
from msdyn.diagnostics.spectra import PANDA_POINTS_PER_PERIOD
from msdyn.systems.l96 import L96Multiscale, L96Params

# A mildly separated system: still genuinely two-scale, but cheap enough for CI.
FAST_TEST_PARAMS = L96Params(K=5, J=4, eps=2.0**-3)


# Dataset IO                                                                    

def _toy_dataset(kind: str = "multiscale") -> TrajectoryDataset:
    params = L96Params(K=3, J=2)
    n_channels = params.dim if kind == "multiscale" else params.K
    rng = np.random.default_rng(0)
    return TrajectoryDataset(
        trajectories=rng.normal(size=(2, 25, n_channels)),
        times=np.arange(25) * 0.01,
        params=params,
        kind=kind,
        metadata={"dt": 1e-4, "note": "toy"},
    )


def test_dataset_round_trip_preserves_everything(tmp_path) -> None:
    original = _toy_dataset()
    path = save_dataset(original, tmp_path / "toy.npz")
    restored = load_dataset(path)

    np.testing.assert_allclose(restored.trajectories, original.trajectories)
    np.testing.assert_allclose(restored.times, original.times)
    assert restored.params == original.params
    assert restored.kind == original.kind
    assert restored.metadata["note"] == "toy"


def test_dataset_rejects_inconsistent_shapes() -> None:
    params = L96Params(K=3, J=2)
    with pytest.raises(ValueError, match="time axis mismatch"):
        TrajectoryDataset(np.zeros((1, 5, params.dim)), np.zeros(4), params, "multiscale")
    with pytest.raises(ValueError, match="must be"):
        TrajectoryDataset(np.zeros((5, params.dim)), np.zeros(5), params, "multiscale")
    with pytest.raises(ValueError, match="unknown kind"):
        TrajectoryDataset(np.zeros((1, 5, params.dim)), np.zeros(5), params, "bogus")


def test_singlescale_dataset_has_no_fast_variables() -> None:
    dataset = _toy_dataset("singlescale")
    with pytest.raises(AttributeError, match="no fast variables"):
        _ = dataset.fast


def test_fast_average_matches_manual_mean() -> None:
    dataset = _toy_dataset()
    expected = dataset.fast.reshape(2, 25, dataset.params.K, dataset.params.J).mean(axis=-1)
    np.testing.assert_allclose(dataset.fast_average(), expected)


# Generation                                                                    

def test_generate_multiscale_shapes_and_metadata() -> None:
    dataset = generate_multiscale(
        FAST_TEST_PARAMS, n_trajectories=3, t_total=1.0, t_burn=0.5, sample_dt=0.05
    )
    assert dataset.kind == "multiscale"
    assert dataset.n_trajectories == 3
    assert dataset.trajectories.shape[2] == FAST_TEST_PARAMS.dim
    assert dataset.dt == pytest.approx(0.05, rel=1e-6)
    assert np.all(np.isfinite(dataset.trajectories))
    assert dataset.metadata["t_burn"] == 0.5


def test_generated_trajectories_are_distinct() -> None:
    """Independent initial conditions must not collapse onto the same orbit."""
    dataset = generate_multiscale(FAST_TEST_PARAMS, n_trajectories=3, t_total=1.0, t_burn=0.5, sample_dt=0.05)
    slow = dataset.slow
    assert np.abs(slow[0] - slow[1]).max() > 1e-3


def test_generate_singlescale_uses_only_slow_channels() -> None:
    params = L96Params(K=5)
    dataset = generate_singlescale(
        params,
        closure=lambda x: np.zeros_like(x),
        n_trajectories=2,
        t_total=1.0,
        t_burn=1.0,
        sample_dt=0.05,
        closure_name="zero",
    )
    assert dataset.kind == "singlescale"
    assert dataset.trajectories.shape[2] == params.K
    assert dataset.metadata["closure"] == "zero"


# Diagnostics                                                                   

def test_dominant_period_recovers_a_known_sinusoid() -> None:
    dt, period = 0.01, 2.5
    t = np.arange(0, 200, dt)
    signal = np.sin(2 * np.pi * t / period)[:, None]
    assert dominant_period(signal, dt) == pytest.approx(period, rel=0.02)


def test_power_spectrum_excludes_dc_and_peaks_at_the_signal_frequency() -> None:
    dt = 0.01
    t = np.arange(0, 100, dt)
    signal = 5.0 + np.sin(2 * np.pi * t / 2.0)  # large DC offset
    freqs, power = power_spectrum(signal, dt)
    assert freqs[0] > 0
    assert 1.0 / freqs[int(np.argmax(power))] == pytest.approx(2.0, rel=0.02)


def test_resampling_hits_the_target_sampling_density() -> None:
    dt, period = 0.001, 1.0
    t = np.arange(0, 100, dt)
    signal = np.sin(2 * np.pi * t / period)[:, None]

    resampled, new_dt = resample_to_points_per_period(signal, dt)
    assert period / new_dt == pytest.approx(PANDA_POINTS_PER_PERIOD, rel=1e-6)
    assert dominant_period(resampled, new_dt) == pytest.approx(period, rel=0.05)


def test_subsampling_is_quantised_but_close() -> None:
    dt, period = 0.001, 1.0
    t = np.arange(0, 100, dt)
    signal = np.sin(2 * np.pi * t / period)[:, None]

    subsampled, new_dt, stride = subsample_for_points_per_period(signal, dt)
    assert stride == 10  # 1000 pts/period -> ~102 requires stride 10
    assert subsampled.shape[0] == signal.shape[0] // stride
    assert new_dt == pytest.approx(dt * stride)


def test_both_period_estimators_agree_on_a_pure_tone() -> None:
    dt, period = 0.01, 2.5
    t = np.arange(0, 200, dt)
    signal = np.sin(2 * np.pi * t / period)[:, None]
    assert dominant_period(signal, dt, method="peak") == pytest.approx(period, rel=0.02)
    assert dominant_period(signal, dt, method="centroid") == pytest.approx(period, rel=0.02)


def test_centroid_is_robust_to_a_dominant_harmonic() -> None:
    """The largest peak can sit on a harmonic; the centroid should not chase it.

    This is exactly the failure seen on real chaotic systems: 
    the peak estimator returned 51 samples/period for the Lorenz attractor where dysts generated at
    102, because the second harmonic carried more power than the fundamental.
    """
    dt, period = 0.01, 4.0
    t = np.arange(0, 400, dt)
    fundamental = np.sin(2 * np.pi * t / period)
    harmonic = 2.0 * np.sin(4 * np.pi * t / period)  # twice the frequency, twice the amplitude
    signal = (fundamental + harmonic)[:, None]

    assert dominant_period(signal, dt, method="peak") == pytest.approx(period / 2, rel=0.05)
    centroid = dominant_period(signal, dt, method="centroid")
    assert period / 2 < centroid < period, f"centroid {centroid} should lie between the two"


def test_dominant_period_rejects_unknown_methods_and_short_records() -> None:
    signal = np.sin(np.arange(0, 100, 0.01))[:, None]
    with pytest.raises(ValueError, match="unknown method"):
        dominant_period(signal, 0.01, method="bogus")
    with pytest.raises(ValueError, match="too short"):
        dominant_period(np.sin(np.arange(8.0))[:, None], 1.0)


def test_points_per_period_is_the_density() -> None:
    dt, period = 0.005, 1.0
    t = np.arange(0, 200, dt)
    signal = np.sin(2 * np.pi * t / period)[:, None]
    assert points_per_period(signal, dt) == pytest.approx(period / dt, rel=0.02)


def test_lyapunov_exponent_recovers_a_known_linear_growth_rate() -> None:
    """For dx/dt = a x the maximal exponent is exactly a."""
    rate = 0.7
    def linear(t, z):  # noqa: ARG001
        return rate * z

    exponent, running = max_lyapunov_exponent(linear, np.ones(3), dt=1e-3, t_renorm=0.05, n_renorm=60, n_discard=5)
    assert exponent == pytest.approx(rate, rel=1e-3)
    assert running.shape == (55,)
    assert lyapunov_time(exponent) == pytest.approx(1 / rate, rel=1e-3)


def test_lyapunov_time_rejects_non_chaotic_input() -> None:
    with pytest.raises(ValueError, match="positive"):
        lyapunov_time(-0.1)


@pytest.mark.slow
def test_multiscale_system_is_chaotic() -> None:
    """The default parameters must actually sit in the chaotic regime."""
    params = L96Params()
    system = L96Multiscale(params)
    from msdyn.data.generate import burn_in
    from msdyn.systems.integrate import suggested_dt

    dt = suggested_dt(params.eps)
    state = burn_in(system, system.default_initial_state(seed=0), t_burn=5.0, dt=dt)
    # t_renorm must be short compared with the *fast* Lyapunov time (~eps).
    exponent, _ = max_lyapunov_exponent(system, state, dt=dt, t_renorm=5 * params.eps, n_renorm=120, n_discard=20)
    assert exponent > 0, f"expected a positive Lyapunov exponent, got {exponent}"


# Correlation dimension                                                         

def test_correlation_dimension_is_monotonic_but_compressive() -> None:
    """Pins the calibration documented in `msdyn.diagnostics.dimension`.

    The paper's estimator is monotonic in true dimension but understates it
    above D2 ~ 2, so raw values are comparable to their Table 2 yet are not absolute fractal dimensions. 
    If this drifts, every dimension comparison in docs/PLAN.md needs rereading.
    """
    from msdyn.diagnostics import correlation_dimension

    rng = np.random.default_rng(0)
    estimates = [correlation_dimension(np.column_stack([rng.random(3000) for _ in range(d)] + [np.zeros(3000)] * (8 - d)),standardize=False,) for d in (1, 2, 3, 4, 5, 6)]
    assert all(a < b for a, b in zip(estimates, estimates[1:], strict=False)), estimates
    # Near-unbiased at d = 2, where PANDA's corpus lives.
    assert estimates[1] == pytest.approx(2.0, abs=0.25)
    # Compressive further up: a true 6 must not read anywhere near 6.
    assert estimates[5] < 4.5, estimates


def test_correlation_dimension_rejects_degenerate_input() -> None:
    from msdyn.diagnostics import correlation_dimension
    with pytest.raises(ValueError, match="expected"):
        correlation_dimension(np.zeros(500))
    with pytest.raises(ValueError, match="at least 100"):
        correlation_dimension(np.zeros((10, 3)))


def test_lorenz96_slow_variables_sit_above_pandas_training_corpus() -> None:
    """The measurement behind the dimension hypothesis in docs/PLAN.md.

    PANDA's paper reports a Grassberger-Procaccia dimension of 2.09 +/- 0.27 for its 129 founder systems (Table 2). 
    Our implementation of their estimator returns 2.12 +/- 0.20 on ten of those systems, so the comparison is sound.
    and Lorenz '96 slow variables at K = 9 land far above that range.
    """
    from msdyn.diagnostics import PANDA_BASE_SYSTEMS_GP_DIM, correlation_dimension
    from msdyn.systems.integrate import integrate_rk4
    from msdyn.systems.l96 import L96Params, L96SingleScale

    params = L96Params()
    system = L96SingleScale(params, closure=lambda x: 0.4 * x)
    rng = np.random.default_rng(0)
    state = burn_in(system, rng.random(params.K) * 15 - 5, t_burn=20.0, dt=1e-3)
    _, traj = integrate_rk4(system, state, t_end=60.0, dt=1e-3, sample_every=15)

    dimension = correlation_dimension(traj)
    mean, std = PANDA_BASE_SYSTEMS_GP_DIM
    assert dimension > mean + 3 * std, (f"expected Lorenz '96 (K=9) well outside PANDA's corpus, got D2={dimension:.2f} against {mean} +/- {std}")
