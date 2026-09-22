"""Closure fitting: recovery of a known ``m``, and the reference GP settings."""

from __future__ import annotations
import numpy as np
import pytest
from msdyn.closures import (ClosurePairs,GaussianProcessClosure,LinearClosure,PolynomialClosure,gather_closure_pairs,)
from msdyn.systems.l96 import L96Params

@pytest.fixture
def cubic_pairs() -> tuple[ClosurePairs, np.ndarray]:
    """Noisy samples of a known cubic -- the shape the Wilks closure assumes."""
    truth = np.array([0.3, -0.5, 0.02, -0.001])
    rng = np.random.default_rng(0)
    x = rng.uniform(-8.0, 12.0, size=20_000)
    ybar = np.polynomial.polynomial.polyval(x, truth) + rng.normal(0.0, 0.5, size=x.size)
    return ClosurePairs(x, ybar), truth


def test_polynomial_closure_recovers_known_coefficients(cubic_pairs) -> None:
    pairs, truth = cubic_pairs
    closure = PolynomialClosure(degree=3).fit(pairs)
    np.testing.assert_allclose(closure.coefficients, truth, atol=0.05)
    assert closure.residual_std == pytest.approx(0.5, abs=0.02)


def test_polynomial_closure_preserves_shape(cubic_pairs) -> None:
    pairs, _ = cubic_pairs
    closure = PolynomialClosure(degree=3).fit(pairs)
    for shape in [(9,), (4, 9), (3, 4, 9)]:
        assert closure(np.zeros(shape)).shape == shape


def test_unfitted_closure_raises(cubic_pairs) -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        PolynomialClosure()(np.zeros(3))
    with pytest.raises(RuntimeError, match="not fitted"):
        GaussianProcessClosure()(np.zeros(3))


def test_linear_closure_reproduces_g0_predictor() -> None:
    """``LinearClosure(slope=h_y)`` is the analytic G0 predictor, unfitted."""
    closure = LinearClosure(slope=1.0)
    x = np.array([-2.0, 0.0, 3.5])
    np.testing.assert_allclose(closure(x), x)


def test_gp_closure_tracks_the_conditional_mean(cubic_pairs) -> None:
    pairs, truth = cubic_pairs
    closure = GaussianProcessClosure(max_pairs=300, n_restarts=2, seed=0).fit(pairs)
    probe = np.linspace(-6.0, 10.0, 40)
    expected = np.polynomial.polynomial.polyval(probe, truth)
    # alpha=1 is a strong nugget, so the GP is a smoother, not an interpolant.
    assert np.mean(np.abs(closure(probe) - expected)) < 0.5


def test_gp_closure_reports_predictive_uncertainty(cubic_pairs) -> None:
    pairs, _ = cubic_pairs
    closure = GaussianProcessClosure(max_pairs=200, n_restarts=2, seed=0).fit(pairs)
    interior = np.array([0.0, 2.0])
    far_outside = np.array([-60.0, 60.0])
    _, std_in = closure.predict_with_std(interior)
    _, std_out = closure.predict_with_std(far_outside)
    assert np.all(std_out > std_in), "GP must be less certain far from the training data"


def test_subsample_is_a_noop_when_smaller_than_requested() -> None:
    pairs = ClosurePairs(np.arange(10.0), np.arange(10.0))
    assert pairs.subsample(50) is pairs
    assert len(pairs.subsample(4, seed=0)) == 4


def test_gather_closure_pairs_pools_over_time_and_ring() -> None:
    params = L96Params(K=3, J=4)
    n_times = 7
    traj = np.zeros((n_times, params.dim))
    traj[:, : params.K] = 1.0
    traj[:, params.K :] = 2.0

    pairs = gather_closure_pairs(traj, params)
    assert len(pairs) == n_times * params.K
    np.testing.assert_allclose(pairs.x, 1.0)
    np.testing.assert_allclose(pairs.ybar, 2.0)


def test_gather_closure_pairs_rejects_wrong_shape() -> None:
    params = L96Params(K=3, J=4)
    with pytest.raises(ValueError, match="expected trajectory"):
        gather_closure_pairs(np.zeros((10, 5)), params)


# Tabulated closures                                                            

def test_tabulated_closure_matches_its_source_inside_the_grid(cubic_pairs) -> None:
    from msdyn.closures import TabulatedClosure

    pairs, _ = cubic_pairs
    source = PolynomialClosure(degree=3).fit(pairs)
    table = TabulatedClosure.from_pairs(source, pairs, n_grid=4001)

    probe = np.linspace(-7.0, 11.0, 500)
    np.testing.assert_allclose(table(probe), source(probe), atol=1e-4)


def test_tabulated_closure_extrapolates_linearly_rather_than_reverting(cubic_pairs) -> None:
    """A GP would revert to its zero prior far from data; the table must not."""
    from msdyn.closures import TabulatedClosure

    pairs, _ = cubic_pairs
    gp = GaussianProcessClosure(max_pairs=200, n_restarts=2, seed=0).fit(pairs)
    table = TabulatedClosure.from_pairs(gp, pairs, n_grid=501)

    far = np.array([table.x_max + 5.0])
    edge = np.array([table.x_max])
    step = np.array([table.x_max + 10.0])

    # Linear: the increment doubles when the distance doubles.
    first = table(far) - table(edge)
    second = table(step) - table(edge)
    assert second == pytest.approx(2 * first, rel=1e-6)
    assert abs(gp(far)[0]) < abs(table(far)[0]), "GP should have reverted toward its prior"


def test_tabulated_closure_preserves_shape_and_reports_coverage(cubic_pairs) -> None:
    from msdyn.closures import TabulatedClosure

    pairs, _ = cubic_pairs
    table = TabulatedClosure.from_pairs(PolynomialClosure(degree=3).fit(pairs), pairs)

    assert table(np.zeros((4, 9))).shape == (4, 9)
    assert table.fraction_outside(pairs.x) == pytest.approx(0.0)
    assert table.fraction_outside(np.array([1e6, 0.0])) == pytest.approx(0.5)


def test_tabulated_closure_rejects_degenerate_grids(cubic_pairs) -> None:
    from msdyn.closures import TabulatedClosure

    source = PolynomialClosure(degree=1).fit(cubic_pairs[0])
    with pytest.raises(ValueError, match="must exceed"):
        TabulatedClosure(source, 1.0, 1.0)
    with pytest.raises(ValueError, match="at least 4"):
        TabulatedClosure(source, 0.0, 1.0, n_grid=2)
