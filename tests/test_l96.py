"""Verify the vectorised Lorenz '96 against the reference Burov implementation.

The reference lives in the EnsembleKalmanMethods. 
If it is absent the comparison tests skip, the intrinsic tests below still run.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest
from msdyn.data.generate import burn_in
from msdyn.systems.integrate import (check_integrator_agreement,integrate_rk4,self_convergence_error,suggested_dt,)
from msdyn.systems.l96 import (L96Multiscale,L96Params,L96SingleScale,average_fast,slow_tendency,)

_REFERENCE_DIR = (Path(__file__).resolve().parents[2] / "external" / "EnsembleKalmanMethods" / "Lorenz96")

def _load_reference_l96m():
    if not (_REFERENCE_DIR / "L96_multiscale.py").exists():
        pytest.skip(f"reference implementation not found at {_REFERENCE_DIR}")
    if str(_REFERENCE_DIR) not in sys.path:
        sys.path.insert(0, str(_REFERENCE_DIR))
    from L96_multiscale import L96M  # type: ignore[import-not-found]
    return L96M


# Agreement with the reference implementation                                   

@pytest.mark.parametrize(
    "params",
    [
        L96Params(),
        L96Params(K=5, J=4, h_x=-0.5, h_y=0.8, F=8.0, eps=2.0**-4),
        L96Params(K=12, J=6, eps=2.0**-3),
    ],
)
def test_multiscale_rhs_matches_reference(params: L96Params) -> None:
    L96M = _load_reference_l96m()
    reference = L96M(K=params.K, J=params.J, hx=params.h_x, hy=params.h_y, F=params.F, eps=params.eps)
    system = L96Multiscale(params)

    rng = np.random.default_rng(0)
    for _ in range(5):
        z = rng.normal(0.0, 3.0, size=params.dim)
        np.testing.assert_allclose(system(0.0, z), reference(0.0, z), rtol=1e-12, atol=1e-12)


def test_average_fast_matches_reference_compute_yk() -> None:
    L96M = _load_reference_l96m()
    params = L96Params()
    reference = L96M(K=params.K, J=params.J)

    rng = np.random.default_rng(1)
    z = rng.normal(size=params.dim)
    np.testing.assert_allclose(average_fast(z[params.K :], params), reference.compute_Yk(z), rtol=1e-13)


def test_single_scale_rhs_matches_reference_regressed() -> None:
    """L96SingleScale must reproduce L96M.regressed for the same closure."""
    L96M = _load_reference_l96m()
    params = L96Params()
    reference = L96M(K=params.K, J=params.J, hx=params.h_x, hy=params.h_y, F=params.F)
    # The reference's G0 predictor is m(x) = h_y * x, applied through a stencil.
    reference.set_stencil()
    reference.set_G0_predictor()

    system = L96SingleScale(params, closure=lambda x: params.h_y * x)

    rng = np.random.default_rng(2)
    for _ in range(5):
        x = rng.normal(0.0, 3.0, size=params.K)
        np.testing.assert_allclose(system(0.0, x), reference.regressed(0.0, x), rtol=1e-12, atol=1e-12)

# Intrinsic properties                                                          #

def test_slow_tendency_is_cyclically_equivariant() -> None:
    """Rotating the slow ring rotates the tendency, the system is ring-symmetric."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=9)
    shift = 4
    np.testing.assert_allclose(
        slow_tendency(np.roll(x, shift), F=10.0),
        np.roll(slow_tendency(x, F=10.0), shift),
        rtol=1e-13,
    )


def test_fast_ring_wraps_across_slow_index() -> None:
    """y_{k,J-1} must couple to y_{k+1,0}, not back to y_{k,0}.

    Guards the single-ring vectorisation: if the fast lattice were wrapped within each slow index, 
    perturbing y_{1,0} would leave dy_{0,J-1} unchanged.
    """
    params = L96Params(K=3, J=4)
    system = L96Multiscale(params)
    z = np.zeros(params.dim)

    idx_last_of_block0 = params.K + 0 * params.J + (params.J - 1)
    idx_first_of_block1 = params.K + 1 * params.J + 0

    baseline = system(0.0, z)[idx_last_of_block0]
    perturbed = z.copy()
    perturbed[idx_first_of_block1] = 1.0
    # dy_{0,J-1} = -y_{0,J}(y_{0,J+1} - y_{0,J-2}) - y_{0,J-1}; with y_{0,J}=1 and
    # everything else zero this is 0 -- so probe the second neighbour instead.
    perturbed[idx_first_of_block1 + 1] = 2.0
    assert not np.isclose(system(0.0, perturbed)[idx_last_of_block0], baseline)


def test_equilibrium_of_uncoupled_slow_system() -> None:
    """With h_x = 0 the constant state x_k = F is a fixed point."""
    params = L96Params(h_x=0.0, F=10.0)
    system = L96Multiscale(params)
    z = np.zeros(params.dim)
    z[: params.K] = params.F
    np.testing.assert_allclose(system(0.0, z)[: params.K], 0.0, atol=1e-12)


# Integrator                                                                    

@pytest.fixture(scope="module")
def attractor_state() -> tuple[L96Params, L96Multiscale, np.ndarray]:
    """A burnt-in state of a moderately stiff system.

    Accuracy checks are meaningless off the attractor (reference initial condition has |dy/dt| ~ 1e4) 
    and slow at eps = 2^-7, so these tests use eps = 2^-4. 
    test_step_size_calibration_holds_at_production_eps covers the production value.
    """
    params = L96Params(eps=2.0**-4)
    system = L96Multiscale(params)
    z0 = system.default_initial_state(seed=4)
    state = burn_in(system, z0, t_burn=5.0, dt=suggested_dt(params.eps))
    return params, system, state


def test_rk4_agrees_with_adaptive_solver_on_short_window(attractor_state) -> None:
    """Window must stay short: the fast subsystem's Lyapunov exponent is ~1/eps."""
    params, system, state = attractor_state
    dt = suggested_dt(params.eps)
    deviation = check_integrator_agreement(system, state, t_end=0.05, dt=dt)
    assert deviation < 1e-4, f"RK4 (dt={dt:g}) deviates from adaptive solve by {deviation:g}"


def test_rk4_is_fourth_order_accurate(attractor_state) -> None:
    """Halving dt should cut the error by ~16x for a 4th-order scheme."""
    params, system, state = attractor_state
    dt = suggested_dt(params.eps)
    coarse = self_convergence_error(system, state, t_end=0.05, dt=dt)
    finer = self_convergence_error(system, state, t_end=0.05, dt=dt / 2)
    assert coarse < 1e-5, f"RK4 self-convergence error {coarse:g} too large at dt={dt:g}"
    assert finer < coarse / 8, f"order-4 convergence not observed: {coarse:g} -> {finer:g}"


@pytest.mark.slow
def test_step_size_calibration_holds_at_production_eps() -> None:
    """suggested_dt must actually resolve the fast ring at eps = 2^-7."""
    params = L96Params()
    system = L96Multiscale(params)
    dt = suggested_dt(params.eps)
    state = burn_in(system, system.default_initial_state(seed=11), t_burn=2.0, dt=dt)
    error = self_convergence_error(system, state, t_end=0.01, dt=dt)
    assert error < 1e-5, f"self-convergence error {error:g} at dt={dt:g}; recalibrate _DT_PER_EPS"


def test_ensemble_integration_matches_serial() -> None:
    """Batched integration must be identical to integrating each member alone."""
    params = L96Params(K=5, J=4, eps=2.0**-4)
    system = L96Multiscale(params)
    z0 = system.default_initial_state(seed=8, n_ensemble=3)
    dt = suggested_dt(params.eps)

    _, batched = integrate_rk4(system, z0, t_end=0.05, dt=dt, sample_every=5)
    assert batched.shape[1:] == (3, params.dim)

    for member in range(3):
        _, single = integrate_rk4(system, z0[member], t_end=0.05, dt=dt, sample_every=5)
        np.testing.assert_allclose(batched[:, member, :], single, rtol=1e-12, atol=1e-12)


def test_integrate_rk4_shapes_and_sampling() -> None:
    params = L96Params(K=4, J=2, eps=0.5)
    system = L96Multiscale(params)
    z0 = system.default_initial_state(seed=5)
    times, traj = integrate_rk4(system, z0, t_end=1.0, dt=0.01, sample_every=10)

    assert traj.shape == (times.size, params.dim)
    np.testing.assert_allclose(np.diff(times), 0.1, rtol=1e-9)
    np.testing.assert_allclose(traj[0], z0)


def test_integrate_rk4_rejects_bad_arguments() -> None:
    system = L96Multiscale(L96Params())
    z0 = system.default_initial_state(seed=6)
    with pytest.raises(ValueError):
        integrate_rk4(system, z0, t_end=1.0, dt=-0.1)
    with pytest.raises(ValueError):
        integrate_rk4(system, z0, t_end=1.0, dt=0.1, sample_every=0)
    with pytest.raises(ValueError):
        integrate_rk4(system, z0, t_end=0.01, dt=0.1)
