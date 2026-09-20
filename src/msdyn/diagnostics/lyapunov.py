"""Maximal Lyapunov exponent by the Benettin algorithm.

Forecast horizons for chaotic systems are only comparable in Lyapunov time 1 / lambda_max. The Lorenz '96 multiscale system has two very different ones: 
the slow variables carry lambda ~ O(1) while the fast subsystem carries lambda ~ O(1/eps). Reporting a lead time of "0.5 time units" is
therefore meaningless on its own, it is a tenth of a slow Lyapunov time and roughly a hundred fast ones.

We use the finite-difference variant of Benettin's algorithm (evolve a nearby trajectory, renormalise the separation periodically, average the log growth),
which needs no Jacobian and so works unchanged for both the multiscale system and any fitted single-scale closure.
"""

from __future__ import annotations
from collections.abc import Callable
import numpy as np
from msdyn.systems.integrate import integrate_rk4

def max_lyapunov_exponent(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    z0: np.ndarray,
    dt: float,
    *,
    t_renorm: float = 0.1,
    n_renorm: int = 200,
    separation: float = 1e-8,
    n_discard: int = 10,
    seed: int | None = 0,
) -> tuple[float, np.ndarray]:
    """Estimate lambda_max by Benettin's method.
    rhs: Batch-aware right-hand side f(t, z).
    z0: Initial condition, assumed to already lie on the attractor.
    t_renorm: Time between renormalisations. Must be short enough that the separation stays in the linear regime. 
        for the multiscale system at eps = 2^-7that means t_renorm of order eps, not order 1.
    n_renorm: # of renormalisation intervals.
    n_discard: Leading intervals dropped so the perturbation can align with the leading Lyapunov vector.

    exponent: The estimate, averaged over the retained intervals.
    running: Running estimate after each retained interval
    """
    if n_renorm <= n_discard:
        raise ValueError(f"n_renorm ({n_renorm}) must exceed n_discard ({n_discard})")

    rng = np.random.default_rng(seed)
    direction = rng.normal(size=z0.shape)
    direction = direction / np.linalg.norm(direction)

    pair = np.stack([z0, z0 + separation * direction])
    log_growth = np.empty(n_renorm)

    for i in range(n_renorm):
        _, traj = integrate_rk4(rhs, pair, t_end=t_renorm, dt=dt, sample_every=max(1, int(t_renorm / dt)))
        reference, perturbed = traj[-1, 0], traj[-1, 1]
        delta = perturbed - reference
        distance = float(np.linalg.norm(delta))
        if distance == 0.0:
            raise FloatingPointError("perturbation collapsed to zero; increase `separation`")
        log_growth[i] = np.log(distance / separation)
        # Renormalise back onto the sphere of radius `separation`.
        pair = np.stack([reference, reference + (separation / distance) * delta])

    kept = log_growth[n_discard:]
    running = np.cumsum(kept) / (np.arange(1, kept.size + 1) * t_renorm)
    return float(running[-1]), running


def lyapunov_time(exponent: float) -> float:
    """1 / lambda_max the natural time unit for forecast horizons."""
    if exponent <= 0:
        raise ValueError(f"expected a positive Lyapunov exponent, got {exponent}")
    return 1.0 / exponent
