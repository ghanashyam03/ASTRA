"""L-BFGS-B gradient-based local refinement for trajectory optimization.
Uses scipy.optimize.minimize with finite-difference gradient computation.
Designed to refine a solution found by global (Bayesian) search.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import OptimizeResult, minimize

logger = logging.getLogger(__name__)

@dataclass
class LocalRefinementResult:
    x_refined: np.ndarray     # [dep_epoch, tof_seconds]
    f_refined: float          # refined objective value (Δv in km/s)
    x_initial: np.ndarray     # starting point from Bayesian search
    f_initial: float          # initial objective value
    improvement_km_s: float   # f_initial - f_refined (positive = improved)
    n_evaluations: int
    converged: bool
    message: str
    wall_time_s: float

def refine_trajectory_lbfgsb(
    objective_fn: Callable[[np.ndarray], float],
    x0: np.ndarray,
    bounds: list[tuple[float, float]],
    eps: float = 1e-4,          # finite-difference step size (seconds for epochs, seconds for TOF)
    ftol: float = 1e-9,
    gtol: float = 1e-6,
    max_iter: int = 200,
) -> LocalRefinementResult:
    """Run L-BFGS-B local refinement from x0 within bounds.

    Parameters
    ----------
    objective_fn : Callable[[x], float]
        Must accept x = [dep_epoch, tof_seconds] and return total Δv in km/s.
        Returns 99.0 (penalty) for infeasible points.
    x0 : starting point [dep_epoch, tof_seconds]
    bounds : [(dep_min, dep_max), (tof_min, tof_max)]
    eps : finite-difference step in seconds — 1e-4 s ≈ 0.1 ms (much smaller than
          epoch quantization, so does NOT hit cache — intentional for gradient accuracy)

    Returns
    -------
    LocalRefinementResult
    """
    f0 = objective_fn(x0)
    n_evals = [1]
    start = time.perf_counter()

    def wrapped(x: np.ndarray) -> float:
        val = objective_fn(x)
        n_evals[0] += 1
        return val

    try:
        result: OptimizeResult = minimize(
            wrapped,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "ftol": ftol,
                "gtol": gtol,
                "maxiter": max_iter,
                "eps": eps,   # finite-diff step size
            },
        )
        x_ref = result.x
        f_ref = float(result.fun)
        converged = bool(result.success) and f_ref < 90.0  # 90 = infeasible sentinel
        msg = result.message
    except Exception as e:
        x_ref = x0.copy()
        f_ref = f0
        converged = False
        msg = str(e)

    elapsed = time.perf_counter() - start
    improvement = max(0.0, f0 - f_ref)

    if improvement > 0.001:
        logger.info(f"Gradient refinement improved Δv by {improvement:.4f} km/s "
                    f"in {n_evals[0]} evals ({elapsed:.2f}s)")
    else:
        logger.info(f"Gradient refinement: no improvement (Δv {f_ref:.4f} km/s, {elapsed:.2f}s)")

    return LocalRefinementResult(
        x_refined=x_ref,
        f_refined=f_ref,
        x_initial=x0,
        f_initial=f0,
        improvement_km_s=improvement,
        n_evaluations=n_evals[0],
        converged=converged,
        message=msg,
        wall_time_s=elapsed,
    )
