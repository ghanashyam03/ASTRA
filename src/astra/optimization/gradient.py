"""L-BFGS-B gradient-based local refinement for trajectory optimization.
Uses scipy.optimize.minimize with finite-difference gradient computation.
Designed to refine a solution found by global (Bayesian) search.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import OptimizeResult, minimize

if TYPE_CHECKING:
    from astra.dsl.compiler import CompiledMission
    from astra.physics.kernel import PhysicsKernel
    from astra.state.trajectory import Trajectory

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


def refine_trajectory_jax(
    trajectory: Trajectory,
    mission: CompiledMission,
    kernel: PhysicsKernel,
    max_iter: int = 50,
    step_size: float = 0.01,
    convergence_tol: float = 1e-6,
) -> LocalRefinementResult:
    """Refine trajectory using JAX-computed exact gradients (gradient descent).
    
    Uses gradient descent with the JAX differentiable Δv function.
    Falls back to refine_trajectory_lbfgsb() if JAX is not available.
    
    Parameters
    ----------
    step_size: learning rate for gradient descent (in seconds for time vars)
    convergence_tol: stop when |gradient| < this value
    
    Returns same LocalRefinementResult as refine_trajectory_lbfgsb.
    """
    try:
        from astra.physics.differentiable import JAX_AVAILABLE, compute_dv_gradient
        if not JAX_AVAILABLE:
            logger.info("JAX not available, falling back to L-BFGS-B refinement")
            return refine_trajectory_lbfgsb(
                lambda x: _eval_traj(x, trajectory, mission, kernel),
                x0=np.array([trajectory.departure_epoch, trajectory.duration_seconds]),
                bounds=[(mission.departure_epoch_start, mission.departure_epoch_end),
                        (mission.tof_min_seconds, mission.tof_max_seconds)],
            )
    except ImportError:
        return refine_trajectory_lbfgsb(
            lambda x: _eval_traj(x, trajectory, mission, kernel),
            x0=np.array([trajectory.departure_epoch, trajectory.duration_seconds]),
            bounds=[(mission.departure_epoch_start, mission.departure_epoch_end),
                    (mission.tof_min_seconds, mission.tof_max_seconds)],
        )

    dep0 = float(trajectory.departure_epoch)
    tof0 = float(trajectory.duration_seconds)
    f0, _ = compute_dv_gradient(dep0, tof0, kernel, mission)
    
    dep, tof = dep0, tof0
    n_evals = 1
    start = time.perf_counter()
    prev_dv = f0
    
    for _ in range(max_iter):
        dv, grad_val = compute_dv_gradient(dep, tof, kernel, mission)
        n_evals += 1
        
        # Gradient clipping
        grad_norm = float(np.linalg.norm(grad_val))
        if grad_norm < convergence_tol:
            break
        grad_clipped = grad_val / max(grad_norm, 1.0)
        
        # Gradient descent step (adaptive: step_size scaled by TOF/day)
        dep_new = dep - step_size * 86400.0 * grad_clipped[0]
        tof_new = tof - step_size * 86400.0 * grad_clipped[1]
        
        # Bound projection
        dep_new = float(np.clip(dep_new,
                                mission.departure_epoch_start,
                                mission.departure_epoch_end))
        tof_new = float(np.clip(tof_new,
                                mission.tof_min_seconds,
                                mission.tof_max_seconds))
        
        dep, tof = dep_new, tof_new
        
        if abs(prev_dv - dv) < convergence_tol:
            break
        prev_dv = dv
    
    # Validate refined solution with exact Lambert
    from astra.optimization.engine import evaluate_transfer
    from astra.state.orbital_state import GM
    mu_sun = GM["SUN"]
    try:
        r1 = kernel.get_body_state(mission.origin_body, dep).position
        v1 = kernel.get_body_state(mission.origin_body, dep).velocity
        r2 = kernel.get_body_state(mission.destination_body, dep+tof).position
        v2 = kernel.get_body_state(mission.destination_body, dep+tof).velocity
        traj_validated = evaluate_transfer(
            r1, v1, r2, v2, dep, tof, mu_sun,
            origin_body=mission.origin_body.name,
            destination_body=mission.destination_body.name,
            parking_altitude_km=mission.parking_altitude_km,
            capture_altitude_km=mission.capture_altitude_km,
            capture_apoapsis_km=getattr(mission, "capture_apoapsis_km", None),
        )
        f_final = traj_validated.delta_v_total if traj_validated else dv
        converged = traj_validated is not None
    except Exception:
        f_final = dv
        converged = False
    
    return LocalRefinementResult(
        x_refined=np.array([dep, tof]),
        f_refined=f_final,
        x_initial=np.array([dep0, tof0]),
        f_initial=f0,
        improvement_km_s=max(0.0, f0 - f_final),
        n_evaluations=n_evals,
        converged=converged,
        message="JAX gradient descent",
        wall_time_s=time.perf_counter() - start,
    )


def _eval_traj(
    x: np.ndarray,
    trajectory: Trajectory,
    mission: CompiledMission,
    kernel: PhysicsKernel,
) -> float:
    """Helper for L-BFGS-B fallback: evaluate Δv at parameter point x."""
    from astra.optimization.engine import evaluate_transfer
    from astra.state.orbital_state import GM
    dep, tof = float(x[0]), float(x[1])
    try:
        r1 = kernel.get_body_state(mission.origin_body, dep).position
        v1 = kernel.get_body_state(mission.origin_body, dep).velocity
        r2 = kernel.get_body_state(mission.destination_body, dep+tof).position
        v2 = kernel.get_body_state(mission.destination_body, dep+tof).velocity
        tr = evaluate_transfer(r1, v1, r2, v2, dep, tof, GM["SUN"],
                               origin_body=mission.origin_body.name,
                               destination_body=mission.destination_body.name,
                               parking_altitude_km=mission.parking_altitude_km,
                               capture_altitude_km=mission.capture_altitude_km)
        return tr.delta_v_total if tr else 99.0
    except Exception:
        return 99.0
