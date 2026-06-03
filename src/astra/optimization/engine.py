"""Main optimization engine for ASTRA trajectory optimization."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna

from astra.dsl.compiler import CompiledMission
from astra.dsl.schema import ConstraintType
from astra.physics.kernel import PhysicsKernel
from astra.physics.lambert import find_best_transfer
from astra.state.orbital_state import GM, CelestialBody, OrbitalState
from astra.state.trajectory import Maneuver, Trajectory

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

@dataclass
class OptimizationResult:
    best_trajectory: Trajectory | None
    pareto_front: list[Trajectory] = field(default_factory=list)
    all_trajectories: list[Trajectory] = field(default_factory=list)
    n_evaluations: int = 0
    n_feasible: int = 0
    wall_time_s: float = 0.0
    converged: bool = False
    error: str | None = None
    phase1_best_dv: float | None = None
    phase2_best_dv: float | None = None
    refinement_improvement_km_s: float | None = None
    refinement_evaluations: int | None = None
    optimizer_strategy: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "n_evaluations": self.n_evaluations,
            "n_feasible": self.n_feasible,
            "wall_time_s": round(self.wall_time_s, 3),
            "converged": self.converged,
            "best_trajectory": self.best_trajectory.to_dict() if self.best_trajectory else None,
            "pareto_front_size": len(self.pareto_front),
            "pareto_front": [t.to_dict() for t in self.pareto_front],
        }
        if self.optimizer_strategy is not None:
            d["optimizer_strategy"] = self.optimizer_strategy
            d["phase1_best_dv"] = self.phase1_best_dv
            d["phase2_best_dv"] = self.phase2_best_dv
            d["refinement_improvement_km_s"] = self.refinement_improvement_km_s
            d["refinement_evaluations"] = self.refinement_evaluations
        return d


def evaluate_transfer(
    r1: np.ndarray,
    v1_body: np.ndarray,
    r2: np.ndarray,
    v2_body: np.ndarray,
    departure_epoch: float,
    tof_seconds: float,
    mu_sun: float,
    use_multirev: bool = True,
    max_revs: int = 2,
    origin_body: str = "EARTH",
    destination_body: str = "MARS",
    parking_altitude_km: float = 200.0,
    capture_altitude_km: float = 300.0,
    use_soi_patching: bool = True,
) -> Trajectory | None:
    """Compute a patched-conics interplanetary transfer.

    Returns Trajectory on success, None if Lambert fails or geometry invalid.
    """
    if tof_seconds <= 0:
        return None

    try:
        sol = find_best_transfer(
            r1=r1,
            v1_body=v1_body,
            r2=r2,
            v2_body=v2_body,
            tof=tof_seconds,
            mu=mu_sun,
            max_revs=max_revs if use_multirev else 0,
        )
        v_dep = sol.v1
        v_arr = sol.v2
        n_revs = sol.n_revs
        branch = sol.branch
    except Exception:
        return None

    # Excess velocities
    v_inf_dep = v_dep - v1_body
    v_inf_arr = v2_body - v_arr

    if use_soi_patching:
        from astra.physics.maneuvers import arrival_delta_v, departure_delta_v
        dv1_mag = departure_delta_v(v_inf_dep, parking_altitude_km, origin_body)
        dv2_mag = arrival_delta_v(v_inf_arr, capture_altitude_km, destination_body)
        # Reconstruct Δv vectors in same direction scaled to SOI magnitudes
        dv1 = (v_inf_dep / max(float(np.linalg.norm(v_inf_dep)), 1e-10)) * dv1_mag
        dv2 = (v_inf_arr / max(float(np.linalg.norm(v_inf_arr)), 1e-10)) * dv2_mag
    else:
        dv1 = v_inf_dep
        dv2 = v_inf_arr

    s0 = OrbitalState(
        epoch=departure_epoch,
        position=r1.copy(),
        velocity=v_dep.copy(),
        central_body=CelestialBody.SUN,
    )
    s1 = OrbitalState(
        epoch=departure_epoch + tof_seconds,
        position=r2.copy(),
        velocity=v_arr.copy(),
        central_body=CelestialBody.SUN,
    )
    m1 = Maneuver(epoch=departure_epoch, delta_v=dv1, label="TMI")
    m2 = Maneuver(epoch=departure_epoch + tof_seconds, delta_v=dv2, label="MOI")

    return Trajectory(
        states=[s0, s1],
        maneuvers=[m1, m2],
        metadata={
            "departure_epoch": departure_epoch,
            "tof_days": tof_seconds / 86400.0,
            "dv1_km_s": float(np.linalg.norm(dv1)),
            "dv2_km_s": float(np.linalg.norm(dv2)),
            "n_revolutions": n_revs,
            "transfer_branch": branch,
            "v_inf_dep_km_s": float(np.linalg.norm(v_inf_dep)),
            "v_inf_arr_km_s": float(np.linalg.norm(v_inf_arr)),
            "c3_km2_s2": float(np.dot(v_inf_dep, v_inf_dep)),
            "parking_altitude_km": parking_altitude_km,
            "capture_altitude_km": capture_altitude_km,
        },
    )

def compute_porkchop(
    mission: CompiledMission,
    kernel: PhysicsKernel,
    n_dep: int = 150,
    n_tof: int = 150,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute porkchop grid: delta-v over departure × TOF.

    Returns
    -------
    dep_epochs : (n_dep,) array of departure J2000 epochs
    tof_days   : (n_tof,) array of TOFs in days
    dv_grid    : (n_dep, n_tof) array of total Δv in km/s; NaN if infeasible
    """
    mu_sun = GM["SUN"]
    dep_epochs = np.linspace(
        mission.departure_epoch_start,
        mission.departure_epoch_end,
        n_dep,
    )
    tof_arr = np.linspace(mission.tof_min_seconds, mission.tof_max_seconds, n_tof)
    dv_grid = np.full((n_dep, n_tof), np.nan)

    for i, dep in enumerate(dep_epochs):
        try:
            r1 = kernel.get_body_state(mission.origin_body, dep).position
            v1 = kernel.get_body_state(mission.origin_body, dep).velocity
        except Exception:
            continue
        for j, tof in enumerate(tof_arr):
            arr = dep + tof
            try:
                r2 = kernel.get_body_state(mission.destination_body, arr).position
                v2 = kernel.get_body_state(mission.destination_body, arr).velocity
            except Exception:
                continue
            traj = evaluate_transfer(
                r1, v1, r2, v2, dep, tof, mu_sun,
                origin_body=mission.origin_body.name,
                destination_body=mission.destination_body.name,
                parking_altitude_km=mission.parking_altitude_km,
                capture_altitude_km=mission.capture_altitude_km,
                use_soi_patching=True,
            )
            if traj is not None:
                dv_grid[i, j] = traj.delta_v_total

    return dep_epochs, tof_arr / 86400.0, dv_grid

def _get_hard_limits(mission: CompiledMission) -> tuple[float, float]:
    max_dv = 99.0
    max_days = 9999.0
    for c in mission.constraints:
        if c.type == ConstraintType.MAX_DELTA_V and c.hard:
            max_dv = min(max_dv, c.limit)
        elif c.type == ConstraintType.MAX_DURATION and c.hard:
            max_days = min(max_days, c.limit / 86400.0)
    return max_dv, max_days

def optimize_mission_bayesian(
    mission: CompiledMission,
    kernel: PhysicsKernel,
    n_trials: int = 2000,
    time_limit: float = 120.0,
    seed: int = 42,
) -> OptimizationResult:
    """Bayesian optimization (Optuna TPE) over departure epoch and TOF."""
    mu_sun = GM["SUN"]
    max_dv, max_days = _get_hard_limits(mission)
    start_time = time.time()
    all_trajs: list[Trajectory] = []

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        dep = trial.suggest_float(
            "departure_epoch",
            mission.departure_epoch_start,
            mission.departure_epoch_end,
        )
        tof = trial.suggest_float(
            "tof_seconds",
            mission.tof_min_seconds,
            mission.tof_max_seconds,
        )
        try:
            r1 = kernel.get_body_state(mission.origin_body, dep).position
            v1 = kernel.get_body_state(mission.origin_body, dep).velocity
            arr = dep + tof
            r2 = kernel.get_body_state(mission.destination_body, arr).position
            v2 = kernel.get_body_state(mission.destination_body, arr).velocity
        except Exception:
            return 99.0, 999.0

        traj = evaluate_transfer(
            r1, v1, r2, v2, dep, tof, mu_sun,
            origin_body=mission.origin_body.name,
            destination_body=mission.destination_body.name,
            parking_altitude_km=mission.parking_altitude_km,
            capture_altitude_km=mission.capture_altitude_km,
            use_soi_patching=True,
        )
        if traj is None:
            return 99.0, 999.0

        all_trajs.append(traj)
        dv = traj.delta_v_total
        days = traj.duration_days

        # Apply hard constraint penalties
        if dv > max_dv or days > max_days:
            return 99.0 + dv, 999.0 + days

        return dv, days

    sampler = optuna.samplers.NSGAIISampler(seed=seed)
    study = optuna.create_study(
        directions=["minimize", "minimize"],
        sampler=sampler,
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=time_limit,
    )

    wall_time = time.time() - start_time

    # Collect feasible trajectories from Pareto front
    pareto = []
    for trial in study.best_trials:
        dv_val, day_val = trial.values
        if dv_val < max_dv and day_val < max_days:
            dep = trial.params["departure_epoch"]
            tof = trial.params["tof_seconds"]
            try:
                r1 = kernel.get_body_state(mission.origin_body, dep).position
                v1 = kernel.get_body_state(mission.origin_body, dep).velocity
                r2 = kernel.get_body_state(mission.destination_body, dep + tof).position
                v2 = kernel.get_body_state(mission.destination_body, dep + tof).velocity
                traj = evaluate_transfer(
                    r1, v1, r2, v2, dep, tof, mu_sun,
                    origin_body=mission.origin_body.name,
                    destination_body=mission.destination_body.name,
                    parking_altitude_km=mission.parking_altitude_km,
                    capture_altitude_km=mission.capture_altitude_km,
                    use_soi_patching=True,
                )
                if traj and traj.is_feasible(max_dv, max_days):
                    pareto.append(traj)
            except Exception:
                pass

    # Pick best by primary objective (minimize delta_v)
    feasible = [t for t in all_trajs if t.is_feasible(max_dv, max_days)]
    best = min(feasible, key=lambda t: t.delta_v_total, default=None)
    if best is None and pareto:
        best = min(pareto, key=lambda t: t.delta_v_total)

    return OptimizationResult(
        best_trajectory=best,
        pareto_front=pareto,
        all_trajectories=feasible,
        n_evaluations=len(study.trials),
        n_feasible=len(feasible),
        wall_time_s=wall_time,
        converged=best is not None,
    )

def optimize_mission_neural_accelerated(
    mission: CompiledMission,
    kernel: PhysicsKernel,
    n_trials: int = 2000,
    time_limit: float = 120.0,
    seed: int = 42,
    pretrain_samples: int = 500,
) -> OptimizationResult:
    """Bayesian optimization with neural feasibility pre-filtering.
    1. Generate pretrain_samples with physics kernel
    2. Train FeasibilityClassifier on them
    3. Run Bayesian optimizer — skip physics eval if P(feasible) < 0.3
    4. Return OptimizationResult (same interface as optimize_mission_bayesian)
    """
    import time as time_mod

    from astra.neural.feasibility import FeasibilityClassifier
    from astra.neural.training.pipeline import generate_transfer_dataset

    start_time = time_mod.time()
    mu_sun = GM["SUN"]
    max_dv, max_days = _get_hard_limits(mission)

    # Seed global numpy random state for neural network weight initialization
    # and data shuffling determinism
    np.random.seed(seed)

    # Phase 1: generate training data
    logger.info(f"Generating {pretrain_samples} samples for neural pretraining...")
    X, dv_y, feas_y = generate_transfer_dataset(
        kernel,
        mission.origin_body,
        mission.destination_body,
        mission.departure_epoch_start,
        mission.departure_epoch_end,
        mission.tof_min_seconds,
        mission.tof_max_seconds,
        n_samples=pretrain_samples,
        seed=seed,
    )

    # Phase 2: train classifier
    clf = FeasibilityClassifier()
    clf.train_on_dataset(X, feas_y, epochs=30, batch_size=128)
    logger.info("Feasibility classifier trained.")

    # Phase 3: Bayesian optimization with neural filter
    import optuna
    all_trajs: list[Trajectory] = []
    n_skipped = 0

    def objective_accelerated(trial: optuna.Trial) -> tuple[float, float]:
        nonlocal n_skipped
        dep = trial.suggest_float("departure_epoch",
                                  mission.departure_epoch_start,
                                  mission.departure_epoch_end)
        tof = trial.suggest_float("tof_seconds",
                                  mission.tof_min_seconds,
                                  mission.tof_max_seconds)

        # Neural pre-filter (conservative threshold 0.3)
        feat = np.array([
            (dep - mission.departure_epoch_start) / max(
                mission.departure_epoch_end - mission.departure_epoch_start, 1.0),
            (tof - mission.tof_min_seconds) / max(
                mission.tof_max_seconds - mission.tof_min_seconds, 1.0),
            *[0.0] * 6,  # placeholder planet positions
        ], dtype=np.float32)

        if trial.number >= 100 and not clf.is_likely_feasible(feat):
            n_skipped += 1
            return 99.0, 999.0  # pruned — no physics call

        # Physics evaluation
        try:
            r1 = kernel.get_body_state(mission.origin_body, dep).position
            v1 = kernel.get_body_state(mission.origin_body, dep).velocity
            arr = dep + tof
            r2 = kernel.get_body_state(mission.destination_body, arr).position
            v2 = kernel.get_body_state(mission.destination_body, arr).velocity
        except Exception:
            return 99.0, 999.0

        traj = evaluate_transfer(
            r1, v1, r2, v2, dep, tof, mu_sun,
            origin_body=mission.origin_body.name,
            destination_body=mission.destination_body.name,
            parking_altitude_km=mission.parking_altitude_km,
            capture_altitude_km=mission.capture_altitude_km,
            use_soi_patching=True,
        )
        if traj is None:
            # Online update: physics says infeasible
            clf.update(feat, 0.0)
            return 99.0, 999.0

        all_trajs.append(traj)
        clf.update(feat, 1.0 if (traj.delta_v_total < 15.0 and traj.duration_days < 350.0) else 0.0)

        dv = traj.delta_v_total
        days = traj.duration_days
        if dv > max_dv or days > max_days:
            return 99.0 + dv, 999.0 + days
        return dv, days

    sampler = optuna.samplers.NSGAIISampler(seed=seed)
    study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler)
    study.optimize(objective_accelerated, n_trials=n_trials, timeout=time_limit)

    wall_time = time_mod.time() - start_time
    logger.info(f"Neural filter skipped {n_skipped}/{n_trials} evaluations "
                f"({100*n_skipped/max(n_trials,1):.1f}% saved).")

    pareto = []
    for trial in study.best_trials:
        dv_val, day_val = trial.values
        if dv_val < max_dv and day_val < max_days:
            dep = trial.params["departure_epoch"]
            tof = trial.params["tof_seconds"]
            try:
                r1 = kernel.get_body_state(mission.origin_body, dep).position
                v1 = kernel.get_body_state(mission.origin_body, dep).velocity
                r2 = kernel.get_body_state(mission.destination_body, dep + tof).position
                v2 = kernel.get_body_state(mission.destination_body, dep + tof).velocity
                traj = evaluate_transfer(
                    r1, v1, r2, v2, dep, tof, mu_sun,
                    origin_body=mission.origin_body.name,
                    destination_body=mission.destination_body.name,
                    parking_altitude_km=mission.parking_altitude_km,
                    capture_altitude_km=mission.capture_altitude_km,
                    use_soi_patching=True,
                )
                if traj and traj.is_feasible(max_dv, max_days):
                    pareto.append(traj)
            except Exception:
                pass

    feasible = [t for t in all_trajs if t.is_feasible(max_dv, max_days)]
    best = min(feasible, key=lambda t: t.delta_v_total, default=None)
    if best is None and pareto:
        best = min(pareto, key=lambda t: t.delta_v_total)

    return OptimizationResult(
        best_trajectory=best,
        pareto_front=pareto,
        all_trajectories=feasible,
        n_evaluations=len(study.trials),
        n_feasible=len(feasible),
        wall_time_s=wall_time,
        converged=best is not None,
    )


def optimize_mission_hybrid(
    mission: CompiledMission,
    kernel: PhysicsKernel,
    n_trials_bayesian: int = 1500,
    n_refine_top_k: int = 5,     # refine top-K Bayesian solutions
    time_limit: float = 150.0,
    seed: int = 42,
) -> OptimizationResult:
    """Two-phase hybrid optimizer: Bayesian global + L-BFGS-B local refinement.

    Phase 1: Run Bayesian optimization for n_trials_bayesian evaluations.
    Phase 2: Take the top-K feasible solutions from Phase 1 and run
             L-BFGS-B local refinement from each starting point.
             Return the best refined solution.

    The total Δv of the hybrid result must be ≤ Phase 1 best Δv.
    """
    import time as t

    from astra.optimization.gradient import refine_trajectory_lbfgsb
    start = t.time()

    # Phase 1: Bayesian global search
    phase1_limit = time_limit * 0.75
    result_p1 = optimize_mission_bayesian(
        mission, kernel,
        n_trials=n_trials_bayesian,
        time_limit=phase1_limit,
        seed=seed,
    )

    if not result_p1.converged or not result_p1.all_trajectories:
        return result_p1

    max_dv, max_days = _get_hard_limits(mission)
    mu_sun = GM["SUN"]

    # Build single-objective sorted list of feasible trajectories from Phase 1
    feasible = sorted(
        result_p1.all_trajectories,
        key=lambda t: t.delta_v_total,
    )[:n_refine_top_k]

    # Phase 2: L-BFGS-B refinement from each top-K starting point
    bounds_dep = (mission.departure_epoch_start, mission.departure_epoch_end)
    bounds_tof = (mission.tof_min_seconds, mission.tof_max_seconds)
    bounds = [bounds_dep, bounds_tof]

    all_refined: list[Trajectory] = list(result_p1.all_trajectories)
    best_refined: Trajectory | None = result_p1.best_trajectory

    total_refinement_evals = 0
    phase1_best_dv = result_p1.best_trajectory.delta_v_total if result_p1.best_trajectory else None

    # Temporarily disable ephemeris cache for continuous gradient precision
    old_cache = kernel.ephemeris.cache
    kernel.ephemeris.cache = None

    try:
        for traj in feasible:
            if t.time() - start > time_limit - 10:
                break
            x0 = np.array([traj.departure_epoch, traj.duration_seconds])

            def obj(x: np.ndarray) -> float:
                dep, tof = float(x[0]), float(x[1])
                if tof <= 0:
                    return 99.0
                try:
                    r1 = kernel.get_body_state(mission.origin_body, dep).position
                    v1 = kernel.get_body_state(mission.origin_body, dep).velocity
                    arr = dep + tof
                    r2 = kernel.get_body_state(mission.destination_body, arr).position
                    v2 = kernel.get_body_state(mission.destination_body, arr).velocity
                except Exception:
                    return 99.0
                t_new = evaluate_transfer(
                    r1, v1, r2, v2, dep, tof, mu_sun,
                    origin_body=mission.origin_body.name,
                    destination_body=mission.destination_body.name,
                    parking_altitude_km=mission.parking_altitude_km,
                    capture_altitude_km=mission.capture_altitude_km,
                    use_soi_patching=True,
                  )
                if t_new is None:
                    return 99.0
                if not t_new.is_feasible(max_dv, max_days):
                    return 99.0 + t_new.delta_v_total
                return t_new.delta_v_total

            ref_result = refine_trajectory_lbfgsb(
                obj, x0, bounds,
                eps=1.0,
                gtol=1e-12,
                ftol=1e-15,
            )
            total_refinement_evals += ref_result.n_evaluations

            if ref_result.converged and ref_result.f_refined < max_dv:
                dep_r, tof_r = float(ref_result.x_refined[0]), float(ref_result.x_refined[1])
                try:
                    r1 = kernel.get_body_state(mission.origin_body, dep_r).position
                    v1 = kernel.get_body_state(mission.origin_body, dep_r).velocity
                    r2 = kernel.get_body_state(mission.destination_body, dep_r + tof_r).position
                    v2 = kernel.get_body_state(mission.destination_body, dep_r + tof_r).velocity
                    t_new = evaluate_transfer(
                        r1, v1, r2, v2, dep_r, tof_r, mu_sun,
                        origin_body=mission.origin_body.name,
                        destination_body=mission.destination_body.name,
                        parking_altitude_km=mission.parking_altitude_km,
                        capture_altitude_km=mission.capture_altitude_km,
                        use_soi_patching=True,
                    )
                    if t_new and t_new.is_feasible(max_dv, max_days):
                        all_refined.append(t_new)
                        if (best_refined is None or
                                t_new.delta_v_total < best_refined.delta_v_total):
                            best_refined = t_new
                except Exception:
                    pass
    finally:
        # Restore cache for subsequent queries
        kernel.ephemeris.cache = old_cache

    phase2_best_dv = best_refined.delta_v_total if best_refined else None
    improvement = (
        phase1_best_dv - phase2_best_dv
        if phase1_best_dv is not None and phase2_best_dv is not None
        else 0.0
    )

    if best_refined is not None:
        best_refined.metadata["phase1_best_dv"] = phase1_best_dv
        best_refined.metadata["phase2_best_dv"] = phase2_best_dv
        best_refined.metadata["refinement_improvement_km_s"] = improvement
        best_refined.metadata["refinement_evaluations"] = total_refinement_evals
        best_refined.metadata["optimizer_strategy"] = "hybrid"

    return OptimizationResult(
        best_trajectory=best_refined,
        pareto_front=result_p1.pareto_front,
        all_trajectories=all_refined,
        n_evaluations=result_p1.n_evaluations + total_refinement_evals,
        n_feasible=len([t for t in all_refined if t.is_feasible(max_dv, max_days)]),
        wall_time_s=t.time() - start,
        converged=best_refined is not None,
        phase1_best_dv=phase1_best_dv,
        phase2_best_dv=phase2_best_dv,
        refinement_improvement_km_s=improvement,
        refinement_evaluations=total_refinement_evals,
        optimizer_strategy="hybrid",
    )


