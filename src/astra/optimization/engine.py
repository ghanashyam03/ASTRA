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
from astra.physics.lambert import lambert_izzo
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_evaluations": self.n_evaluations,
            "n_feasible": self.n_feasible,
            "wall_time_s": round(self.wall_time_s, 3),
            "converged": self.converged,
            "best_trajectory": self.best_trajectory.to_dict() if self.best_trajectory else None,
            "pareto_front_size": len(self.pareto_front),
            "pareto_front": [t.to_dict() for t in self.pareto_front],
        }

def evaluate_transfer(
    r1: np.ndarray,
    v1_body: np.ndarray,
    r2: np.ndarray,
    v2_body: np.ndarray,
    departure_epoch: float,
    tof_seconds: float,
    mu_sun: float,
) -> Trajectory | None:
    """Compute a patched-conics interplanetary transfer.

    Returns Trajectory on success, None if Lambert fails or geometry invalid.
    """
    if tof_seconds <= 0:
        return None

    try:
        v_dep, v_arr, converged = lambert_izzo(r1, r2, tof_seconds, mu_sun)
    except Exception:
        return None

    if not converged:
        return None

    # Departure delta-v: difference from body velocity
    dv1 = v_dep - v1_body
    dv2 = v2_body - v_arr

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
            traj = evaluate_transfer(r1, v1, r2, v2, dep, tof, mu_sun)
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

        traj = evaluate_transfer(r1, v1, r2, v2, dep, tof, mu_sun)
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
                traj = evaluate_transfer(r1, v1, r2, v2, dep, tof, mu_sun)
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

    # Seed global numpy random state for neural network weight initialization and data shuffling determinism
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

        traj = evaluate_transfer(r1, v1, r2, v2, dep, tof, mu_sun)
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
                traj = evaluate_transfer(r1, v1, r2, v2, dep, tof, mu_sun)
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

