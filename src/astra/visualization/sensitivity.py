"""Trajectory sensitivity analysis via central finite differences / finite-difference perturbation.
Answers: 'If TOF changes by ±1 day, how much does total Δv change?'
All physics calls go through the existing evaluate_transfer() function.
No new physics here — only perturbation and finite differences.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from astra.dsl.compiler import CompiledMission
    from astra.physics.kernel import PhysicsKernel
    from astra.state.trajectory import Trajectory

# --- NEW CLASSES & FUNCTIONS (PART B) ---


@dataclass
class SensitivityPoint:
    parameter_name: str
    baseline_value: float
    perturbation_step: float
    units: str
    baseline_dv: float
    dv_plus: float  # f(x + h)
    dv_minus: float  # f(x - h)
    gradient: float  # central difference: (f_plus - f_minus) / (2*h)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter_name,
            "units": self.units,
            "baseline": round(self.baseline_value, 6),
            "perturbation_step": round(self.perturbation_step, 6),
            "baseline_dv_km_s": round(self.baseline_dv, 6),
            "dv_plus": round(self.dv_plus, 6),
            "dv_minus": round(self.dv_minus, 6),
            "gradient_km_s_per_unit": round(self.gradient, 8),
        }


@dataclass
class TrajectorySensitivity:
    mission_id: str
    points: list[SensitivityPoint]
    wall_time_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "wall_time_s": round(self.wall_time_s, 3),
            "sensitivities": [p.to_dict() for p in self.points],
        }


def central_difference(
    f: Callable[[float], float],
    x0: float,
    h: float,
) -> tuple[float, float, float]:
    """Compute f(x0), f(x0+h), f(x0-h). Returns (f0, f_plus, f_minus)."""
    return f(x0), f(x0 + h), f(x0 - h)


def analyze_sensitivity(
    trajectory: Trajectory,
    mission: CompiledMission,
    kernel: PhysicsKernel,
    dep_step_days: float = 1.0,
    tof_step_days: float = 1.0,
) -> TrajectorySensitivity:
    """Compute sensitivity of optimal trajectory Δv with respect to:
    1. Departure epoch (±dep_step_days)
    2. Time of flight (±tof_step_days)

    Uses central finite differences at the optimal point. All evaluations
    call evaluate_transfer() with full SOI patching — no physics shortcuts.
    Returns 99.0 for infeasible perturbations (same convention as optimizer).
    """
    import time as t_mod

    from astra.optimization.engine import evaluate_transfer
    from astra.state.orbital_state import GM

    start = t_mod.time()
    mu_sun = GM["SUN"]
    dep0 = float(trajectory.departure_epoch)
    tof0 = float(trajectory.duration_seconds)

    def dv_at_dep(dep: float) -> float:
        arr = dep + tof0
        try:
            r1 = kernel.get_body_state(mission.origin_body, dep).position
            v1 = kernel.get_body_state(mission.origin_body, dep).velocity
            r2 = kernel.get_body_state(mission.destination_body, arr).position
            v2 = kernel.get_body_state(mission.destination_body, arr).velocity
            tr = evaluate_transfer(
                r1,
                v1,
                r2,
                v2,
                dep,
                tof0,
                mu_sun,
                origin_body=mission.origin_body.name,
                destination_body=mission.destination_body.name,
                parking_altitude_km=mission.parking_altitude_km,
                capture_altitude_km=mission.capture_altitude_km,
            )
            return tr.delta_v_total if tr is not None else 99.0
        except Exception:
            return 99.0

    def dv_at_tof(tof: float) -> float:
        arr = dep0 + tof
        try:
            r1 = kernel.get_body_state(mission.origin_body, dep0).position
            v1 = kernel.get_body_state(mission.origin_body, dep0).velocity
            r2 = kernel.get_body_state(mission.destination_body, arr).position
            v2 = kernel.get_body_state(mission.destination_body, arr).velocity
            tr = evaluate_transfer(
                r1,
                v1,
                r2,
                v2,
                dep0,
                tof,
                mu_sun,
                origin_body=mission.origin_body.name,
                destination_body=mission.destination_body.name,
                parking_altitude_km=mission.parking_altitude_km,
                capture_altitude_km=mission.capture_altitude_km,
            )
            return tr.delta_v_total if tr is not None else 99.0
        except Exception:
            return 99.0

    dep_step_s = dep_step_days * 86400.0
    tof_step_s = tof_step_days * 86400.0

    f0_dep, fp_dep, fm_dep = central_difference(dv_at_dep, dep0, dep_step_s)
    grad_dep = (fp_dep - fm_dep) / (2.0 * dep_step_s / 86400.0)

    f0_tof, fp_tof, fm_tof = central_difference(dv_at_tof, tof0, tof_step_s)
    grad_tof = (fp_tof - fm_tof) / (2.0 * tof_step_s / 86400.0)

    points = [
        SensitivityPoint(
            parameter_name="departure_epoch",
            baseline_value=dep0,
            perturbation_step=dep_step_s,
            units="km/s per day of departure shift",
            baseline_dv=f0_dep,
            dv_plus=fp_dep,
            dv_minus=fm_dep,
            gradient=grad_dep,
        ),
        SensitivityPoint(
            parameter_name="time_of_flight",
            baseline_value=tof0,
            perturbation_step=tof_step_s,
            units="km/s per day of TOF change",
            baseline_dv=f0_tof,
            dv_plus=fp_tof,
            dv_minus=fm_tof,
            gradient=grad_tof,
        ),
    ]

    return TrajectorySensitivity(
        mission_id=mission.mission_id,
        points=points,
        wall_time_s=t_mod.time() - start,
    )


# --- DEPRECATED/OLD CLASSES & FUNCTIONS FOR BACKWARD COMPATIBILITY ---


@dataclass
class SensitivityResult:
    parameter_name: str
    baseline_value: float
    baseline_dv: float
    perturbation_step: float
    dv_plus: float  # f(x + h)
    dv_minus: float  # f(x - h)
    central_gradient: float  # (f(x+h) - f(x-h)) / (2h)
    sensitivity_label: str  # human-readable units
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        grad_val = self.central_gradient
        return {
            "parameter": self.parameter_name,
            "baseline": round(self.baseline_value, 4),
            "baseline_dv_km_s": round(self.baseline_dv, 6),
            "dv_plus": round(self.dv_plus, 6) if self.dv_plus is not None else None,
            "dv_minus": round(self.dv_minus, 6) if self.dv_minus is not None else None,
            "gradient": round(grad_val, 8) if grad_val is not None else None,
            "units": self.sensitivity_label,
            "metadata": self.metadata,
        }


@dataclass
class TrajectoryParameterSensitivity:
    """Complete sensitivity analysis for a trajectory."""

    mission_id: str
    sensitivities: list[SensitivityResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "sensitivities": [s.to_dict() for s in self.sensitivities],
        }


def compute_sensitivity(
    objective_fn: Callable[[float], float],
    baseline: float,
    step: float,
    param_name: str,
    label: str,
) -> SensitivityResult:
    """Central finite difference for one parameter. Handles failures gracefully."""
    meta: dict[str, Any] = {}

    # Evaluate baseline
    try:
        f0 = objective_fn(baseline)
    except Exception as e:
        f0 = 99.0
        meta["baseline_error"] = str(e)

    # Evaluate baseline + step
    try:
        fp = objective_fn(baseline + step)
    except Exception as e:
        fp = 99.0
        meta["plus_error"] = str(e)

    # Evaluate baseline - step
    try:
        fm = objective_fn(baseline - step)
    except Exception as e:
        fm = 99.0
        meta["minus_error"] = str(e)

    # Robust handling of infeasible bounds
    infeasible_limit = 90.0
    if fp >= infeasible_limit or fm >= infeasible_limit or f0 >= infeasible_limit:
        grad = 0.0
        meta["infeasible"] = True
        if fp >= infeasible_limit:
            meta["plus_status"] = "infeasible"
        if fm >= infeasible_limit:
            meta["minus_status"] = "infeasible"
        if f0 >= infeasible_limit:
            meta["baseline_status"] = "infeasible"
    else:
        grad = (fp - fm) / (2.0 * step)
        meta["infeasible"] = False

    return SensitivityResult(
        parameter_name=param_name,
        baseline_value=baseline,
        baseline_dv=f0,
        perturbation_step=step,
        dv_plus=fp,
        dv_minus=fm,
        central_gradient=grad,
        sensitivity_label=label,
        metadata=meta,
    )


def analyze_trajectory_sensitivity(
    trajectory: Trajectory,
    mission: CompiledMission,
    kernel: PhysicsKernel,
) -> TrajectoryParameterSensitivity:
    """Compute sensitivity of optimal trajectory Δv to key parameters.

    Perturbs: TOF ± 1 day, departure epoch ± 1 day.
    Uses central finite differences at the optimal point.
    """
    from astra.optimization.engine import evaluate_transfer
    from astra.state.orbital_state import GM

    mu_sun = GM["SUN"]
    dep0 = float(trajectory.departure_epoch)
    tof0 = float(trajectory.duration_seconds)

    def dv_vs_tof(tof: float) -> float:
        arr = dep0 + tof
        try:
            r1 = kernel.get_body_state(mission.origin_body, dep0).position
            v1 = kernel.get_body_state(mission.origin_body, dep0).velocity
            r2 = kernel.get_body_state(mission.destination_body, arr).position
            v2 = kernel.get_body_state(mission.destination_body, arr).velocity
            t = evaluate_transfer(
                r1,
                v1,
                r2,
                v2,
                dep0,
                tof,
                mu_sun,
                origin_body=mission.origin_body.name,
                destination_body=mission.destination_body.name,
                parking_altitude_km=mission.parking_altitude_km,
                capture_altitude_km=mission.capture_altitude_km,
                use_soi_patching=True,
                capture_apoapsis_km=getattr(mission, "capture_apoapsis_km", None),
            )
            return t.delta_v_total if t else 99.0
        except Exception:
            return 99.0

    def dv_vs_dep(dep: float) -> float:
        arr = dep + tof0
        try:
            r1 = kernel.get_body_state(mission.origin_body, dep).position
            v1 = kernel.get_body_state(mission.origin_body, dep).velocity
            r2 = kernel.get_body_state(mission.destination_body, arr).position
            v2 = kernel.get_body_state(mission.destination_body, arr).velocity
            t = evaluate_transfer(
                r1,
                v1,
                r2,
                v2,
                dep,
                tof0,
                mu_sun,
                origin_body=mission.origin_body.name,
                destination_body=mission.destination_body.name,
                parking_altitude_km=mission.parking_altitude_km,
                capture_altitude_km=mission.capture_altitude_km,
                use_soi_patching=True,
                capture_apoapsis_km=getattr(mission, "capture_apoapsis_km", None),
            )
            return t.delta_v_total if t else 99.0
        except Exception:
            return 99.0

    step_day = 86400.0  # 1-day step
    sensitivities = [
        compute_sensitivity(dv_vs_tof, tof0, step_day, "tof", "km/s per day of TOF change"),
        compute_sensitivity(
            dv_vs_dep, dep0, step_day, "departure_epoch", "km/s per day of departure shift"
        ),
    ]

    return TrajectoryParameterSensitivity(
        mission_id=getattr(mission, "mission_id", "unknown"),
        sensitivities=sensitivities,
    )
