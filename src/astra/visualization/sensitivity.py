"""Trajectory sensitivity analysis via finite-difference perturbation.
Answers: 'How much does Δv change if TOF changes by ±1 day?'
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from astra.dsl.compiler import CompiledMission
    from astra.physics.kernel import PhysicsKernel
    from astra.state.trajectory import Trajectory


@dataclass
class SensitivityResult:
    parameter_name: str
    baseline_value: float
    baseline_dv: float
    perturbation_step: float
    dv_plus: float          # f(x + h)
    dv_minus: float         # f(x - h)
    central_gradient: float # (f(x+h) - f(x-h)) / (2h)
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
                r1, v1, r2, v2, dep0, tof, mu_sun,
                origin_body=mission.origin_body.name,
                destination_body=mission.destination_body.name,
                parking_altitude_km=mission.parking_altitude_km,
                capture_altitude_km=mission.capture_altitude_km,
                use_soi_patching=True,
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
                r1, v1, r2, v2, dep, tof0, mu_sun,
                origin_body=mission.origin_body.name,
                destination_body=mission.destination_body.name,
                parking_altitude_km=mission.parking_altitude_km,
                capture_altitude_km=mission.capture_altitude_km,
                use_soi_patching=True,
            )
            return t.delta_v_total if t else 99.0
        except Exception:
            return 99.0

    step_day = 86400.0  # 1-day step
    sensitivities = [
        compute_sensitivity(
            dv_vs_tof, tof0, step_day, "tof",
            "km/s per day of TOF change"
        ),
        compute_sensitivity(
            dv_vs_dep, dep0, step_day, "departure_epoch",
            "km/s per day of departure shift"
        ),
    ]

    return TrajectoryParameterSensitivity(
        mission_id=getattr(mission, "mission_id", "unknown"),
        sensitivities=sensitivities,
    )
