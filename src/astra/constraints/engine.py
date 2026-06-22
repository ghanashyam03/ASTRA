from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from astra.constraints.physical import (
    PhysicalConstraintResult,
    check_max_delta_v,
    check_min_periapsis,
)
from astra.constraints.propellant import (
    PropellantConstraintResult,
    check_propellant_budget,
)
from astra.constraints.temporal import (
    TemporalConstraintResult,
    check_launch_window,
    check_max_duration,
)
from astra.dsl.compiler import CompiledMission
from astra.dsl.schema import ConstraintType
from astra.state.spacecraft import Spacecraft
from astra.state.trajectory import Trajectory


@dataclass
class ConstraintViolation:
    constraint_type: str
    severity: str  # "hard" or "soft"
    message: str
    actual_value: float
    limit_value: float
    body: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint_type": self.constraint_type,
            "severity": self.severity,
            "message": self.message,
            "actual_value": float(self.actual_value),
            "limit_value": float(self.limit_value),
            "body": self.body,
        }


@dataclass
class ConstraintReport:
    all_satisfied: bool
    hard_violations: list[ConstraintViolation]
    soft_violations: list[ConstraintViolation]
    physical_results: list[PhysicalConstraintResult]
    propellant_result: PropellantConstraintResult | None
    temporal_results: list[TemporalConstraintResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "all_satisfied": self.all_satisfied,
            "hard_violations": [v.to_dict() for v in self.hard_violations],
            "soft_violations": [v.to_dict() for v in self.soft_violations],
            "physical_results": [
                {
                    "constraint_type": r.constraint_type,
                    "body": r.body,
                    "limit_km": float(r.limit_km),
                    "actual_km": float(r.actual_km),
                    "satisfied": r.satisfied,
                    "violation_km": float(r.violation_km),
                }
                for r in self.physical_results
            ],
            "propellant_result": {
                "required_dv_km_s": float(self.propellant_result.required_dv_km_s),
                "available_dv_km_s": float(self.propellant_result.available_dv_km_s),
                "required_propellant_kg": float(self.propellant_result.required_propellant_kg),
                "available_propellant_kg": float(self.propellant_result.available_propellant_kg),
                "satisfied": self.propellant_result.satisfied,
                "margin_kg": float(self.propellant_result.margin_kg),
            }
            if self.propellant_result
            else None,
            "temporal_results": [
                {
                    "constraint_type": r.constraint_type,
                    "limit_days": float(r.limit_days),
                    "actual_days": float(r.actual_days),
                    "satisfied": r.satisfied,
                    "margin_days": float(r.margin_days),
                }
                for r in self.temporal_results
            ],
        }

    @property
    def is_hard_feasible(self) -> bool:
        return len(self.hard_violations) == 0


def evaluate_all_constraints(
    trajectory: Trajectory,
    mission: CompiledMission,
    spacecraft: Spacecraft,
) -> ConstraintReport:
    """Evaluates all constraints for a given trajectory, mission, and spacecraft."""
    hard_violations: list[ConstraintViolation] = []
    soft_violations: list[ConstraintViolation] = []
    physical_results: list[PhysicalConstraintResult] = []
    temporal_results: list[TemporalConstraintResult] = []

    # Evaluate compiled constraints
    for c in mission.constraints:
        if c.type == ConstraintType.MAX_DELTA_V:
            res_dv = check_max_delta_v(trajectory, c.limit)
            physical_results.append(res_dv)
            if not res_dv.satisfied:
                violation = ConstraintViolation(
                    constraint_type="max_delta_v",
                    severity="hard" if c.hard else "soft",
                    message=(
                        f"Delta-V limit of {c.limit} km/s exceeded with "
                        f"{res_dv.actual_km:.3f} km/s"
                    ),
                    actual_value=res_dv.actual_km,
                    limit_value=c.limit,
                )
                if c.hard:
                    hard_violations.append(violation)
                else:
                    soft_violations.append(violation)

        elif c.type == ConstraintType.MAX_DURATION:
            limit_days = c.limit / 86400.0
            res_dur = check_max_duration(trajectory, limit_days)
            temporal_results.append(res_dur)
            if not res_dur.satisfied:
                violation = ConstraintViolation(
                    constraint_type="max_duration",
                    severity="hard" if c.hard else "soft",
                    message=(
                        f"Duration limit of {limit_days:.2f} days exceeded with "
                        f"{res_dur.actual_days:.2f} days"
                    ),
                    actual_value=res_dur.actual_days,
                    limit_value=limit_days,
                )
                if c.hard:
                    hard_violations.append(violation)
                else:
                    soft_violations.append(violation)

        elif c.type == ConstraintType.MIN_PERIAPSIS:
            res_periapsis = check_min_periapsis(trajectory, c.limit, c.body)
            physical_results.append(res_periapsis)
            if not res_periapsis.satisfied:
                violation = ConstraintViolation(
                    constraint_type="min_periapsis",
                    severity="hard" if c.hard else "soft",
                    message=(
                        f"Periapsis for {c.body or 'body'} is {res_periapsis.actual_km:.1f} km, "
                        f"which is below min {c.limit:.1f} km"
                    ),
                    actual_value=res_periapsis.actual_km,
                    limit_value=c.limit,
                    body=c.body,
                )
                if c.hard:
                    hard_violations.append(violation)
                else:
                    soft_violations.append(violation)

    # Propellant check (always evaluated, hard constraint)
    prop_result = check_propellant_budget(trajectory, spacecraft)
    if not prop_result.satisfied:
        hard_violations.append(
            ConstraintViolation(
                constraint_type="propellant_budget",
                severity="hard",
                message=(
                    f"Required delta-V {prop_result.required_dv_km_s:.3f} km/s "
                    f"exceeds available budget {prop_result.available_dv_km_s:.3f} km/s"
                ),
                actual_value=prop_result.required_dv_km_s,
                limit_value=prop_result.available_dv_km_s,
            )
        )

    # Launch window check (always evaluated, soft constraint by default)
    lw_result = check_launch_window(
        trajectory.departure_epoch,
        mission.departure_epoch_start,
        mission.departure_epoch_end,
    )
    temporal_results.append(lw_result)
    if not lw_result.satisfied:
        soft_violations.append(
            ConstraintViolation(
                constraint_type="launch_window",
                severity="soft",
                message=(
                    f"Departure epoch {trajectory.departure_epoch} is outside " f"the launch window"
                ),
                actual_value=trajectory.departure_epoch,
                limit_value=mission.departure_epoch_start,
            )
        )

    all_satisfied = len(hard_violations) == 0 and len(soft_violations) == 0

    return ConstraintReport(
        all_satisfied=all_satisfied,
        hard_violations=hard_violations,
        soft_violations=soft_violations,
        physical_results=physical_results,
        propellant_result=prop_result,
        temporal_results=temporal_results,
    )
