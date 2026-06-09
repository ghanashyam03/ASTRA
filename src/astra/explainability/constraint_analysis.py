from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from astra.constraints.engine import evaluate_all_constraints
from astra.dsl.compiler import CompiledConstraint, CompiledMission
from astra.dsl.schema import ConstraintType
from astra.state.orbital_state import PHYSICAL_RADIUS, CelestialBody
from astra.state.spacecraft import PropulsionSystem, PropulsionType, Spacecraft
from astra.state.trajectory import Trajectory


@dataclass
class ConstraintStatus:
    name: str
    type: ConstraintType
    limit: float
    actual: float
    satisfied: bool
    margin_pct: float   # how far below limit (positive = headroom)
    binding: bool       # within 5% of limit = binding

@dataclass
class ConstraintAnalysis:
    statuses: list[ConstraintStatus]
    all_satisfied: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "all_satisfied": self.all_satisfied,
            "constraints": [
                {
                    "name": s.name,
                    "type": s.type.value,
                    "limit": round(s.limit, 4),
                    "actual": round(s.actual, 4),
                    "satisfied": s.satisfied,
                    "margin_pct": round(s.margin_pct, 1),
                    "binding": s.binding,
                }
                for s in self.statuses
            ],
        }

def analyze_constraints(
    trajectory: Trajectory,
    constraints: list[CompiledConstraint],
    mission: CompiledMission | None = None,
    spacecraft: Spacecraft | None = None,
) -> ConstraintAnalysis:
    if spacecraft is None:
        if mission is not None:
            spacecraft = mission.spacecraft
        else:
            spacecraft = Spacecraft(
                name="DummySpacecraft",
                dry_mass_kg=1000.0,
                propulsion=PropulsionSystem(
                    type=PropulsionType.CHEMICAL,
                    isp_seconds=300.0,
                    thrust_newtons=0.0,
                    propellant_mass_kg=1000.0,
                ),
            )
    if mission is None:
        mission = CompiledMission(
            mission_id="dummy",
            spacecraft=spacecraft,
            origin_body=CelestialBody.EARTH,
            destination_body=CelestialBody.MARS,
            departure_epoch_start=trajectory.departure_epoch - 86400.0,
            departure_epoch_end=trajectory.departure_epoch + 86400.0,
            tof_min_seconds=0.0,
            tof_max_seconds=864000.0,
            tof_step_seconds=86400.0,
            constraints=constraints,
            objectives=[],
            seed=42,
            max_evaluations=100,
        )

    # Evaluate constraints using engine
    report = evaluate_all_constraints(trajectory, mission, spacecraft)

    statuses = []
    for c in constraints:
        limit = c.limit
        actual = 0.0
        satisfied = True

        if c.type == ConstraintType.MAX_DELTA_V:
            actual = trajectory.delta_v_total
            limit = c.limit
            res_list = [r for r in report.physical_results if r.constraint_type == "max_delta_v"]
            if res_list:
                satisfied = res_list[0].satisfied
            else:
                satisfied = actual <= limit
            margin_pct = (limit - actual) / limit * 100.0 if limit > 0 else 0.0
            binding = margin_pct < 5.0 and satisfied

        elif c.type == ConstraintType.MAX_DURATION:
            actual = trajectory.duration_days
            limit = c.limit / 86400.0
            res_list = [r for r in report.temporal_results if r.constraint_type == "max_duration"]
            if res_list:
                satisfied = res_list[0].satisfied
                actual = res_list[0].actual_days
                limit = res_list[0].limit_days
            else:
                satisfied = actual <= limit
            margin_pct = (limit - actual) / limit * 100.0 if limit > 0 else 0.0
            binding = margin_pct < 5.0 and satisfied

        elif c.type == ConstraintType.MIN_PERIAPSIS:
            res_list = [
                r for r in report.physical_results
                if r.constraint_type == "min_periapsis" and r.body == c.body
            ]
            if res_list:
                satisfied = res_list[0].satisfied
                actual = res_list[0].actual_km
                limit = res_list[0].limit_km
            else:
                metadata = trajectory.metadata or {}
                b_name = c.body if c.body else "EARTH"
                try:
                    origin_body = CelestialBody[b_name.upper()]
                except (KeyError, AttributeError):
                    origin_body = CelestialBody.EARTH
                if "capture_altitude_km" in metadata and b_name.upper() != "EARTH":
                    actual = PHYSICAL_RADIUS[origin_body] + metadata["capture_altitude_km"]
                elif "parking_altitude_km" in metadata:
                    actual = PHYSICAL_RADIUS[origin_body] + metadata["parking_altitude_km"]
                else:
                    actual = 0.0
                satisfied = actual >= limit
            margin_pct = (actual - limit) / limit * 100.0 if limit > 0 else 0.0
            binding = margin_pct < 5.0 and satisfied
        else:
            continue

        statuses.append(ConstraintStatus(
            name=c.type.value,
            type=c.type,
            limit=limit,
            actual=actual,
            satisfied=satisfied,
            margin_pct=margin_pct,
            binding=binding,
        ))

    return ConstraintAnalysis(
        statuses=statuses,
        all_satisfied=all(s.satisfied for s in statuses),
    )

