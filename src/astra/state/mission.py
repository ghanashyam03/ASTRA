"""Mission domain state objects.

These are richer than CompiledMission — they represent the current state
of a mission optimization run (in-progress, completed, failed) and are
used as interchange types between the optimization engine, constraints
engine, and explainability engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from astra.dsl.compiler import CompiledMission
    from astra.optimization.engine import OptimizationResult



class MissionPhase(StrEnum):
    DEPARTURE = "DEPARTURE"
    TRANSFER = "TRANSFER"
    FLYBY = "FLYBY"
    ARRIVAL = "ARRIVAL"


class MissionStatus(StrEnum):
    PENDING = "PENDING"
    OPTIMIZING = "OPTIMIZING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    INFEASIBLE = "INFEASIBLE"


@dataclass
class MissionLeg:
    """A single transfer arc in a multi-body mission."""

    phase: MissionPhase
    origin: str              # body name
    destination: str
    departure_epoch: float   # J2000 seconds
    arrival_epoch: float
    delta_v_km_s: float
    tof_days: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def tof_seconds(self) -> float:
        return self.tof_days * 86400.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "origin": self.origin,
            "destination": self.destination,
            "departure_epoch_j2000": self.departure_epoch,
            "arrival_epoch_j2000": self.arrival_epoch,
            "delta_v_km_s": round(self.delta_v_km_s, 4),
            "tof_days": round(self.tof_days, 2),
            "metadata": self.metadata,
        }


@dataclass
class MissionSummary:
    """High-level summary of a completed mission optimization.

    Produced by the explainability engine for mission report generation.
    """

    mission_id: str
    status: MissionStatus
    origin: str
    destination: str
    flyby_bodies: list[str]
    total_delta_v_km_s: float
    total_duration_days: float
    legs: list[MissionLeg]
    n_optimization_trials: int
    n_feasible_solutions: int
    wall_time_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "status": self.status.value,
            "route": " → ".join([self.origin] + self.flyby_bodies + [self.destination]),
            "total_delta_v_km_s": round(self.total_delta_v_km_s, 4),
            "total_duration_days": round(self.total_duration_days, 2),
            "legs": [leg.to_dict() for leg in self.legs],
            "optimization": {
                "n_trials": self.n_optimization_trials,
                "n_feasible": self.n_feasible_solutions,
                "wall_time_s": round(self.wall_time_s, 2),
            },
        }


def mission_summary_from_result(
    result: OptimizationResult,
    mission: CompiledMission,
) -> MissionSummary:
    """Build a MissionSummary from an optimization result."""
    best = result.best_trajectory
    if best is None:
        return MissionSummary(
            mission_id=mission.mission_id,
            status=MissionStatus.INFEASIBLE,
            origin=mission.origin_body.value,
            destination=mission.destination_body.value,
            flyby_bodies=[],
            total_delta_v_km_s=0.0,
            total_duration_days=0.0,
            legs=[],
            n_optimization_trials=result.n_evaluations,
            n_feasible_solutions=result.n_feasible,
            wall_time_s=result.wall_time_s,
        )
    legs = []
    for i, m in enumerate(best.maneuvers):
        phase = MissionPhase.DEPARTURE if i == 0 else (
            MissionPhase.ARRIVAL if i == len(best.maneuvers) - 1
            else MissionPhase.FLYBY
        )
        origin_name = mission.origin_body.value if i == 0 else "UNKNOWN"
        dest_name = mission.destination_body.value if i == len(best.maneuvers) - 1 else "UNKNOWN"
        tof = 0.0
        if i + 1 < len(best.states):
            tof = (best.states[i+1].epoch - best.states[i].epoch) / 86400.0
        legs.append(MissionLeg(
            phase=phase,
            origin=origin_name,
            destination=dest_name,
            departure_epoch=best.states[i].epoch if i < len(best.states) else 0.0,
            arrival_epoch=best.states[i+1].epoch if i+1 < len(best.states) else 0.0,
            delta_v_km_s=m.magnitude,
            tof_days=tof,
            metadata={"label": m.label},
        ))
    flyby_bodies = [
        m.label.replace("FLY_", "")
        for m in best.maneuvers
        if m.label.startswith("FLY_")
    ]
    return MissionSummary(
        mission_id=mission.mission_id,
        status=MissionStatus.COMPLETE,
        origin=mission.origin_body.value,
        destination=mission.destination_body.value,
        flyby_bodies=flyby_bodies,
        total_delta_v_km_s=best.delta_v_total,
        total_duration_days=best.duration_days,
        legs=legs,
        n_optimization_trials=result.n_evaluations,
        n_feasible_solutions=result.n_feasible,
        wall_time_s=result.wall_time_s,
    )
