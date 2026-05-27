"""ASTRA explainability coordinator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from astra.dsl.compiler import CompiledMission
from astra.explainability.constraint_analysis import ConstraintAnalysis, analyze_constraints
from astra.explainability.deltav_decomp import DeltaVDecomposition, decompose_delta_v
from astra.explainability.pareto_analysis import ParetoAnalysis, analyze_pareto
from astra.explainability.window_rationale import WindowRationale, build_window_rationale
from astra.state.trajectory import Trajectory

if TYPE_CHECKING:
    from astra.physics.ephemeris import EphemerisEngine


@dataclass
class ExplanationTrace:
    mission_id: str
    delta_v_decomposition: DeltaVDecomposition
    constraint_analysis: ConstraintAnalysis
    window_rationale: WindowRationale | None
    pareto_analysis: ParetoAnalysis | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "delta_v_decomposition": self.delta_v_decomposition.to_dict(),
            "constraint_analysis": self.constraint_analysis.to_dict(),
            "window_rationale": self.window_rationale.to_dict() if self.window_rationale else None,
            "pareto_analysis": self.pareto_analysis.to_dict() if self.pareto_analysis else None,
        }

def explain(
    trajectory: Trajectory,
    mission: CompiledMission,
    pareto_front: list[Trajectory] | None = None,
    ephemeris: EphemerisEngine | None = None,
) -> ExplanationTrace:
    """Generate complete ExplanationTrace from trajectory and mission data."""
    dv_decomp = decompose_delta_v(trajectory)
    constraint_analysis = analyze_constraints(trajectory, mission.constraints)

    window_rationale = None
    if ephemeris is not None:
        try:
            window_rationale = build_window_rationale(
                trajectory, mission.origin_body, mission.destination_body, ephemeris
            )
        except Exception:
            pass

    pareto_analysis = None
    if pareto_front and len(pareto_front) >= 2:
        try:
            pareto_analysis = analyze_pareto(pareto_front)
        except Exception:
            pass

    return ExplanationTrace(
        mission_id=mission.mission_id,
        delta_v_decomposition=dv_decomp,
        constraint_analysis=constraint_analysis,
        window_rationale=window_rationale,
        pareto_analysis=pareto_analysis,
    )
