from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from astra.dsl.compiler import CompiledConstraint
from astra.dsl.schema import ConstraintType
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
) -> ConstraintAnalysis:
    statuses = []
    for c in constraints:
        if c.type == ConstraintType.MAX_DELTA_V:
            actual = trajectory.delta_v_total
            limit = c.limit
        elif c.type == ConstraintType.MAX_DURATION:
            actual = trajectory.duration_days
            limit = c.limit / 86400.0
        else:
            continue

        satisfied = actual <= limit
        margin_pct = (limit - actual) / limit * 100.0 if limit > 0 else 0.0
        binding = margin_pct < 5.0 and satisfied

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
