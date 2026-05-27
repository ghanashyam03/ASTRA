from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from astra.state.trajectory import Trajectory


@dataclass
class DeltaVComponent:
    label: str
    magnitude_km_s: float
    fraction_of_total: float
    epoch_j2000: float

@dataclass
class DeltaVDecomposition:
    components: list[DeltaVComponent]
    total_km_s: float
    margin_km_s: float  # 3% budget margin

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_km_s": round(self.total_km_s, 4),
            "margin_km_s": round(self.margin_km_s, 4),
            "components": [
                {
                    "label": c.label,
                    "magnitude_km_s": round(c.magnitude_km_s, 4),
                    "fraction_pct": round(c.fraction_of_total * 100, 1),
                    "epoch_j2000": c.epoch_j2000,
                }
                for c in self.components
            ],
        }

def decompose_delta_v(trajectory: Trajectory) -> DeltaVDecomposition:
    """Decompose trajectory delta-v budget by maneuver."""
    total = trajectory.delta_v_total
    if total == 0.0:
        return DeltaVDecomposition(components=[], total_km_s=0.0, margin_km_s=0.0)

    components = [
        DeltaVComponent(
            label=m.label if m.label else f"Maneuver {i+1}",
            magnitude_km_s=m.magnitude,
            fraction_of_total=m.magnitude / total,
            epoch_j2000=m.epoch,
        )
        for i, m in enumerate(trajectory.maneuvers)
    ]
    return DeltaVDecomposition(
        components=components,
        total_km_s=total,
        margin_km_s=total * 0.03,  # 3% navigation margin
    )
