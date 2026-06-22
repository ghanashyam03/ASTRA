from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from astra.state.orbital_state import OrbitalState


@dataclass
class Maneuver:
    epoch: float  # J2000 seconds
    delta_v: np.ndarray  # [dvx, dvy, dvz] km/s
    label: str = ""

    @property
    def magnitude(self) -> float:
        return float(np.linalg.norm(self.delta_v))


@dataclass
class Trajectory:
    states: list[OrbitalState] = field(default_factory=list)
    maneuvers: list[Maneuver] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def delta_v_total(self) -> float:
        return sum(m.magnitude for m in self.maneuvers)

    @property
    def duration_seconds(self) -> float:
        if len(self.states) < 2:
            return 0.0
        return self.states[-1].epoch - self.states[0].epoch

    @property
    def duration_days(self) -> float:
        return self.duration_seconds / 86400.0

    @property
    def departure_epoch(self) -> float:
        return self.states[0].epoch if self.states else 0.0

    @property
    def arrival_epoch(self) -> float:
        return self.states[-1].epoch if self.states else 0.0

    def is_feasible(self, max_dv: float, max_days: float) -> bool:
        return self.delta_v_total <= max_dv and self.duration_days <= max_days

    def to_dict(self) -> dict[str, Any]:
        return {
            "delta_v_total_km_s": round(self.delta_v_total, 6),
            "duration_days": round(self.duration_days, 3),
            "departure_epoch_j2000": self.departure_epoch,
            "arrival_epoch_j2000": self.arrival_epoch,
            "n_maneuvers": len(self.maneuvers),
            "maneuvers": [
                {
                    "epoch": m.epoch,
                    "dv_km_s": m.delta_v.tolist(),
                    "magnitude_km_s": round(m.magnitude, 6),
                    "label": m.label,
                }
                for m in self.maneuvers
            ],
            "metadata": self.metadata,
        }


@dataclass
class TrajectoryValidationResult:
    is_valid: bool
    dv_diff: float
    pos_diff: float
    diagnostics: dict[str, Any] = field(default_factory=dict)
