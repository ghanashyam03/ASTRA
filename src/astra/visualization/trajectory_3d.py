"""Convert trajectories to 3D rendering-ready data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from astra.state.trajectory import Trajectory

AU = 1.496e8  # km


@dataclass
class Body3DData:
    name: str
    positions_au: list[list[float]]  # [x,y,z] in AU for each epoch
    epochs: list[float]


@dataclass
class TrajectoryRenderData:
    mission_id: str
    spacecraft_positions_au: list[list[float]]
    spacecraft_epochs: list[float]
    maneuver_epochs: list[float]
    maneuver_dv_vectors: list[list[float]]
    body_data: dict[str, Body3DData]
    delta_v_total_km_s: float
    duration_days: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "spacecraft": {
                "positions_au": self.spacecraft_positions_au,
                "epochs_j2000": self.spacecraft_epochs,
            },
            "maneuvers": [
                {"epoch": e, "dv_km_s": dv}
                for e, dv in zip(self.maneuver_epochs, self.maneuver_dv_vectors)
            ],
            "bodies": {
                name: {
                    "positions_au": b.positions_au,
                    "epochs_j2000": b.epochs,
                }
                for name, b in self.body_data.items()
            },
            "summary": {
                "delta_v_total_km_s": round(self.delta_v_total_km_s, 4),
                "duration_days": round(self.duration_days, 2),
            },
        }


def build_render_data(
    trajectory: Trajectory,
    mission_id: str,
    body_states: dict[str, list[tuple[float, np.ndarray]]] | None = None,
) -> TrajectoryRenderData:
    """Convert Trajectory into visualization-ready dict."""
    sc_positions = [
        [float(s.position[0] / AU), float(s.position[1] / AU), float(s.position[2] / AU)]
        for s in trajectory.states
    ]
    sc_epochs = [s.epoch for s in trajectory.states]

    body_data: dict[str, Body3DData] = {}
    if body_states:
        for name, state_list in body_states.items():
            body_data[name] = Body3DData(
                name=name,
                positions_au=[
                    [float(pos[0] / AU), float(pos[1] / AU), float(pos[2] / AU)]
                    for _, pos in state_list
                ],
                epochs=[e for e, _ in state_list],
            )

    return TrajectoryRenderData(
        mission_id=mission_id,
        spacecraft_positions_au=sc_positions,
        spacecraft_epochs=sc_epochs,
        maneuver_epochs=[m.epoch for m in trajectory.maneuvers],
        maneuver_dv_vectors=[m.delta_v.tolist() for m in trajectory.maneuvers],
        body_data=body_data,
        delta_v_total_km_s=trajectory.delta_v_total,
        duration_days=trajectory.duration_days,
    )
