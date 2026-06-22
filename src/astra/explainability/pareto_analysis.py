from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from astra.state.trajectory import Trajectory


@dataclass
class ParetoPoint:
    delta_v_km_s: float
    duration_days: float
    rank: int


@dataclass
class ParetoAnalysis:
    points: list[ParetoPoint]
    fuel_optimal: Trajectory
    time_optimal: Trajectory
    dv_range_km_s: tuple[float, float]
    tof_range_days: tuple[float, float]
    tradeoff_km_s_per_day: float  # avg Δv cost per day saved

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_pareto_solutions": len(self.points),
            "fuel_optimal_dv_km_s": round(self.dv_range_km_s[0], 4),
            "time_optimal_dv_km_s": round(self.dv_range_km_s[1], 4),
            "tof_range_days": [round(self.tof_range_days[0], 1), round(self.tof_range_days[1], 1)],
            "avg_tradeoff_km_s_per_day": round(self.tradeoff_km_s_per_day, 5),
            "interpretation": (
                f"Saving 1 day of flight time costs approximately "
                f"{self.tradeoff_km_s_per_day:.4f} km/s extra Δv."
            ),
        }


def analyze_pareto(pareto_trajectories: list[Trajectory]) -> ParetoAnalysis:
    if not pareto_trajectories:
        raise ValueError("Cannot analyze empty Pareto front")
    dvs = np.array([t.delta_v_total for t in pareto_trajectories])
    days = np.array([t.duration_days for t in pareto_trajectories])
    fuel_idx = int(np.argmin(dvs))
    time_idx = int(np.argmin(days))
    dv_range = (float(dvs.min()), float(dvs.max()))
    tof_range = (float(days.min()), float(days.max()))

    # Average tradeoff slope
    if tof_range[1] > tof_range[0]:
        tradeoff = (dv_range[1] - dv_range[0]) / (tof_range[1] - tof_range[0])
    else:
        tradeoff = 0.0

    points = [
        ParetoPoint(delta_v_km_s=float(dvs[i]), duration_days=float(days[i]), rank=i)
        for i in range(len(pareto_trajectories))
    ]
    return ParetoAnalysis(
        points=points,
        fuel_optimal=pareto_trajectories[fuel_idx],
        time_optimal=pareto_trajectories[time_idx],
        dv_range_km_s=dv_range,
        tof_range_days=tof_range,
        tradeoff_km_s_per_day=abs(tradeoff),
    )
