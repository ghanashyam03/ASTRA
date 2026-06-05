"""Convert Pareto front data to Plotly-ready scatter plot structure."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from astra.optimization.pareto import hypervolume_indicator_2d, pareto_spread
from astra.state.trajectory import Trajectory


@dataclass
class ParetoPlotData:
    dv_km_s: list[float]
    tof_days: list[float]
    hypervolume: float
    spread: float
    fuel_optimal_idx: int
    time_optimal_idx: int
    n_solutions: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "dv_km_s": [round(v, 4) for v in self.dv_km_s],
            "tof_days": [round(v, 2) for v in self.tof_days],
            "hypervolume_indicator": round(self.hypervolume, 4),
            "pareto_spread": round(self.spread, 4),
            "fuel_optimal_idx": self.fuel_optimal_idx,
            "time_optimal_idx": self.time_optimal_idx,
            "n_solutions": self.n_solutions,
        }


def build_pareto_plot(trajectories: list[Trajectory]) -> ParetoPlotData:
    """Build Plotly-ready Pareto scatter data with quality metrics."""
    if not trajectories:
        return ParetoPlotData(
            dv_km_s=[],
            tof_days=[],
            hypervolume=0.0,
            spread=0.0,
            fuel_optimal_idx=-1,
            time_optimal_idx=-1,
            n_solutions=0,
        )

    dvs = [t.delta_v_total for t in trajectories]
    days = [t.duration_days for t in trajectories]
    pts = np.array([[d, t] for d, t in zip(dvs, days)])
    
    # Safe reference point definition
    ref = np.array([max(dvs) * 1.1 if dvs else 10.0, max(days) * 1.1 if days else 100.0])
    hv = hypervolume_indicator_2d(pts, ref)
    spread = pareto_spread(pts)
    
    return ParetoPlotData(
        dv_km_s=dvs,
        tof_days=days,
        hypervolume=hv,
        spread=spread,
        fuel_optimal_idx=int(np.argmin(dvs)),
        time_optimal_idx=int(np.argmin(days)),
        n_solutions=len(trajectories),
    )
