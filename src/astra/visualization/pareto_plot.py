"""Convert Pareto front trajectory data into Plotly-ready scatter structure."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from astra.optimization.pareto import ParetoQualityMetrics, compute_pareto_quality
from astra.state.trajectory import Trajectory


@dataclass
class ParetoPlotData:
    dv_km_s: list[float]
    tof_days: list[float]
    departure_dates: list[str]     # ISO date strings if ephemeris provided else empty
    quality: ParetoQualityMetrics
    fuel_optimal_idx: int
    time_optimal_idx: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "dv_km_s": [round(v, 4) for v in self.dv_km_s],
            "tof_days": [round(v, 2) for v in self.tof_days],
            "departure_dates": self.departure_dates,
            "fuel_optimal_idx": self.fuel_optimal_idx,
            "time_optimal_idx": self.time_optimal_idx,
            "quality": self.quality.to_dict(),
            # For backward compatibility with existing tests/endpoints:
            "hypervolume_indicator": round(self.quality.hypervolume_indicator, 4),
            "pareto_spread": round(self.quality.spread, 4),
            "n_solutions": self.quality.n_solutions,
        }

def build_pareto_plot(
    trajectories: list[Trajectory],
    ephemeris: Any = None,
) -> ParetoPlotData:
    """Build Plotly-ready Pareto scatter data with quality metrics.
    If ephemeris provided, converts departure epochs to ISO date strings."""
    if not trajectories:
        default_quality = ParetoQualityMetrics(
            n_solutions=0,
            hypervolume_indicator=0.0,
            spread=0.0,
            dv_range_km_s=(0.0, 0.0),
            tof_range_days=(0.0, 0.0),
            tradeoff_km_s_per_day=0.0,
            reference_point=(0.0, 0.0),
        )
        return ParetoPlotData(
            dv_km_s=[],
            tof_days=[],
            departure_dates=[],
            quality=default_quality,
            fuel_optimal_idx=-1,
            time_optimal_idx=-1,
        )

    dvs = [t.delta_v_total for t in trajectories]
    days = [t.duration_days for t in trajectories]
    quality = compute_pareto_quality(trajectories)
    
    dep_dates: list[str] = []
    if ephemeris is not None:
        for t in trajectories:
            try:
                dep_dates.append(ephemeris.date_from_epoch(t.departure_epoch)[:10])
            except Exception:
                dep_dates.append(f"J2000+{t.departure_epoch/86400:.0f}d")
    
    return ParetoPlotData(
        dv_km_s=dvs,
        tof_days=days,
        departure_dates=dep_dates,
        quality=quality,
        fuel_optimal_idx=int(np.argmin(dvs)),
        time_optimal_idx=int(np.argmin(days)),
    )
