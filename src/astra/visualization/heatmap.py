"""Convert porkchop grid data to Plotly-ready heatmap structure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PorkchopPlotData:
    departure_labels: list[str]  # ISO date strings
    tof_labels: list[float]  # days
    dv_grid: list[list[float | None]]  # (n_dep, n_tof) — NaN replaced with null
    dv_min: float
    dv_max: float
    optimal_departure_idx: int
    optimal_tof_idx: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "departure_labels": self.departure_labels,
            "tof_days": self.tof_labels,
            "dv_grid_km_s": self.dv_grid,
            "dv_min_km_s": round(self.dv_min, 4),
            "dv_max_km_s": round(self.dv_max, 4),
            "optimal_point": {
                "departure_idx": self.optimal_departure_idx,
                "tof_idx": self.optimal_tof_idx,
            },
        }


def build_porkchop_plot(
    dep_epochs: np.ndarray,
    tof_days: np.ndarray,
    dv_grid: np.ndarray,
    ephemeris: object = None,
) -> PorkchopPlotData:
    """Build Plotly-ready porkchop data from raw grid arrays."""
    # Convert epochs to date labels
    if ephemeris is not None:
        try:
            date_fn = getattr(ephemeris, "date_from_epoch")
            dep_labels = [date_fn(e)[:10] for e in dep_epochs]
        except Exception:
            dep_labels = [f"J2000+{e/86400:.0f}d" for e in dep_epochs]
    else:
        dep_labels = [f"J2000+{e/86400:.0f}d" for e in dep_epochs]

    finite = dv_grid[np.isfinite(dv_grid)]
    dv_min = float(np.min(finite)) if len(finite) > 0 else 0.0
    dv_max = float(np.min(finite[finite <= np.percentile(finite, 90)])) if len(finite) > 0 else 10.0

    # Find optimal point
    opt_flat = int(np.nanargmin(dv_grid))
    opt_i, opt_j = np.unravel_index(opt_flat, dv_grid.shape)

    # Replace NaN with None for JSON serialization
    grid_list: list[list[float | None]] = []
    for row in dv_grid:
        grid_list.append([None if np.isnan(v) else round(float(v), 4) for v in row])

    return PorkchopPlotData(
        departure_labels=dep_labels,
        tof_labels=[round(float(t), 1) for t in tof_days],
        dv_grid=grid_list,
        dv_min=dv_min,
        dv_max=dv_max,
        optimal_departure_idx=int(opt_i),
        optimal_tof_idx=int(opt_j),
    )
