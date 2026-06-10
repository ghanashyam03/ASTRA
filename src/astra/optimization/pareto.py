"""Pareto dominance, non-dominated front extraction, and quality metrics.
All functions operate on (N, M) numpy arrays of objective values.
Minimization is assumed for all objectives.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from astra.state.trajectory import Trajectory


def is_dominated(point: np.ndarray, population: np.ndarray) -> bool:
    """Return True if point is Pareto-dominated by any member of population.
    Point a dominates b iff: a[i] <= b[i] for all i, AND a[j] < b[j] for some j.
    Both inputs are 1-D objective vectors (minimization)."""
    if len(population) == 0:
        return False
    dominated_all = np.all(population <= point, axis=1)
    strictly_better = np.any(population < point, axis=1)
    return bool(np.any(dominated_all & strictly_better))

def compute_pareto_front(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract non-dominated points from an (N, M) objective array.
    Returns (pareto_points, pareto_indices) where pareto_indices are
    the original row indices of the non-dominated points.
    Time complexity O(N² × M) — suitable for N < 10,000."""
    n = len(points)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        dominated_by_i = (
            np.all(points[i] <= points, axis=1)
            & np.any(points[i] < points, axis=1)
        )
        dominated_by_i[i] = False
        is_pareto[dominated_by_i] = False
    indices = np.where(is_pareto)[0]
    return points[indices], indices

def hypervolume_indicator_2d(
    pareto_points: np.ndarray,
    reference_point: np.ndarray,
) -> float:
    """Compute the 2-D hypervolume indicator (HVI).
    Measures the area of objective space dominated by the Pareto front
    but not by the reference point. Larger = better quality front.
    reference_point must strictly dominate all Pareto points.
    
    Algorithm: sweep line from left to right after sorting by f1 ascending.
    Time O(N log N).
    
    Example: points=[[1,4],[4,1]], ref=[5,5] → HVI = 7.0
    Verification:
      Sorted by f1: [1,4], [4,1]
      prev_f2 = 5
      pt=[1,4]: width=(5-1)=4, height=(5-4)=1, area=4. prev_f2=min(5,4)=4
      pt=[4,1]: width=(5-4)=1, height=(4-1)=3, area=3. Total=7.
    """
    if len(pareto_points) == 0:
        return 0.0
    pts = pareto_points[np.argsort(pareto_points[:, 0])]
    hv = 0.0
    prev_f2 = float(reference_point[1])
    for pt in pts:
        w = float(reference_point[0]) - float(pt[0])
        h = prev_f2 - float(pt[1])
        if w > 0 and h > 0:
            hv += w * h
        prev_f2 = min(prev_f2, float(pt[1]))
    return hv

def pareto_spread(pareto_points: np.ndarray) -> float:
    """Mean pairwise Euclidean distance between normalized Pareto points.
    Measures diversity of the Pareto front. Higher = more spread = better.
    Normalizes each objective by its range before computing distances."""
    if len(pareto_points) < 2:
        return 0.0
    pts = pareto_points.astype(float)
    ranges = pts.max(axis=0) - pts.min(axis=0)
    ranges[ranges == 0] = 1.0
    pts_norm = (pts - pts.min(axis=0)) / ranges
    n = len(pts_norm)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += float(np.linalg.norm(pts_norm[i] - pts_norm[j]))
            count += 1
    return total / count if count > 0 else 0.0

def pareto_from_trajectories(
    trajectories: list[Trajectory],
) -> tuple[list[Trajectory], np.ndarray]:
    """Extract Pareto-optimal trajectories from a list.
    Objectives: [delta_v_total, duration_days] (both minimize).
    Returns (pareto_trajectories, pareto_indices)."""
    if not trajectories:
        return [], np.array([], dtype=int)
    points = np.array(
        [[t.delta_v_total, t.duration_days] for t in trajectories],
        dtype=float
    )
    _, idx = compute_pareto_front(points)
    return [trajectories[i] for i in idx], idx

@dataclass
class ParetoQualityMetrics:
    n_solutions: int
    hypervolume_indicator: float
    spread: float
    dv_range_km_s: tuple[float, float]
    tof_range_days: tuple[float, float]
    tradeoff_km_s_per_day: float
    reference_point: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_solutions": self.n_solutions,
            "hypervolume_indicator": round(self.hypervolume_indicator, 6),
            "spread": round(self.spread, 6),
            "dv_range_km_s": list(self.dv_range_km_s),
            "tof_range_days": list(self.tof_range_days),
            "tradeoff_km_s_per_day": round(self.tradeoff_km_s_per_day, 6),
            "reference_point": list(self.reference_point),
        }

def compute_pareto_quality(
    trajectories: list[Trajectory],
    reference_multiplier: float = 1.1,
) -> ParetoQualityMetrics:
    """Compute all Pareto quality metrics for a list of trajectories.
    reference_point = (max_dv * 1.1, max_tof * 1.1) — standard practice."""
    if not trajectories:
        raise ValueError("Cannot compute Pareto quality on empty trajectory list")
    dvs = np.array([t.delta_v_total for t in trajectories])
    days = np.array([t.duration_days for t in trajectories])
    pts = np.column_stack([dvs, days])
    ref = np.array([float(dvs.max()) * reference_multiplier,
                    float(days.max()) * reference_multiplier])
    hvi = hypervolume_indicator_2d(pts, ref)
    sp = pareto_spread(pts)
    tof_range = (float(days.min()), float(days.max()))
    dv_range = (float(dvs.min()), float(dvs.max()))
    if tof_range[1] > tof_range[0]:
        tradeoff = abs(dv_range[1] - dv_range[0]) / (tof_range[1] - tof_range[0])
    else:
        tradeoff = 0.0
    return ParetoQualityMetrics(
        n_solutions=len(trajectories),
        hypervolume_indicator=hvi,
        spread=sp,
        dv_range_km_s=dv_range,
        tof_range_days=tof_range,
        tradeoff_km_s_per_day=tradeoff,
        reference_point=(float(ref[0]), float(ref[1])),
    )
