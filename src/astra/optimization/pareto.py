"""Pareto front computation, dominance testing, and quality metrics.
Implements hypervolume indicator (HVI) for Pareto front quality assessment.
"""
from __future__ import annotations

import numpy as np


def is_dominated(point: np.ndarray, other_points: np.ndarray) -> bool:
    """Return True if point is dominated by any point in other_points.
    Point a dominates b if: a[i] <= b[i] for all i, and a[j] < b[j] for some j.
    Assumes minimization objectives.
    """
    if len(other_points) == 0:
        return False
    dominated_in_all = np.all(other_points <= point, axis=1)
    strictly_better = np.any(other_points < point, axis=1)
    return bool(np.any(dominated_in_all & strictly_better))


def compute_pareto_front(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract Pareto-optimal points from a set of objective vectors.

    Parameters
    ----------
    points : (N, M) array of objective values (minimization)

    Returns
    -------
    pareto_points : (K, M) array of non-dominated points
    pareto_indices : (K,) indices into original points array
    """
    n = len(points)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if is_pareto[i]:
            dominated_by_i = (
                np.all(points[i] <= points, axis=1)
                & np.any(points[i] < points, axis=1)
            )
            is_pareto &= ~dominated_by_i
            is_pareto[i] = True
    indices = np.where(is_pareto)[0]
    return points[indices], indices


def hypervolume_indicator_2d(
    pareto_points: np.ndarray,
    reference_point: np.ndarray,
) -> float:
    """Compute 2D hypervolume indicator (area dominated by Pareto front).

    The reference point must dominate all Pareto points.
    Larger HVI = better Pareto front quality.

    Algorithm: sort by first objective, then sum rectangles.
    Time: O(N log N) for N Pareto points.
    """
    if len(pareto_points) == 0:
        return 0.0
    pts = pareto_points.copy()
    # Sort by first objective ascending
    pts = pts[np.argsort(pts[:, 0])]
    hv = 0.0
    prev_f2 = reference_point[1]
    for pt in pts:
        width = reference_point[0] - pt[0]
        height = prev_f2 - pt[1]
        if width > 0 and height > 0:
            hv += width * height
        prev_f2 = min(prev_f2, pt[1])
    return float(hv)


def pareto_spread(pareto_points: np.ndarray) -> float:
    """Mean pairwise Euclidean distance between normalized Pareto points.
    Measures diversity of the Pareto front (higher = more spread = better).
    """
    if len(pareto_points) < 2:
        return 0.0
    pts = pareto_points.astype(float)
    ranges = pts.max(axis=0) - pts.min(axis=0)
    ranges[ranges == 0] = 1.0
    pts_norm = (pts - pts.min(axis=0)) / ranges
    dists = []
    for i in range(len(pts_norm)):
        for j in range(i + 1, len(pts_norm)):
            dists.append(float(np.linalg.norm(pts_norm[i] - pts_norm[j])))
    return float(np.mean(dists)) if dists else 0.0
