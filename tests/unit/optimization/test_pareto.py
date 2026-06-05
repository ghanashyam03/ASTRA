from __future__ import annotations

import numpy as np

from astra.optimization.pareto import (
    compute_pareto_front,
    hypervolume_indicator_2d,
    is_dominated,
    pareto_spread,
)


def test_dominated() -> None:
    a = np.array([1.0, 1.0])
    b = np.array([2.0, 2.0])
    assert is_dominated(b, np.array([a]))
    assert not is_dominated(a, np.array([b]))


def test_pareto_front_2d() -> None:
    pts = np.array([[1, 4], [2, 3], [3, 2], [4, 1], [2, 4], [3, 3]])
    pareto, idx = compute_pareto_front(pts)
    # [1,4], [2,3], [3,2], [4,1] are non-dominated
    assert len(pareto) == 4
    assert set(idx) == {0, 1, 2, 3}


def test_hypervolume_known_case() -> None:
    """2 points: (1,4) and (4,1), reference (5,5).
    HVI = (5-4)*(5-1) + (4-1)*(1-1+4-1) ... computed geometrically."""
    pts = np.array([[1.0, 4.0], [4.0, 1.0]])
    ref = np.array([5.0, 5.0])
    hv = hypervolume_indicator_2d(pts, ref)
    # Sorted by f1: [1,4], [4,1]
    # Rectangle 1: width=(5-1)=4, height=(5-4)=1 → 4; prev_f2=min(5,4)=4
    # pt=[4,1]: width=(5-4)=1, height=(4-1)=3 → 3; total=7
    assert abs(hv - 7.0) < 0.01, f"Expected 7.0, got {hv}"


def test_pareto_spread_single_point() -> None:
    pts = np.array([[3.0, 200.0]])
    assert pareto_spread(pts) == 0.0
