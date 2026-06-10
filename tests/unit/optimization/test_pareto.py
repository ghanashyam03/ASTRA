from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from astra.optimization.pareto import (
    compute_pareto_front,
    compute_pareto_quality,
    hypervolume_indicator_2d,
    is_dominated,
    pareto_spread,
)
from astra.state.trajectory import Trajectory


def test_dominance_basic() -> None:
    a = np.array([1.0, 1.0])
    b = np.array([2.0, 2.0])
    assert is_dominated(b, np.array([a]))
    assert not is_dominated(a, np.array([b]))


def test_dominance_nondominated() -> None:
    a = np.array([1.0, 4.0])
    b = np.array([4.0, 1.0])
    assert not is_dominated(a, np.array([b]))
    assert not is_dominated(b, np.array([a]))


def test_pareto_front_2d() -> None:
    pts = np.array([[1, 4], [2, 3], [3, 2], [4, 1], [2, 4], [3, 3]], dtype=float)
    pf, idx = compute_pareto_front(pts)
    assert len(pf) == 4
    assert set(idx) == {0, 1, 2, 3}


def test_hypervolume_known_case() -> None:
    pts = np.array([[1.0, 4.0], [4.0, 1.0]])
    ref = np.array([5.0, 5.0])
    hv = hypervolume_indicator_2d(pts, ref)
    assert abs(hv - 7.0) < 1e-10, f"Expected 7.0, got {hv}"


def test_hypervolume_single_point() -> None:
    pts = np.array([[2.0, 3.0]])
    ref = np.array([5.0, 5.0])
    hv = hypervolume_indicator_2d(pts, ref)
    assert abs(hv - (5 - 2) * (5 - 3)) < 1e-10  # 3*2=6


def test_hypervolume_empty() -> None:
    assert hypervolume_indicator_2d(np.empty((0, 2)), np.array([5.0, 5.0])) == 0.0


def test_pareto_spread_single() -> None:
    pts = np.array([[3.0, 200.0]])
    assert pareto_spread(pts) == 0.0


def test_pareto_spread_two_points() -> None:
    pts = np.array([[1.0, 2.0], [3.0, 4.0]])
    s = pareto_spread(pts)
    assert s > 0.0


def test_compute_pareto_quality() -> None:
    t1 = MagicMock(spec=Trajectory)
    t1.delta_v_total = 5.0
    t1.duration_days = 10.0
    t2 = MagicMock(spec=Trajectory)
    t2.delta_v_total = 4.0
    t2.duration_days = 12.0
    
    metrics = compute_pareto_quality([t1, t2])
    assert metrics.n_solutions == 2
    assert metrics.dv_range_km_s == (4.0, 5.0)
    assert metrics.tof_range_days == (10.0, 12.0)
    assert metrics.to_dict()["n_solutions"] == 2
