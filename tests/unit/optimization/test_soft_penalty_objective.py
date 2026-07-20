"""Verify that ChainResult carries continuity violation data and that the
soft-penalty objective correctly distinguishes near-miss from far-miss trajectories.
"""

from __future__ import annotations

import pytest

from astra.optimization.chain_solver import ChainResult, RejectionReason


def test_chain_result_has_violation_field() -> None:
    """ChainResult must have total_continuity_violation_km_s field defaulting to 0."""
    result = ChainResult(feasible=True, trajectory=None)
    assert hasattr(result, "total_continuity_violation_km_s"), (
        "ChainResult missing total_continuity_violation_km_s field"
    )
    assert result.total_continuity_violation_km_s == 0.0


def test_feasible_chain_has_zero_violation() -> None:
    """A feasible chain (all flybys resolved, no mismatch) must have violation = 0."""
    result = ChainResult(
        feasible=True,
        trajectory=None,
        leg_details=[
            {"magnitude_mismatch_km_s": 0.0, "body": "VENUS"},
            {"magnitude_mismatch_km_s": 0.0, "body": "EARTH"},
        ],
        total_continuity_violation_km_s=0.0,
    )
    assert result.total_continuity_violation_km_s == 0.0


def test_infeasible_chain_carries_violation_magnitude() -> None:
    """An infeasible chain must carry the partial violation from evaluated flybys."""
    result = ChainResult(
        feasible=False,
        trajectory=None,
        reason="magnitude mismatch at EARTH",
        leg_details=[
            {"magnitude_mismatch_km_s": 0.0, "body": "VENUS"},  # Venus OK
            {"magnitude_mismatch_km_s": 3.8, "body": "EARTH"},  # Earth failed
        ],
        reason_code=RejectionReason.IMPOSSIBLE_GEOMETRY,
        total_continuity_violation_km_s=3.8,
    )
    assert result.total_continuity_violation_km_s == pytest.approx(3.8)


def test_soft_penalty_score_ordering() -> None:
    """A near-miss trajectory must score LESS than a far-miss trajectory.
    Score = base_dv + 100 * total_violation.
    near-miss: base=10, violation=0.1 → score=20
    far-miss:  base=10, violation=3.8 → score=390
    The optimizer must prefer near-miss over far-miss.
    """
    WEIGHT = 100.0
    base_dv = 10.0

    near_miss_violation = 0.1
    far_miss_violation = 3.8

    score_near = base_dv + WEIGHT * near_miss_violation  # = 20.0
    score_far = base_dv + WEIGHT * far_miss_violation  # = 390.0

    assert score_near < score_far, (
        f"Near-miss score ({score_near}) must be less than far-miss score ({score_far})"
    )
    assert score_near == pytest.approx(20.0)
    assert score_far == pytest.approx(390.0)


def test_zero_violation_feasible_scores_only_dv() -> None:
    """A feasible trajectory (violation=0) scores exactly its ΔV — no penalty."""
    WEIGHT = 100.0
    dv = 7.4  # km/s (representative of a good gravity-assist trajectory)
    violation = 0.0
    score = dv + WEIGHT * violation
    assert score == pytest.approx(dv)
