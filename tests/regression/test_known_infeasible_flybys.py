# MANDATORY: this test file must pass before any change to physics/flyby.py,
# optimization/chain_solver.py, or optimization/mcts.py is merged. It is not
# marked slow and requires no SPICE kernels — it must run on every commit.

"""Permanent regression bank — codifies the manual audit methodology that caught
the original Venus failure into an automatically-run, data-driven test suite.
Adding a new known-bad or known-good case requires only a JSON entry in
known_infeasible_cases.json, never a code change to this file.
"""

import json
import math
from pathlib import Path
from typing import Any

import pytest

from astra.physics.flyby import check_flyby_feasibility

CASES_PATH = Path(__file__).parent / "known_infeasible_cases.json"


def load_cases() -> list[dict[str, Any]]:
    return json.loads(CASES_PATH.read_text())  # type: ignore[no-any-return]


@pytest.mark.parametrize("case", load_cases(), ids=lambda c: c["name"])
def test_known_case(case: dict[str, Any]) -> None:

    v_inf = case["v_inf_km_s"]
    required_turn_rad = math.radians(case["required_turn_deg"])
    feas = check_flyby_feasibility(v_inf, required_turn_rad, case["body"])

    expected_unpowered = case["must_be_achievable_unpowered"]
    assert feas.is_achievable_unpowered == expected_unpowered, (
        f"{case['name']}: expected unpowered achievability={expected_unpowered}, "
        f"got {feas.is_achievable_unpowered}. Ceiling was "
        f"{math.degrees(feas.max_unpowered_turn_rad):.2f}° vs required "
        f"{case['required_turn_deg']:.2f}°"
    )

    expected_with_burn = case.get("must_be_achievable_with_unlimited_burn")
    if expected_with_burn is not None:
        assert feas.is_achievable_with_bounded_burn == expected_with_burn, (
            f"{case['name']}: expected powered achievability={expected_with_burn}, "
            f"got {feas.is_achievable_with_bounded_burn}"
        )

    if "expected_unpowered_ceiling_deg_approx" in case:
        actual_ceiling_deg = math.degrees(feas.max_unpowered_turn_rad)
        expected = case["expected_unpowered_ceiling_deg_approx"]
        assert abs(actual_ceiling_deg - expected) < 5.0, (
            f"{case['name']}: computed ceiling {actual_ceiling_deg:.2f}° does not "
            f"match the hand-derived expected ceiling {expected:.2f}° within 5° — "
            f"this means either the formula or the test case derivation is wrong"
        )


def test_case_bank_has_both_feasible_and_infeasible_cases() -> None:
    """Sanity check on the bank itself: it must contain a mix, or it isn't
    testing discriminating power."""
    cases = load_cases()
    feasible_count = sum(1 for c in cases if c["must_be_achievable_unpowered"])
    infeasible_count = sum(1 for c in cases if not c["must_be_achievable_unpowered"])
    assert feasible_count >= 2, "Bank needs at least 2 known-feasible control cases"
    assert infeasible_count >= 2, "Bank needs at least 2 known-infeasible cases"
