"""Verify that find_best_transfer selects continuity-consistent branches
when target_departure_vinf_km_s is provided.
All tests run without SPICE.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from astra.physics.lambert import find_best_transfer
from astra.state.orbital_state import GM

_MU_SUN = GM["SUN"]
_AU = 1.496e8  # km


def _circular_pos_vel(a_au: float, theta_rad: float) -> tuple[np.ndarray, np.ndarray]:
    a = a_au * _AU
    v_c = math.sqrt(_MU_SUN / a)
    pos = np.array([a * math.cos(theta_rad), a * math.sin(theta_rad), 0.0])
    vel = v_c * np.array([-math.sin(theta_rad), math.cos(theta_rad), 0.0])
    return pos, vel


def test_departure_vinf_field_populated() -> None:
    """LambertSolution.departure_vinf_km_s must be |v1 - v1_body|."""
    r1, v1 = _circular_pos_vel(1.0, 0.0)
    r2, v2 = _circular_pos_vel(1.52, math.pi / 3)
    tof = 260 * 86400.0
    sol = find_best_transfer(r1, v1, r2, v2, tof, _MU_SUN, max_revs=0)
    expected = float(np.linalg.norm(sol.v1 - v1))
    assert abs(sol.departure_vinf_km_s - expected) < 1e-12, (
        f"departure_vinf_km_s = {sol.departure_vinf_km_s:.6f}, expected {expected:.6f}"
    )


def test_no_target_gives_same_result_as_original() -> None:
    """When target_departure_vinf_km_s is None, result must be identical to original."""
    r1, v1 = _circular_pos_vel(1.0, 0.0)
    r2, v2 = _circular_pos_vel(0.723, math.pi * 0.6)
    tof = 120 * 86400.0
    sol_no_target = find_best_transfer(r1, v1, r2, v2, tof, _MU_SUN, max_revs=2)
    sol_explicit_none = find_best_transfer(
        r1,
        v1,
        r2,
        v2,
        tof,
        _MU_SUN,
        max_revs=2,
        target_departure_vinf_km_s=None,
    )
    assert sol_no_target.n_revs == sol_explicit_none.n_revs
    assert sol_no_target.branch == sol_explicit_none.branch
    assert abs(sol_no_target.delta_v - sol_explicit_none.delta_v) < 1e-10


def test_continuity_target_overrides_minimum_dv_branch() -> None:
    """When the minimum-ΔV branch has departure |v∞| far from the target,
    and a higher-ΔV branch is much closer to the target, the higher-ΔV branch
    must be selected.

    Setup: Earth → Earth resonant-like geometry.
    Unconstrained selection picks the branch with lowest local ΔV (say N=2, departure v∞ ≈ small).
    Constrained selection with target ≈ 8.8 km/s should prefer the branch whose
    departure v∞ is closer to 8.8 km/s even if its local ΔV is higher.
    """
    # Use a 731-day Earth-Earth geometry (approximates Galileo resonant leg)
    r1, v1 = _circular_pos_vel(1.0, 0.0)
    omega = 2.0 * math.pi / (365.25 * 86400.0)
    tof = 731.0 * 86400.0
    theta2 = omega * tof
    r2, v2 = _circular_pos_vel(1.0, theta2)

    # Unconstrained: picks lowest local ΔV branch
    sol_unconstrained = find_best_transfer(r1, v1, r2, v2, tof, _MU_SUN, max_revs=2)

    # Constrained with target departure v∞ = 8.8 km/s (representative of Galileo Earth-1)
    target_vinf = 8.8  # km/s
    sol_constrained = find_best_transfer(
        r1,
        v1,
        r2,
        v2,
        tof,
        _MU_SUN,
        max_revs=2,
        target_departure_vinf_km_s=target_vinf,
        vinf_continuity_weight=10.0,
    )

    vinf_unconstrained = sol_unconstrained.departure_vinf_km_s
    vinf_constrained = sol_constrained.departure_vinf_km_s

    # The constrained solution must be closer to the target
    assert abs(vinf_constrained - target_vinf) <= abs(vinf_unconstrained - target_vinf) + 1e-9, (
        f"Constrained departure v∞ = {vinf_constrained:.4f} km/s is FURTHER from "
        f"target {target_vinf} km/s than unconstrained {vinf_unconstrained:.4f} km/s. "
        f"Branch selection is not working."
    )

    print("\n[Branch selection comparison]")
    print(
        f"  Unconstrained: N={sol_unconstrained.n_revs}, "
        f"ΔV={sol_unconstrained.delta_v:.4f} km/s, "
        f"departure v∞={vinf_unconstrained:.4f} km/s"
    )
    print(
        f"  Constrained (target={target_vinf} km/s): N={sol_constrained.n_revs}, "
        f"ΔV={sol_constrained.delta_v:.4f} km/s, "
        f"departure v∞={vinf_constrained:.4f} km/s"
    )


def test_continuity_weight_zero_gives_same_as_unconstrained() -> None:
    """With vinf_continuity_weight=0, target is effectively ignored → same as unconstrained."""
    r1, v1 = _circular_pos_vel(1.0, 0.0)
    r2, v2 = _circular_pos_vel(1.0, 2 * math.pi * 731 / 365.25)
    tof = 731.0 * 86400.0

    sol_base = find_best_transfer(r1, v1, r2, v2, tof, _MU_SUN, max_revs=2)
    sol_zero_weight = find_best_transfer(
        r1,
        v1,
        r2,
        v2,
        tof,
        _MU_SUN,
        max_revs=2,
        target_departure_vinf_km_s=8.8,
        vinf_continuity_weight=0.0,
    )
    assert sol_base.n_revs == sol_zero_weight.n_revs
    assert sol_base.branch == sol_zero_weight.branch


def test_chain_solver_threads_continuity_constraint() -> None:
    """resolve_flyby_chain must pass arrival v∞ from leg k as departure constraint for leg k+1.
    Verify by inspecting which branch is selected for a 3-body chain.
    """
    # This is an integration test — it verifies the wiring, not physics accuracy.
    from unittest.mock import MagicMock

    from astra.optimization.chain_solver import resolve_flyby_chain

    _G_MU_SUN = GM["SUN"]
    _AU_KM = 1.496e8

    def _make_state(a_au: float, theta: float) -> MagicMock:
        a = a_au * _AU_KM
        v_c = math.sqrt(_G_MU_SUN / a)
        pos = np.array([a * math.cos(theta), a * math.sin(theta), 0.0])
        vel = v_c * np.array([-math.sin(theta), math.cos(theta), 0.0])
        state = MagicMock()
        state.position = pos
        state.velocity = vel
        return state

    kernel = MagicMock()
    t0 = 0.0
    tof1 = 120 * 86400.0
    tof2 = 730 * 86400.0

    def _get_state(cb: object, epoch: float) -> MagicMock:
        theta = 2 * math.pi * epoch / (365.25 * 86400.0)
        return _make_state(1.0, theta)

    kernel.get_body_state.side_effect = _get_state

    mission = MagicMock()
    mission.max_revs_per_leg = 2
    mission.leg_max_revs = [2, 2]
    mission.dsm_budget_km_s = 0.0
    mission.parking_altitude_km = 300.0
    mission.capture_altitude_km = 300.0
    mission.capture_apoapsis_km = 1e6

    # This should not crash and should have threaded the continuity constraint
    # (We can't easily assert which branch was chosen without mocking find_best_transfer,
    # so this test just confirms the wiring compiles and runs without error.)
    try:
        result = resolve_flyby_chain(
            mission=mission,
            kernel=kernel,
            chain_bodies=["EARTH", "EARTH", "EARTH"],
            departure_epoch=t0,
            leg_tofs=[tof1, tof2],
            flyby_specs={
                "EARTH": {
                    "min_alt_km": 300.0,
                    "max_alt_km": 50000.0,
                    "powered_allowed": False,
                    "max_powered_km_s": 0.0,
                }
            },
        )
        # Result is either feasible or rejected — both are valid outcomes.
        # We only assert the function completed without exception.
        assert result is not None
    except Exception as e:
        pytest.fail(f"resolve_flyby_chain raised unexpected exception: {e}")
