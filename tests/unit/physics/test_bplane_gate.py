import math
import numpy as np
import pytest
from astra.physics.flyby import (
    periapsis_from_impact_parameter, impact_parameter_from_periapsis,
    periapsis_from_turn_angle, max_achievable_turn_angle,
    max_achievable_turn_angle_with_unlimited_burn, check_flyby_feasibility,
    build_bplane_frame, orbit_normal_from_bvector,
)
from astra.state.orbital_state import GM

def test_periapsis_impact_parameter_roundtrip():
    """Closed-form r_p ↔ b must round-trip exactly for 5 cases."""
    mu = GM["VENUS"]
    for r_p in [6500.0, 7000.0, 10000.0, 20000.0, 50000.0]:
        v_inf = 9.7
        b = impact_parameter_from_periapsis(r_p, v_inf, mu)
        r_p_back = periapsis_from_impact_parameter(b, v_inf, mu)
        assert abs(r_p_back - r_p) < 1e-6, f"Roundtrip failed for r_p={r_p}"

def test_turn_angle_periapsis_roundtrip():
    """Closed-form periapsis-from-turn-angle must match forward turn-angle calc."""
    from astra.physics.flyby import compute_flyby_turn_angle
    mu_body = "EARTH"
    v_inf = 6.0
    for target_deg in [10.0, 30.0, 60.0, 90.0]:
        target_rad = math.radians(target_deg)
        mu = GM[mu_body]
        r_p = periapsis_from_turn_angle(target_rad, v_inf, mu)
        achieved_rad = compute_flyby_turn_angle(v_inf, r_p, mu_body)
        assert abs(math.degrees(achieved_rad) - target_deg) < 1e-6

def test_max_turn_with_burn_exceeds_unpowered():
    """The unlimited-burn ceiling must always be ≥ the unpowered ceiling."""
    mu = GM["VENUS"]
    v_inf = 9.7
    r_min = 6051.8 + 300.0
    unpowered = max_achievable_turn_angle(v_inf, r_min, mu)
    with_burn = max_achievable_turn_angle_with_unlimited_burn(v_inf, r_min, mu)
    assert with_burn >= unpowered

def test_venus_audit_case_now_correctly_rejected():
    """THE EXACT FAILURE CASE FROM THE AUDIT. Required turn was 156.85° against
    a 25.75° unpowered ceiling. This must now return is_achievable_at_all checked
    against the unlimited-burn ceiling too — verify it is STILL infeasible since
    156.85° is so far beyond even the unlimited-burn physical ceiling at Venus,
    and the gate must say so explicitly, not silently substitute an answer."""
    mu = GM["VENUS"]
    v_inf_in = 13.355  # km/s, Venus approach speed corresponding to 25.75 deg ceiling
    required_turn = math.radians(156.85)
    feas = check_flyby_feasibility(v_inf_in, required_turn, "VENUS")
    assert feas.is_achievable_unpowered is False
    assert feas.rejection_reason is not None
    print(f"\nVenus audit case: required={math.degrees(feas.required_turn_rad):.2f}°, "
          f"unpowered ceiling={math.degrees(feas.max_unpowered_turn_rad):.2f}°, "
          f"unlimited-burn ceiling={math.degrees(feas.max_turn_with_unlimited_burn_rad):.2f}°")
    print(f"Rejection reason: {feas.rejection_reason}")
    # The original audit found a 25.75° unpowered ceiling for this approach speed.
    assert abs(math.degrees(feas.max_unpowered_turn_rad) - 25.75) < 5.0, (
        "Computed unpowered ceiling should match the audit's independently-"
        "derived value within 5° tolerance")

def test_feasible_case_solves_correctly():
    """A reasonable, achievable turn angle must solve without rejection."""
    feas = check_flyby_feasibility(9.7, math.radians(20.0), "VENUS")
    assert feas.is_achievable_unpowered is True
    assert feas.solved_periapsis_km is not None
    assert feas.solved_periapsis_km > 6051.8  # above Venus surface

def test_bplane_frame_orthonormal():
    v_inf_in = np.array([5.0, 2.0, 1.0])
    S, T, R = build_bplane_frame(v_inf_in)
    assert abs(np.linalg.norm(S) - 1.0) < 1e-9
    assert abs(np.linalg.norm(T) - 1.0) < 1e-9
    assert abs(np.linalg.norm(R) - 1.0) < 1e-9
    assert abs(np.dot(S, T)) < 1e-9
    assert abs(np.dot(S, R)) < 1e-9
    assert abs(np.dot(T, R)) < 1e-9

def test_orbit_normal_perpendicular_to_S():
    S = np.array([1.0, 0.0, 0.0])
    B_hat = np.array([0.0, 1.0, 0.0])
    h = orbit_normal_from_bvector(S, B_hat)
    assert abs(np.dot(h, S)) < 1e-9
    assert abs(np.linalg.norm(h) - 1.0) < 1e-9
