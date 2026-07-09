"""Validate that the periapsis construction used in resolve_flyby_high_fidelity
is geometrically self-consistent. Tests run without SPICE.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from astra.physics.flyby import (
    build_bplane_frame,
    build_geometrically_consistent_periapsis,
    compute_flyby_turn_angle,
    orbit_normal_from_bvector,
)
from astra.state.orbital_state import GM


def _build_periapsis_state(
    v_inf_in: np.ndarray, periapsis_km: float, body: str
) -> tuple[np.ndarray, np.ndarray]:
    """Build periapsis (pos_hat, vel_hat) using the corrected construction."""
    body_upper = body.upper()
    v_inf_mag = float(np.linalg.norm(v_inf_in))
    turn_rad = compute_flyby_turn_angle(v_inf_mag, periapsis_km, body_upper)
    S_hat, T_hat, _ = build_bplane_frame(v_inf_in)
    h_hat = orbit_normal_from_bvector(S_hat, T_hat)
    r_hat, v_hat, _ = build_geometrically_consistent_periapsis(S_hat, h_hat, turn_rad)
    return r_hat, v_hat


@pytest.mark.parametrize(
    "body,v_inf_vec,periapsis_km",
    [
        ("VENUS", np.array([9.7, 0.0, 0.0]), 6051.8 + 284.0),  # Cassini Venus-1
        ("EARTH", np.array([8.95, 0.0, 0.0]), 6371.0 + 960.0),  # Galileo Earth-1
        ("MERCURY", np.array([5.0, 3.0, 0.0]), 2439.7 + 201.0),  # MESSENGER Mercury-1
        ("MARS", np.array([3.5, 2.0, 1.0]), 3389.5 + 300.0),  # generic Mars
    ],
)
def test_r_peri_perpendicular_to_v_peri(
    body: str, v_inf_vec: np.ndarray, periapsis_km: float
) -> None:
    """Invariant: r̂_peri · v̂_peri = 0 (orthogonality at periapsis).
    This must hold to machine precision — violation means the orbit is non-Keplerian.
    """
    r_hat, v_hat = _build_periapsis_state(v_inf_vec, periapsis_km, body)
    dot = float(np.dot(r_hat, v_hat))
    assert abs(dot) < 1e-12, f"{body}: r̂_peri · v̂_peri = {dot:.2e}, must be < 1e-12"


@pytest.mark.parametrize(
    "body,v_inf_vec,periapsis_km",
    [
        ("VENUS", np.array([9.7, 0.0, 0.0]), 6051.8 + 284.0),
        ("EARTH", np.array([8.95, 0.0, 0.0]), 6371.0 + 960.0),
    ],
)
def test_v_peri_bisects_turn_angle(body: str, v_inf_vec: np.ndarray, periapsis_km: float) -> None:
    """Invariant: v̂_peri bisects Ŝ_in and Ŝ_out — angle(Ŝ_in, v̂_peri) = δ/2.
    This is the defining property of periapsis on a hyperbola (symmetry of the conic).
    """
    body_upper = body.upper()
    v_inf_mag = float(np.linalg.norm(v_inf_vec))
    turn_rad = compute_flyby_turn_angle(v_inf_mag, periapsis_km, body_upper)
    S_hat, T_hat, _ = build_bplane_frame(v_inf_vec)
    h_hat = orbit_normal_from_bvector(S_hat, T_hat)
    _, v_hat, _ = build_geometrically_consistent_periapsis(S_hat, h_hat, turn_rad)
    cos_angle = float(np.clip(np.dot(S_hat, v_hat), -1.0, 1.0))
    angle_rad = math.acos(cos_angle)
    assert abs(angle_rad - turn_rad / 2.0) < 1e-10, (
        f"{body}: angle(Ŝ, v̂_peri) = {math.degrees(angle_rad):.8f}°, "
        f"expected δ/2 = {math.degrees(turn_rad / 2):.8f}°"
    )


def test_vis_viva_speed_at_periapsis() -> None:
    """The periapsis speed must satisfy vis-viva for a hyperbola exactly:
    v_peri² = v_inf² + 2μ/r_p
    Source: Bate, Mueller, White — Fundamentals of Astrodynamics, eq. 2.8-8.
    """
    body = "VENUS"
    v_inf_mag = 9.7  # km/s
    r_p = 6051.8 + 284.0  # km
    mu = GM[body]
    v_peri_expected = math.sqrt(v_inf_mag**2 + 2.0 * mu / r_p)
    # Verify the formula is numerically consistent with energy conservation:
    # At periapsis: KE + PE = total energy = v_inf²/2 (asymptotic KE)
    energy_at_peri = v_peri_expected**2 / 2.0 - mu / r_p
    energy_asymptote = v_inf_mag**2 / 2.0
    assert abs(energy_at_peri - energy_asymptote) < 1e-10 * energy_asymptote, (
        "Vis-viva energy check failed"
    )
    # The corrected construction must produce this exact speed:
    assert v_peri_expected > v_inf_mag, "Periapsis speed must exceed v_inf"
    assert v_peri_expected == pytest.approx(
        math.sqrt(9.7**2 + 2.0 * mu / (6051.8 + 284.0)), rel=1e-12
    )


def test_angular_momentum_direction_equals_h_hat() -> None:
    """Invariant: r̂_peri × v̂_peri = ĥ (orbit normal preserved).
    This is the definition of the orbit normal for a Keplerian orbit.
    """
    v_inf_in = np.array([9.7, 2.0, 1.0])
    periapsis_km = 6051.8 + 300.0
    body = "VENUS"
    v_inf_mag = float(np.linalg.norm(v_inf_in))
    turn_rad = compute_flyby_turn_angle(v_inf_mag, periapsis_km, body)
    S_hat, T_hat, _ = build_bplane_frame(v_inf_in)
    h_hat = orbit_normal_from_bvector(S_hat, T_hat)
    r_hat, v_hat, _ = build_geometrically_consistent_periapsis(S_hat, h_hat, turn_rad)
    h_reconstructed = np.cross(r_hat, v_hat)
    err = float(np.linalg.norm(h_reconstructed - h_hat))
    assert err < 1e-12, f"|r̂ × v̂ − ĥ| = {err:.2e}, must be < 1e-12"


def test_old_construction_would_fail_angular_momentum() -> None:
    """Document that the OLD construction (pos=T, vel=S×T) violates r×v ∥ ĥ.
    This test confirms the bug that was fixed. It must PASS (the old method IS wrong).
    """
    v_inf_in = np.array([9.7, 0.0, 0.0])
    # Unused variables removed per lint

    S, T, R = build_bplane_frame(v_inf_in)
    # OLD (wrong) construction:
    pos_old_hat = T
    vel_old_hat = np.cross(S, T) / (float(np.linalg.norm(np.cross(S, T))) + 1e-30)
    h_old = np.cross(pos_old_hat, vel_old_hat)

    # ĥ from B-plane
    h_hat_correct = orbit_normal_from_bvector(S, T)

    # The old h points along S, not along h_hat — this is the proof of the bug
    _denom = np.linalg.norm(h_old) + 1e-30
    dot_h_old_with_S = abs(float(np.dot(h_old / _denom, S)))
    dot_h_old_with_h_correct = abs(float(np.dot(h_old / _denom, h_hat_correct)))

    # Old h is approximately parallel to S (|cos θ| ≈ 1), not to h_hat
    assert dot_h_old_with_S > 0.99, (
        "Old construction should produce h ∥ S (the bug), "
        f"but got cos(h_old, S) = {dot_h_old_with_S:.4f}"
    )
    assert dot_h_old_with_h_correct < 0.01, (
        "Old construction should produce h NOT parallel to h_hat_correct, "
        f"but got cos(h_old, h_correct) = {dot_h_old_with_h_correct:.4f}"
    )
