"""Exhaustive geometric self-consistency tests for build_geometrically_consistent_periapsis.

Every test targets a specific physical invariant. A failure unambiguously identifies
which invariant is violated, not just that "something is wrong".
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from astra.physics.flyby import (
    build_bplane_frame,
    build_geometrically_consistent_periapsis,
    compute_flyby,
    compute_flyby_from_geometry,
    compute_flyby_turn_angle,
    orbit_normal_from_bvector,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_h_hat(s: np.ndarray, theta_rad: float = 0.0) -> np.ndarray:
    """Build an orbit normal perpendicular to S via B-plane frame."""
    S_hat, T, R = build_bplane_frame(s)
    B_hat = math.cos(theta_rad) * T + math.sin(theta_rad) * R
    return orbit_normal_from_bvector(S_hat, B_hat)


# ── Invariant 1: r_peri ⊥ v_peri ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "v_inf_dir",
    [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.707, 0.707, 0.0]),
        np.array([0.577, 0.577, 0.577]),
        np.array([-1.0, 0.5, 0.3]),
    ],
)
@pytest.mark.parametrize("turn_deg", [10.0, 30.0, 60.0, 90.0, 120.0, 150.0])
def test_r_peri_perpendicular_to_v_peri(v_inf_dir: np.ndarray, turn_deg: float) -> None:
    """Invariant 1: r̂_p · v̂_p = 0 (periapsis vectors are orthogonal)."""
    S_hat = v_inf_dir / np.linalg.norm(v_inf_dir)
    h_hat = _make_h_hat(S_hat, theta_rad=math.pi / 4)
    r_hat, v_hat, _ = build_geometrically_consistent_periapsis(S_hat, h_hat, math.radians(turn_deg))
    dot = float(np.dot(r_hat, v_hat))
    assert abs(dot) < 1e-12, (
        f"Invariant 1 violated: r̂_p · v̂_p = {dot:.2e} for v_inf_dir={v_inf_dir}, turn={turn_deg}°"
    )


# ── Invariant 2: r_peri × v_peri ∥ h_hat ─────────────────────────────────────


@pytest.mark.parametrize("theta_deg", [0.0, 45.0, 90.0, 135.0, 180.0, 270.0, 315.0])
def test_angular_momentum_direction_preserved(theta_deg: float) -> None:
    """Invariant 2: r̂_p × v̂_p = ĥ (angular momentum points along orbit normal)."""
    S_hat = np.array([0.6, 0.8, 0.0])
    S_hat /= np.linalg.norm(S_hat)
    h_hat = _make_h_hat(S_hat, math.radians(theta_deg))
    turn_rad = math.radians(45.0)
    r_hat, v_hat, _ = build_geometrically_consistent_periapsis(S_hat, h_hat, turn_rad)
    h_reconstructed = np.cross(r_hat, v_hat)
    discrepancy = float(np.linalg.norm(h_reconstructed - h_hat))
    assert discrepancy < 1e-12, (
        f"Invariant 2 violated at theta={theta_deg}°: |r̂_p × v̂_p − ĥ| = {discrepancy:.2e}"
    )


# ── Invariant 3: outgoing asymptote lies in orbital plane ─────────────────────


def test_s_out_perpendicular_to_h_hat() -> None:
    """Invariant 3: Ŝ_out · ĥ = 0 (outgoing asymptote stays in orbital plane)."""
    S_hat = np.array([1.0, 0.0, 0.0])
    h_hat = np.array([0.0, 0.0, 1.0])
    for turn_deg in [20.0, 50.0, 80.0, 100.0]:
        _, _, S_out = build_geometrically_consistent_periapsis(S_hat, h_hat, math.radians(turn_deg))
        out_of_plane = abs(float(np.dot(S_out, h_hat)))
        assert out_of_plane < 1e-12, (
            f"Invariant 3 violated at turn={turn_deg}°: |Ŝ_out · ĥ| = {out_of_plane:.2e}"
        )


# ── Invariant 4: turn angle is preserved end-to-end ──────────────────────────


@pytest.mark.parametrize("turn_deg", [5.0, 15.0, 45.0, 75.0, 100.0, 160.0])
def test_s_in_to_s_out_angle_matches_turn(turn_deg: float) -> None:
    """Invariant 4: angle(Ŝ_in, Ŝ_out) = δ (total turn angle preserved)."""
    S_hat = np.array([0.0, 1.0, 0.0])
    h_hat = np.array([0.0, 0.0, 1.0])
    turn_rad = math.radians(turn_deg)
    _, _, S_out = build_geometrically_consistent_periapsis(S_hat, h_hat, turn_rad)
    cos_angle = float(np.clip(np.dot(S_hat, S_out), -1.0, 1.0))
    recovered_deg = math.degrees(math.acos(cos_angle))
    assert abs(recovered_deg - turn_deg) < 1e-9, (
        f"Invariant 4 violated: input δ={turn_deg}°, recovered={recovered_deg:.6f}°"
    )


# ── Invariant 5: v_peri bisects S_in and S_out ───────────────────────────────


def test_v_peri_bisects_turn() -> None:
    """Invariant 5: v̂_p bisects the angle between Ŝ_in and Ŝ_out."""
    S_hat = np.array([1.0, 0.0, 0.0])
    h_hat = np.array([0.0, 0.0, 1.0])
    turn_rad = math.radians(60.0)
    _, v_hat, S_out = build_geometrically_consistent_periapsis(S_hat, h_hat, turn_rad)
    angle_in = math.degrees(math.acos(float(np.clip(np.dot(S_hat, v_hat), -1, 1))))
    angle_out = math.degrees(math.acos(float(np.clip(np.dot(v_hat, S_out), -1, 1))))
    assert abs(angle_in - angle_out) < 1e-9, (
        f"Invariant 5 violated: angle(S_in, v_p)={angle_in:.6f}°, "
        f"angle(v_p, S_out)={angle_out:.6f}°"
    )


# ── Analytic check against known hyperbola (e=2, perifocal frame) ─────────────


def test_analytic_e2_hyperbola() -> None:
    """Cross-check against the e=2 analytic case derived in perifocal coordinates.

    For e=2:
      δ = 2·arcsin(1/2) = 60°
      S_in  in perifocal frame = (1/2,  √3/2, 0) (analytically derived)
      S_out in perifocal frame = (-1/2, √3/2, 0)
      v̂_p   in perifocal frame = (0, 1, 0)
      r̂_p   in perifocal frame = (1, 0, 0)
      ĥ = (0, 0, 1)
    """
    S_in = np.array([0.5, math.sqrt(3) / 2.0, 0.0])
    h_hat = np.array([0.0, 0.0, 1.0])
    turn_rad = 2.0 * math.asin(0.5)  # 60°

    r_hat, v_hat, S_out = build_geometrically_consistent_periapsis(S_in, h_hat, turn_rad)

    assert (
        abs(float(np.dot(v_hat - np.array([0.0, 1.0, 0.0]), v_hat - np.array([0.0, 1.0, 0.0]))))
        < 1e-12
    ), f"v̂_p mismatch: {v_hat}"
    assert (
        abs(float(np.dot(r_hat - np.array([1.0, 0.0, 0.0]), r_hat - np.array([1.0, 0.0, 0.0]))))
        < 1e-12
    ), f"r̂_p mismatch: {r_hat}"
    S_out_expected = np.array([-0.5, math.sqrt(3) / 2.0, 0.0])
    assert float(np.linalg.norm(S_out - S_out_expected)) < 1e-12, (
        f"Ŝ_out mismatch: {S_out} vs {S_out_expected}"
    )


# ── Validation error: non-perpendicular h_hat ────────────────────────────────


def test_raises_on_non_perpendicular_h_hat() -> None:
    """build_geometrically_consistent_periapsis must reject h_hat not ⊥ S_hat."""
    S_hat = np.array([1.0, 0.0, 0.0])
    h_bad = np.array([0.5, 0.0, 0.866])  # NOT perpendicular to S_hat (dot = 0.5)
    with pytest.raises(ValueError, match="must be perpendicular"):
        build_geometrically_consistent_periapsis(S_hat, h_bad, math.radians(30.0))


# ── Validation error: degenerate turn angle ───────────────────────────────────


@pytest.mark.parametrize("bad_turn", [0.0, math.pi, -0.1, math.pi + 0.1])
def test_raises_on_invalid_turn_angle(bad_turn: float) -> None:
    """Turn angle outside (0, π) must be rejected."""
    S_hat = np.array([1.0, 0.0, 0.0])
    h_hat = np.array([0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="turn_angle_rad"):
        build_geometrically_consistent_periapsis(S_hat, h_hat, bad_turn)


# ── compute_flyby_from_geometry agrees with compute_flyby ─────────────────────


def test_compute_flyby_from_geometry_matches_compute_flyby() -> None:
    """compute_flyby_from_geometry must return results consistent with compute_flyby
    when the same orbit normal is passed to both."""
    v_inf_in = np.array([5.0, 2.0, 1.0])
    v_inf_mag = float(np.linalg.norm(v_inf_in))
    S_hat = v_inf_in / v_inf_mag
    periapsis_km = 9500.0
    body = "VENUS"

    # Build h_hat from B-plane
    _, T, R = build_bplane_frame(v_inf_in)
    B_hat = T  # theta = 0
    h_hat = orbit_normal_from_bvector(S_hat, B_hat)

    r1 = compute_flyby(v_inf_in, periapsis_km, body, flyby_plane_normal=h_hat)
    r2 = compute_flyby_from_geometry(S_hat, h_hat, v_inf_mag, periapsis_km, body)

    assert abs(r1.turn_angle_deg - r2.turn_angle_deg) < 1e-6, (
        f"turn_angle mismatch: {r1.turn_angle_deg:.6f}° vs {r2.turn_angle_deg:.6f}°"
    )
    assert abs(r1.dv_helio_km_s - r2.dv_helio_km_s) < 1e-6, (
        f"dv_helio mismatch: {r1.dv_helio_km_s:.6f} vs {r2.dv_helio_km_s:.6f}"
    )


# ── Cassini Venus flyby historical check ──────────────────────────────────────


def test_cassini_venus1_flyby_geometry_self_consistent() -> None:
    """Verify geometric self-consistency for Cassini Venus-1 flyby (1998-04-26).

    Historical parameters: v_inf ≈ 9.7 km/s, periapsis alt ≈ 284 km (r_p ≈ 6335.8 km).
    Turn angle computed from these must round-trip through geometry construction without error.
    """
    body = "VENUS"
    v_inf_km_s = 9.7
    periapsis_km = 6051.8 + 284.0  # physical radius + altitude

    turn_rad = compute_flyby_turn_angle(v_inf_km_s, periapsis_km, body)
    assert turn_rad > 0.0, "Turn angle should be positive for this flyby."

    # Build a representative S_hat and h_hat
    S_hat = np.array([0.8, 0.6, 0.0])
    h_hat = _make_h_hat(S_hat, math.pi / 3)

    r_hat, v_hat, S_out = build_geometrically_consistent_periapsis(S_hat, h_hat, turn_rad)

    # All invariants
    assert abs(float(np.dot(r_hat, v_hat))) < 1e-12, "r̂_p · v̂_p ≠ 0"
    h_check = np.cross(r_hat, v_hat)
    assert float(np.linalg.norm(h_check - h_hat)) < 1e-12, "r̂_p × v̂_p ≠ ĥ"
    cos_turn = float(np.clip(np.dot(S_hat, S_out), -1, 1))
    assert abs(math.degrees(math.acos(cos_turn)) - math.degrees(turn_rad)) < 1e-9, (
        "Turn angle not preserved through geometry construction"
    )
