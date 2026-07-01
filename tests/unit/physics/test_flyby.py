import math

import numpy as np

from astra.physics.flyby import (
    compute_flyby,
    compute_flyby_turn_angle,
)


def test_venus_flyby_turn_angle() -> None:
    """Venus flyby with 5 km/s v_inf at 6500 km periapsis."""
    angle_rad = compute_flyby_turn_angle(5.0, 6500.0, "VENUS")
    angle_deg = math.degrees(angle_rad)
    # μ_Venus = 324859, e = 1 + 6500*25/324859 ≈ 1.500
    # arcsin(1/1.5) ≈ 41.8°, δ = 83.6°
    assert 60.0 < angle_deg < 100.0, f"Venus turn angle {angle_deg:.1f}° unexpected"


def test_flyby_conserves_speed_unpowered() -> None:
    """Unpowered flyby must conserve |v_inf| exactly."""
    v_inf_in = np.array([4.0, 1.0, 0.5])
    result = compute_flyby(v_inf_in, periapsis_km=8000.0, body="EARTH", powered_dv_km_s=0.0)
    assert abs(result.v_inf_out_km_s - result.v_inf_in_km_s) < 1e-6, (
        "Unpowered flyby must conserve |v_inf|"
    )


def test_powered_flyby_increases_vinf() -> None:
    """Powered flyby with 0.5 km/s burn must increase |v_inf_out|."""
    v_inf_in = np.array([3.0, 0.0, 0.0])
    r_unpow = compute_flyby(v_inf_in, 7000.0, "EARTH", powered_dv_km_s=0.0)
    r_pow = compute_flyby(v_inf_in, 7000.0, "EARTH", powered_dv_km_s=0.5)
    assert r_pow.v_inf_out_km_s > r_unpow.v_inf_out_km_s, "Powered flyby must increase |v_inf_out|"


def test_flyby_below_safe_altitude_invalid() -> None:
    """Flyby periapsis below safe altitude must return is_valid=False."""
    v_inf_in = np.array([5.0, 0.0, 0.0])
    result = compute_flyby(v_inf_in, periapsis_km=3390.0, body="MARS")  # below surface
    assert result.is_valid is False


def test_flyby_above_safe_altitude_valid() -> None:
    v_inf_in = np.array([3.0, 0.0, 0.0])
    result = compute_flyby(v_inf_in, periapsis_km=4000.0, body="MARS")  # above surface+margin
    assert result.is_valid is True


def test_higher_vinf_smaller_turn_angle() -> None:
    """Faster approach = less deflection for same periapsis."""
    r1 = compute_flyby_turn_angle(3.0, 7000.0, "EARTH")
    r2 = compute_flyby_turn_angle(10.0, 7000.0, "EARTH")
    assert r1 > r2, "Higher v_inf should produce smaller turn angle"


def test_build_geometrically_consistent_periapsis() -> None:
    from astra.physics.flyby import (
        build_geometrically_consistent_periapsis,
    )

    S_hat = np.array([1.0, 0.0, 0.0])
    h_hat = np.array([0.0, 0.0, 1.0])
    turn_angle = 1.0

    r_peri, v_peri, S_out = build_geometrically_consistent_periapsis(S_hat, h_hat, turn_angle)

    # Unit norms
    assert abs(np.linalg.norm(r_peri) - 1.0) < 1e-12
    assert abs(np.linalg.norm(v_peri) - 1.0) < 1e-12
    assert abs(np.linalg.norm(S_out) - 1.0) < 1e-12

    # Orthogonality
    assert abs(np.dot(r_peri, v_peri)) < 1e-12
    assert abs(np.dot(r_peri, h_hat)) < 1e-12
    assert abs(np.dot(v_peri, h_hat)) < 1e-12

    # Cross product (angular momentum direction)
    assert np.linalg.norm(np.cross(r_peri, v_peri) - h_hat) < 1e-9

    # S_out in plane
    assert abs(np.dot(S_out, h_hat)) < 1e-9

    # Input validation: non-orthogonal
    h_bad = np.array([0.1, 0.0, 1.0])
    try:
        build_geometrically_consistent_periapsis(S_hat, h_bad, turn_angle)
        assert False, "Should have raised ValueError for non-orthogonal h_hat"
    except ValueError:
        pass

    # Input validation: turn angle >= pi
    try:
        build_geometrically_consistent_periapsis(S_hat, h_hat, math.pi)
        assert False, "Should have raised ValueError for turn_angle >= pi"
    except ValueError:
        pass


def test_compute_flyby_from_geometry_compatibility() -> None:
    from astra.physics.flyby import compute_flyby_from_geometry

    S_hat = np.array([0.70710678, -0.70710678, 0.0])
    h_hat = np.array([0.70710678, 0.70710678, 0.0])
    v_inf = 8.0
    peri_km = 7000.0
    body = "EARTH"

    # Unpowered comparison
    res_geom = compute_flyby_from_geometry(S_hat, h_hat, v_inf, peri_km, body)
    res_orig = compute_flyby(S_hat * v_inf, peri_km, body, flyby_plane_normal=h_hat)

    assert abs(res_geom.turn_angle_deg - res_orig.turn_angle_deg) < 1e-6
    assert abs(res_geom.v_inf_out_km_s - res_orig.v_inf_out_km_s) < 1e-6
    assert abs(res_geom.dv_helio_km_s - res_orig.dv_helio_km_s) < 1e-6

    # Powered comparison
    res_geom_p = compute_flyby_from_geometry(
        S_hat, h_hat, v_inf, peri_km, body, powered_dv_km_s=1.0
    )
    res_orig_p = compute_flyby(
        S_hat * v_inf, peri_km, body, powered_dv_km_s=1.0, flyby_plane_normal=h_hat
    )

    assert abs(res_geom_p.turn_angle_deg - res_orig_p.turn_angle_deg) < 1e-6
    assert abs(res_geom_p.v_inf_out_km_s - res_orig_p.v_inf_out_km_s) < 1e-6
    assert abs(res_geom_p.dv_helio_km_s - res_orig_p.dv_helio_km_s) < 1e-6
