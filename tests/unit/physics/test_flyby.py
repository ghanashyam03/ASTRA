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
    assert abs(result.v_inf_out_km_s - result.v_inf_in_km_s) < 1e-6, \
        "Unpowered flyby must conserve |v_inf|"

def test_powered_flyby_increases_vinf() -> None:
    """Powered flyby with 0.5 km/s burn must increase |v_inf_out|."""
    v_inf_in = np.array([3.0, 0.0, 0.0])
    r_unpow = compute_flyby(v_inf_in, 7000.0, "EARTH", powered_dv_km_s=0.0)
    r_pow   = compute_flyby(v_inf_in, 7000.0, "EARTH", powered_dv_km_s=0.5)
    assert r_pow.v_inf_out_km_s > r_unpow.v_inf_out_km_s, \
        "Powered flyby must increase |v_inf_out|"

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
