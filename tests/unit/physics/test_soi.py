import math

import numpy as np
from astra.physics.maneuvers import arrival_delta_v, c3_from_vinf, departure_delta_v
from astra.physics.soi import compute_soi_radius, is_in_soi


def test_earth_soi_radius() -> None:
    r = compute_soi_radius("EARTH")
    # Published Earth SOI: ~925,000 km
    assert 800_000 < r < 1_100_000, f"Earth SOI {r:.0f} km outside expected range"


def test_mars_soi_radius() -> None:
    r = compute_soi_radius("MARS")
    # Published Mars SOI: ~577,000 km
    assert 450_000 < r < 700_000, f"Mars SOI {r:.0f} km outside expected range"


def test_departure_dv_from_leo() -> None:
    """TMI Δv from 200km LEO with 3.5 km/s v_inf should be ~3.6 km/s."""
    v_inf = np.array([3.5, 0.0, 0.0])
    dv = departure_delta_v(v_inf, parking_altitude_km=200.0, body="EARTH")
    # Analytical check: v_park = sqrt(398600 / 6571) ≈ 7.784 km/s
    # v_hyp = sqrt(3.5² + 2*398600/6571) ≈ 11.30 km/s
    # Δv = 11.30 - 7.784 ≈ 3.52 km/s
    assert 3.0 < dv < 4.5, f"Departure Δv {dv:.4f} km/s outside expected range"


def test_arrival_dv_at_mars() -> None:
    """MOI Δv at Mars 300km orbit with 2.5 km/s arrival v_inf."""
    v_inf = np.array([2.5, 0.0, 0.0])
    dv = arrival_delta_v(v_inf, capture_altitude_km=300.0, body="MARS")
    # μ_Mars = 42828 km³/s², R_Mars = 3389.5 km, r_cap = 3689.5 km
    # v_cap = sqrt(42828 / 3689.5) ≈ 3.408 km/s
    # v_hyp = sqrt(2.5² + 2*42828/3689.5) ≈ sqrt(6.25 + 23.22) ≈ 5.453 km/s
    # MOI = 5.453 - 3.408 ≈ 2.045 km/s
    assert 1.5 < dv < 3.0, f"Arrival Δv {dv:.4f} km/s outside expected range"


def test_c3_matches_vinf_squared() -> None:
    v_inf = np.array([3.5, 1.0, 0.5])
    c3 = c3_from_vinf(v_inf)
    assert abs(c3 - (3.5**2 + 1.0**2 + 0.5**2)) < 1e-10


def test_soi_patching_increases_delta_v() -> None:
    """SOI-patched Δv must exceed heliocentric Δv (parking orbit adds cost)."""
    from astra.physics.lambert import lambert_izzo
    from astra.state.orbital_state import GM

    MU = GM["SUN"]
    EARTH_R, MARS_R = 1.496e8, 2.279e8
    r1 = np.array([EARTH_R, 0.0, 0.0])
    angle = 179.0 * math.pi / 180.0
    r2 = MARS_R * np.array([math.cos(angle), math.sin(angle), 0.0])
    v1_body = np.array([0.0, math.sqrt(MU / EARTH_R), 0.0])
    v2_body = math.sqrt(MU / MARS_R) * np.array([-math.sin(angle), math.cos(angle), 0.0])
    v_dep, v_arr, _ = lambert_izzo(r1, r2, 258.5 * 86400.0, MU)
    # Heliocentric Δv
    dv_helio = float(np.linalg.norm(v_dep - v1_body) + np.linalg.norm(v2_body - v_arr))
    # SOI-patched Δv
    dv_tmi = departure_delta_v(v_dep - v1_body, 200.0, "EARTH")
    dv_moi = arrival_delta_v(v2_body - v_arr, 300.0, "MARS")
    dv_soi = dv_tmi + dv_moi
    assert dv_soi > dv_helio, f"SOI Δv {dv_soi:.4f} must exceed heliocentric {dv_helio:.4f} km/s"
    # Difference should be positive
    delta = dv_soi - dv_helio
    assert 0.0 < delta < 2.0, f"SOI overhead {delta:.4f} km/s seems wrong"


def test_is_in_soi() -> None:
    """Test standard SOI boundary logic."""
    r_sc = np.array([EARTH_SMA := 1.496e8, 500_000.0, 0.0])
    r_earth = np.array([EARTH_SMA, 0.0, 0.0])
    assert is_in_soi(r_sc, r_earth, "EARTH")

    r_sc_far = np.array([EARTH_SMA, 2_000_000.0, 0.0])
    assert not is_in_soi(r_sc_far, r_earth, "EARTH")
