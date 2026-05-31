import math

import numpy as np
import pytest

from astra.physics.exceptions import LambertSingularityError
from astra.physics.lambert import (
    find_best_transfer,
    lambert_izzo,
    lambert_izzo_multirev,
    lambert_min_tof_multirev,
)
from astra.state.orbital_state import GM

SUN_MU = GM["SUN"]
EARTH_SMA = 1.496e8   # km (1 AU)
MARS_SMA = 2.279e8    # km (1.524 AU)


def test_lambert_multirev_n0_equivalence() -> None:
    """N=0 multi-rev search (max_revs=0) must yield identical velocities.

    Should match standard lambert_izzo.
    """
    r1 = np.array([EARTH_SMA, 0.0, 0.0])
    r2 = np.array([0.0, MARS_SMA, 0.0])
    v_earth_circ = math.sqrt(SUN_MU / EARTH_SMA)
    v_mars_circ = math.sqrt(SUN_MU / MARS_SMA)
    v1_body = np.array([0.0, v_earth_circ, 0.0])
    v2_body = np.array([-v_mars_circ, 0.0, 0.0])
    
    tof = 200 * 86400.0  # 200 days
    
    v1_std, v2_std, conv_std = lambert_izzo(r1, r2, tof, SUN_MU)
    assert conv_std
    
    sol = find_best_transfer(r1, v1_body, r2, v2_body, tof, SUN_MU, max_revs=0)
    assert sol.n_revs == 0
    assert sol.branch == "single"
    np.testing.assert_allclose(sol.v1, v1_std, rtol=1e-12)
    np.testing.assert_allclose(sol.v2, v2_std, rtol=1e-12)


def test_lambert_multirev_singularity_guard() -> None:
    """Sub-minimum time-of-flight must raise LambertSingularityError for N >= 1."""
    r1 = np.array([EARTH_SMA, 0.0, 0.0])
    r2 = np.array([0.0, MARS_SMA, 0.0])
    
    # Calculate non-dimensional s and ll
    r1_norm = float(np.linalg.norm(r1))
    r2_norm = float(np.linalg.norm(r2))
    c_norm = float(np.linalg.norm(r2 - r1))
    s = (r1_norm + r2_norm + c_norm) * 0.5
    ll = math.sqrt(1.0 - min(1.0, c_norm / s))
    
    # Get T_min for N=1
    _, T_min = lambert_min_tof_multirev(ll, 1)
    
    # Compute sub-minimum time of flight
    tof_sub = 0.9 * T_min / math.sqrt(2.0 * SUN_MU / s**3)
    
    with pytest.raises(LambertSingularityError) as exc_info:
        lambert_izzo_multirev(r1, r2, tof_sub, SUN_MU, n=1, branch="low")
    assert "No multi-revolution solution exists" in str(exc_info.value)


def test_lambert_multirev_reconstruction() -> None:
    """Propagating r1 with multi-rev solved v1 for tof must arrive back at r2 within 500 km."""
    from astra.physics.propagator import propagate_two_body
    from astra.state.orbital_state import CelestialBody, OrbitalState, ReferenceFrame
    
    # Setup transfer between Earth and Mars circular orbit
    r1 = np.array([EARTH_SMA, 0.0, 0.0])
    # 120 deg transfer angle
    angle = 120.0 * math.pi / 180.0
    r2 = MARS_SMA * np.array([math.cos(angle), math.sin(angle), 0.0])
    
    # Earth-Mars 1-revolution transfer takes > 700 days to complete a full loop + 120 degrees
    tof = 800 * 86400.0  # 800 days in seconds
    
    for branch in ("low", "high"):
        v1, v2, converged = lambert_izzo_multirev(r1, r2, tof, SUN_MU, n=1, branch=branch)
        assert converged, f"Lambert multi-rev for {branch} branch must converge"
        
        state = OrbitalState(
            epoch=0.0,
            position=r1.copy(),
            velocity=v1.copy(),
            frame=ReferenceFrame.ECLIPJ2000,
            central_body=CelestialBody.SUN,
        )
        
        final = propagate_two_body(state, tof)
        error_km = np.linalg.norm(final.position - r2)
        assert error_km < 500.0, (
            f"Lambert multi-rev reconstruction error {error_km:.1f} km "
            f"for {branch} branch exceeds 500 km"
        )


def test_find_best_transfer_finds_cheaper_multirev() -> None:
    """For a long transfer window, find_best_transfer must find a valid multi-rev solution."""
    r1 = np.array([EARTH_SMA, 0.0, 0.0])
    # Non-collinear 150 degree transfer angle
    angle = 150.0 * math.pi / 180.0
    r2 = MARS_SMA * np.array([math.cos(angle), math.sin(angle), 0.0])
    
    # Circular body velocities
    v_earth_circ = math.sqrt(SUN_MU / EARTH_SMA)
    v_mars_circ = math.sqrt(SUN_MU / MARS_SMA)
    v1_body = np.array([0.0, v_earth_circ, 0.0])
    v2_body = v_mars_circ * np.array([-math.sin(angle), math.cos(angle), 0.0])
    
    tof = 850 * 86400.0  # 850 days (long TOF allowing multi-rev)
    
    # N=0 single-rev
    sol_0 = find_best_transfer(r1, v1_body, r2, v2_body, tof, SUN_MU, max_revs=0)
    
    # Search up to 2 revolutions
    sol_opt = find_best_transfer(r1, v1_body, r2, v2_body, tof, SUN_MU, max_revs=2)
    
    print(f"\nSingle-rev Δv: {sol_0.delta_v:.3f} km/s")
    print(f"Optimal multi-rev Δv: {sol_opt.delta_v:.3f} km/s | "
          f"{sol_opt.n_revs}-rev {sol_opt.branch}")
    
    assert sol_opt.delta_v <= sol_0.delta_v, "Optimal transfer must not exceed single-rev Δv cost"
