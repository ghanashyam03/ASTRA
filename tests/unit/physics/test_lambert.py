import math

import numpy as np
from astra.physics.lambert import lambert_izzo
from astra.state.orbital_state import GM

SUN_MU = GM["SUN"]
EARTH_SMA = 1.496e8  # km (1 AU)
MARS_SMA = 2.279e8  # km (1.524 AU)


def test_lambert_circular_to_circular() -> None:
    """Hohmann-like: Earth circular orbit → Mars circular orbit."""
    r1 = np.array([EARTH_SMA, 0.0, 0.0])
    r2 = np.array([0.0, MARS_SMA, 0.0])  # 90° transfer
    tof = 200 * 86400.0  # 200 days in seconds
    v1, v2, converged = lambert_izzo(r1, r2, tof, SUN_MU)
    assert converged, "Lambert solver must converge"
    assert v1.shape == (3,), "v1 must be 3-vector"
    assert v2.shape == (3,), "v2 must be 3-vector"
    # Departure Δv from 1 AU circular orbit ≈ 14.8 km/s
    v_earth_circ = math.sqrt(SUN_MU / EARTH_SMA)
    earth_vel = np.array([0.0, v_earth_circ, 0.0])
    dv1 = np.linalg.norm(v1 - earth_vel)
    assert 10.0 < dv1 < 20.0, f"Departure Δv {dv1:.3f} km/s out of expected range"


def test_lambert_converges_edge_cases() -> None:
    """Lambert must converge for 1000 random geometry cases."""
    rng = np.random.default_rng(seed=0)
    failures = []
    for i in range(1000):
        r1 = rng.uniform(1e7, 3e8, 3)
        angle = rng.uniform(0.1, 3.0)
        r2_norm = rng.uniform(1e7, 3e8)
        r2 = r2_norm * np.array([math.cos(angle), math.sin(angle), 0.0])
        tof = rng.uniform(30, 400) * 86400.0
        _, _, conv = lambert_izzo(r1, r2, tof, SUN_MU)
        if not conv:
            failures.append(i)
    failure_rate = len(failures) / 1000
    assert failure_rate < 0.001, f"Lambert failure rate {failure_rate:.1%} exceeds 0.1%"


def test_lambert_reconstruction() -> None:
    """Propagate r1 with solved v1 for tof → must arrive at r2."""
    from astra.physics.propagator import propagate_two_body
    from astra.state.orbital_state import CelestialBody, OrbitalState, ReferenceFrame

    r1 = np.array([EARTH_SMA, 0.0, 0.0])
    r2 = np.array([0.0, MARS_SMA, 0.0])
    tof = 200 * 86400.0
    v1, v2, _ = lambert_izzo(r1, r2, tof, SUN_MU)
    state = OrbitalState(
        epoch=0.0,
        position=r1,
        velocity=v1,
        frame=ReferenceFrame.ECLIPJ2000,
        central_body=CelestialBody.SUN,
    )
    final = propagate_two_body(state, tof)
    error_km = np.linalg.norm(final.position - r2)
    assert error_km < 100.0, f"Lambert reconstruction error {error_km:.1f} km (must be < 100 km)"
