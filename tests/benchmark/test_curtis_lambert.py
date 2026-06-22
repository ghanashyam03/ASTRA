"""Curtis Example 5.2 - Lambert BVP analytical validation.

Source: Howard D. Curtis, Orbital Mechanics for Engineering Students,
4th edition, Example 5.2, p. 243.
This is the gold standard analytical test for Lambert solver correctness.
"""

from __future__ import annotations

import math

import numpy as np

from astra.physics.lambert import lambert_izzo
from astra.state.orbital_state import GM


def test_curtis_example_5_2_v1() -> None:
    """Curtis Example 5.2 published Lambert velocity vectors."""
    r1 = np.array([5000.0, 10000.0, 2100.0])
    r2 = np.array([-14600.0, 2500.0, 7000.0])
    tof = 3600.0
    mu = 398600.4418

    v1, v2, conv = lambert_izzo(r1, r2, tof, mu)
    assert conv, "Lambert must converge for Curtis Example 5.2"

    v1_ref = np.array([-5.9925, 1.9254, 3.2456])
    v2_ref = np.array([-3.3125, -4.1966, -0.3853])

    np.testing.assert_allclose(
        v1,
        v1_ref,
        atol=1e-4,
        err_msg=f"v1={v1} does not match Curtis reference {v1_ref}",
    )
    np.testing.assert_allclose(
        v2,
        v2_ref,
        atol=1e-4,
        err_msg=f"v2={v2} does not match Curtis reference {v2_ref}",
    )


def test_curtis_example_5_2_reconstruction() -> None:
    """Propagate with Lambert v1 for tof seconds. Must arrive at r2."""
    from astra.physics.propagator import propagate_two_body
    from astra.state.orbital_state import CelestialBody, OrbitalState, ReferenceFrame

    r1 = np.array([5000.0, 10000.0, 2100.0])
    r2 = np.array([-14600.0, 2500.0, 7000.0])
    tof = 3600.0
    mu = 398600.4418
    v1, _, _ = lambert_izzo(r1, r2, tof, mu)

    state = OrbitalState(
        epoch=0.0,
        position=r1,
        velocity=v1,
        central_body=CelestialBody.EARTH,
        frame=ReferenceFrame.ICRF,
    )
    final = propagate_two_body(state, tof)
    err = float(np.linalg.norm(final.position - r2))
    assert err < 1.0, f"Curtis 5.2 reconstruction error {err:.3f} km > 1.0 km"


def test_lambert_hohmann_earth_mars() -> None:
    """Hohmann transfer approximation: Earth to Mars delta-v from 1 AU."""
    mu = GM["SUN"]
    r1 = np.array([1.496e8, 0.0, 0.0])
    theta = math.radians(179.0)
    r2 = np.array([2.279e8 * math.cos(theta), 2.279e8 * math.sin(theta), 0.0])
    tof = 259 * 86400.0

    v_earth = np.array([0.0, math.sqrt(mu / 1.496e8), 0.0])
    v_mars = np.array(
        [
            -math.sqrt(mu / 2.279e8) * math.sin(theta),
            math.sqrt(mu / 2.279e8) * math.cos(theta),
            0.0,
        ]
    )

    v_dep, v_arr, conv = lambert_izzo(r1, r2, tof, mu)
    assert conv

    dv_dep = float(np.linalg.norm(v_dep - v_earth))
    dv_arr = float(np.linalg.norm(v_mars - v_arr))

    assert 2.0 < dv_dep < 5.0, f"Departure delta-v {dv_dep:.3f} outside expected range"
    assert 1.5 < dv_arr < 5.0, f"Arrival delta-v {dv_arr:.3f} outside expected range"
