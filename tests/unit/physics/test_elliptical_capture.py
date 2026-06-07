import numpy as np
import pytest
from pydantic import ValidationError

from astra.dsl.compiler import compile_mission
from astra.dsl.parser import parse_mission_string
from astra.dsl.schema import OrbitSchema
from astra.physics.maneuvers import arrival_delta_v, circularization_delta_v
from astra.state.orbital_state import GM, PHYSICAL_RADIUS, CelestialBody


def test_circular_capture_backward_compatibility() -> None:
    """Verify arrival_delta_v with apoapsis_km=None matches circular capture bit-for-bit."""
    v_inf = np.array([2.5, 0.0, 0.0])
    alt = 300.0
    body = "MARS"

    # Original math
    mu = GM[body.upper()]
    r_body = PHYSICAL_RADIUS[CelestialBody[body.upper()]]
    r_cap = r_body + alt
    v_cap = np.sqrt(mu / r_cap)
    v_hyp = np.sqrt(2.5**2 + 2.0 * mu / r_cap)
    expected_dv = v_hyp - v_cap

    dv = arrival_delta_v(v_inf, alt, body, apoapsis_km=None)
    assert dv == expected_dv


def test_elliptical_capture_and_apoapsis_interpretation() -> None:
    """Verify elliptical capture math and ensure apoapsis_km is interpreted as radius, not altitude.

    Mars case:
      R_body = 3389.5 km
      mu = 4.282837e4 km^3/s^2
      v_inf = 2.5 km/s
      capture_altitude_km = 300 km -> r_peri = 3689.5 km
      apoapsis_km = 80000 km (radius from center)
      a_capture = (r_peri + r_apo) / 2 = 41844.75 km
      v_peri_ellipse = sqrt(mu * (2/r_peri - 1/a_capture)) = 4.710928 km/s
      v_hyp = sqrt(v_inf^2 + 2*mu/r_peri) = 5.428292 km/s
      dv = v_hyp - v_peri_ellipse = 0.717363 km/s

      If apoapsis_km was altitude, r_apo would be 83389.5 km, yielding dv = 0.713137 km/s.
    """
    v_inf = np.array([2.5, 0.0, 0.0])
    alt = 300.0
    body = "MARS"
    apo_radius = 80000.0

    dv = arrival_delta_v(v_inf, alt, body, apoapsis_km=apo_radius)

    expected_dv = 0.7173634622894802  # Computed using radius interpretation
    incorrect_alt_dv = 0.7131366  # Computed using altitude interpretation

    assert np.allclose(dv, expected_dv, rtol=1e-10)
    assert not np.allclose(dv, incorrect_alt_dv, rtol=1e-10)


def test_circularization_maneuver() -> None:
    """Verify circularization_delta_v math.

    Mars case:
      r_peri = 3689.5 km
      r_apo = 80000 km
      a = 41844.75 km
      v_apo_ellipse = sqrt(mu * (2/r_apo - 1/a)) = 0.217262 km/s
      v_circular = sqrt(mu / r_apo) = 0.731679 km/s
      dv = v_circular - v_apo_ellipse = 0.514417 km/s
    """
    dv = circularization_delta_v(
        capture_periapsis_km=300.0,
        capture_apoapsis_km=80000.0,
        body="MARS",
    )
    expected_dv = 0.5144171808701444
    assert np.allclose(dv, expected_dv, rtol=1e-10)



def test_orbit_schema_validation() -> None:
    """Verify OrbitSchema model validator logic."""
    # 1. Circular requires altitude
    with pytest.raises(ValidationError):
        OrbitSchema(type="circular")

    # 2. Elliptical requires both periapsis and apoapsis
    with pytest.raises(ValidationError):
        OrbitSchema(type="elliptical", periapsis_km=300.0)

    # 3. Elliptical requires apoapsis > periapsis
    with pytest.raises(ValidationError):
        OrbitSchema(type="elliptical", periapsis_km=5000.0, apoapsis_km=4000.0)
    with pytest.raises(ValidationError):
        OrbitSchema(type="elliptical", periapsis_km=5000.0, apoapsis_km=5000.0)

    # 4. Valid cases
    circ = OrbitSchema(type="circular", altitude_km=300.0)
    assert circ.altitude_km == 300.0

    ellip = OrbitSchema(type="elliptical", periapsis_km=300.0, apoapsis_km=80000.0)
    assert ellip.periapsis_km == 300.0
    assert ellip.apoapsis_km == 80000.0

    # 5. Hyperbolic passes validation (no fields checked)
    hyp = OrbitSchema(type="hyperbolic")
    assert hyp.type == "hyperbolic"


def test_compile_mission_extracts_apoapsis() -> None:
    """Verify that compiler compiles mission with capture_apoapsis_km correctly."""
    dsl_text_elliptical = """
version: "1.0"
mission_id: test_elliptical
spacecraft:
  name: TestCraft
  dry_mass_kg: 1000.0
  fuel_mass_kg: 500.0
  propulsion:
    type: chemical
    isp_seconds: 310.0
trajectory:
  origin:
    body: Earth
    orbit:
      type: circular
      altitude_km: 200.0
  destination:
    body: Mars
    orbit:
      type: elliptical
      periapsis_km: 300.0
      apoapsis_km: 80000.0
launch_window:
  start: "2030-01-01T00:00:00Z"
  end: "2031-01-01T00:00:00Z"
  tof_min_days: 90
  tof_max_days: 300
objectives:
  - metric: delta_v_total
    direction: minimize
"""
    m = parse_mission_string(dsl_text_elliptical)
    compiled = compile_mission(m)
    assert compiled.capture_altitude_km == 300.0
    assert compiled.capture_apoapsis_km == 80000.0
