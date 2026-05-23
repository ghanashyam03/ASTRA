import numpy as np
import pytest
from astra.state.orbital_state import OrbitalState, CelestialBody, ReferenceFrame, GM

def circular_orbit_state(altitude_km: float) -> OrbitalState:
    """Circular orbit around Earth at given altitude."""
    mu = GM["EARTH"]
    r = 6371.0 + altitude_km  # km
    v_circular = (mu / r) ** 0.5  # km/s
    return OrbitalState(
        epoch=0.0,
        position=np.array([r, 0.0, 0.0]),
        velocity=np.array([0.0, v_circular, 0.0]),
        frame=ReferenceFrame.ICRF,
        central_body=CelestialBody.EARTH,
    )

def test_shape_validation():
    s = circular_orbit_state(400.0)
    assert s.position.shape == (3,)
    assert s.velocity.shape == (3,)

def test_specific_energy_circular_orbit():
    """For circular orbit: ε = -μ/(2a) = -μ/(2r)."""
    s = circular_orbit_state(400.0)
    mu = GM["EARTH"]
    expected = -mu / (2.0 * s.r)
    assert abs(s.specific_energy - expected) < 1e-6

def test_eccentricity_circular_orbit():
    """Circular orbit must have eccentricity ≈ 0."""
    s = circular_orbit_state(400.0)
    assert abs(s.eccentricity) < 1e-8

def test_semi_major_axis_circular_orbit():
    """SMA equals radius for circular orbit."""
    altitude = 400.0
    s = circular_orbit_state(altitude)
    expected_r = 6371.0 + altitude
    assert abs(s.semi_major_axis - expected_r) < 0.01  # within 10 meters

def test_delta_v_budget_tsiolkovsky():
    from astra.state.spacecraft import Spacecraft, PropulsionSystem, PropulsionType
    prop = PropulsionSystem(
        type=PropulsionType.CHEMICAL,
        isp_seconds=450.0,
        thrust_newtons=22000.0,
        propellant_mass_kg=2400.0,
    )
    sc = Spacecraft(name="TestCraft", dry_mass_kg=1800.0, propulsion=prop)
    import math
    expected = 450.0 * 9.80665e-3 * math.log(4200.0 / 1800.0)
    assert abs(sc.delta_v_budget() - expected) < 1e-6
