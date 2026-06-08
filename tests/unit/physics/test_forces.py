import math

import numpy as np
import pytest

from astra.physics.forces.drag import AtmosphericDrag
from astra.physics.forces.gravity import J2_CONSTANTS, ForceModel, J2Perturbation, PointMassGravity
from astra.physics.forces.srp import SolarRadiationPressure
from astra.physics.propagator import propagate_two_body
from astra.state.orbital_state import (
    GM,
    PHYSICAL_RADIUS,
    CelestialBody,
    OrbitalState,
    ReferenceFrame,
)


def test_force_model_interface() -> None:
    """Verify that ForceModel is an ABC and requires acceleration method."""
    with pytest.raises(TypeError):
        ForceModel()  # type: ignore[abstract]


def test_point_mass_gravity() -> None:
    """Verify PointMassGravity physics and safety guards."""
    mu = GM["EARTH"]
    gravity = PointMassGravity(mu)

    # Standard state
    state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0], dtype=np.float64)
    a = gravity.acceleration(state, 0.0)
    expected = -mu * np.array([7000.0, 0.0, 0.0]) / (7000.0**3)
    assert np.allclose(a, expected)

    # Safety guard: near-zero radius
    zero_state = np.array([1e-7, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    a_zero = gravity.acceleration(zero_state, 0.0)
    assert np.allclose(a_zero, np.zeros(3))


def test_j2_perturbation() -> None:
    """Verify J2 perturbation signs, directions, and safety guards."""
    mu = GM["EARTH"]
    J2 = J2_CONSTANTS["EARTH"]
    R_earth = PHYSICAL_RADIUS[CelestialBody.EARTH]
    j2_model = J2Perturbation(mu, J2, R_earth)

    # 1. Equatorial orbit (z = 0)
    state_eq = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0], dtype=np.float64)
    a_eq = j2_model.acceleration(state_eq, 0.0)
    
    # At z = 0, z-acceleration must be zero
    assert abs(a_eq[2]) < 1e-15
    # Radial perturbation: ax should be in opposite direction of x (pulling inwards)
    factor = 1.5 * J2 * mu * (R_earth**2) / (7000.0**5)
    expected_ax = factor * 7000.0 * (-1.0)
    assert np.allclose(a_eq[0], expected_ax)
    assert a_eq[0] < 0  # Inward pull

    # 2. Polar position (x = 0, y = 0, z = r)
    state_pol = np.array([0.0, 0.0, 7000.0, 7.5, 0.0, 0.0], dtype=np.float64)
    a_pol = j2_model.acceleration(state_pol, 0.0)
    
    # In polar position, x and y accelerations must be zero
    assert abs(a_pol[0]) < 1e-15
    assert abs(a_pol[1]) < 1e-15
    # az should be in same direction of z (pointing outward) due to oblate mass distribution
    expected_az = factor * 7000.0 * (5.0 * 1.0 - 3.0)  # = 2 * factor * z
    assert np.allclose(a_pol[2], expected_az)
    assert a_pol[2] > 0  # Outward perturbation

    # 3. Near-zero radius guard
    zero_state = np.array([0.0, 0.0, 1e-7, 0.0, 0.0, 0.0], dtype=np.float64)
    a_zero = j2_model.acceleration(zero_state, 0.0)
    assert np.allclose(a_zero, np.zeros(3))


def test_solar_radiation_pressure() -> None:
    """Verify SolarRadiationPressure calculations, units, and safety guards."""
    area = 10.0  # m^2
    mass = 1000.0  # kg
    Cr = 1.8
    srp = SolarRadiationPressure(area_m2=area, mass_kg=mass, Cr=Cr)

    # Standard state relative to the Sun (at 1 AU)
    AU = 1.496e8
    state = np.array([AU, 0.0, 0.0, 0.0, 30.0, 0.0], dtype=np.float64)
    a = srp.acceleration(state, 0.0)

    # Hand calculation:
    # pressure = P_solar * (AU/r_mag)^2 = 4.56e-6 * 1 = 4.56e-6 N/m^2
    # force = Cr * area * pressure = 1.8 * 10 * 4.56e-6 = 8.208e-5 N
    # accel (m/s^2) = force / mass = 8.208e-5 / 1000 = 8.208e-8 m/s^2
    # accel (km/s^2) = 8.208e-11 km/s^2
    # Direction is away from Sun (opposite to state position) -> [-8.208e-11, 0, 0]
    expected = np.array([-8.208e-11, 0.0, 0.0])
    assert np.allclose(a, expected, rtol=1e-10)

    # Near-zero radius guard
    zero_state = np.array([0.0, 1e-7, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    a_zero = srp.acceleration(zero_state, 0.0)
    assert np.allclose(a_zero, np.zeros(3))


def test_atmospheric_drag_leo() -> None:
    """Verify AtmosphericDrag units and physically reasonable values in LEO."""
    area = 15.0  # m^2
    mass = 500.0  # kg
    Cd = 2.2
    drag = AtmosphericDrag(area_m2=area, mass_kg=mass, Cd=Cd, body="EARTH")

    # Earth LEO case: 200 km altitude
    # R_earth = 6378.137 km
    # r = 6378.137 + 200 = 6578.137 km
    # Velocity = 7.78 km/s
    r_mag = 6378.137 + 200.0
    v_mag = 7.78
    state = np.array([r_mag, 0.0, 0.0, 0.0, v_mag, 0.0], dtype=np.float64)
    a = drag.acceleration(state, 0.0)

    # Hand calculation:
    # density = rho0 * 1e9 * exp(-altitude / H) = 1.225 * exp(-200 / 8.5)
    # density = 1.225 * 6.0315e-11 = 7.3886e-11 kg/m^3
    # v_m_s = 7780 m/s
    # accel_m_s2 = -0.5 * Cd * (A/m) * density * v_mag^2
    #            = -0.5 * 2.2 * (15/500) * 7.3886e-11 * (7780**2)
    #            = -1.1 * 0.03 * 7.3886e-11 * 60528400 = -1.4759e-4 m/s^2
    # accel_km_s2 = -1.4759e-7 km/s^2
    # Direction should be opposite to velocity vector
    expected_a_y = -1.4759e-7
    assert np.allclose(a[1], expected_a_y, rtol=1e-3)
    assert abs(a[0]) < 1e-15
    assert abs(a[2]) < 1e-15

    # Check physical reasonableness: drag acceleration should be on the order of 1e-7 km/s^2
    assert 1e-8 < abs(a[1]) < 1e-6

    # Verify cutoff height returns 0
    state_high = np.array([6378.137 + 1050.0, 0.0, 0.0, 0.0, v_mag, 0.0], dtype=np.float64)
    a_high = drag.acceleration(state_high, 0.0)
    assert np.allclose(a_high, np.zeros(3))

    # Verify invalid body raises error
    with pytest.raises(ValueError):
        AtmosphericDrag(area_m2=area, mass_kg=mass, body="JUPITER")


def test_propagator_with_forces() -> None:
    """Verify that propagating with custom forces works and matches two-body when forces=None."""
    mu = GM["EARTH"]
    r = 6378.137 + 400.0
    v = math.sqrt(mu / r)
    s0 = OrbitalState(
        epoch=0.0,
        position=np.array([r, 0.0, 0.0]),
        velocity=np.array([0.0, v, 0.0]),
        frame=ReferenceFrame.ICRF,
        central_body=CelestialBody.EARTH,
    )

    # 1. Without forces (standard Keplerian)
    s1_standard = propagate_two_body(s0, dt_seconds=600.0)

    # 2. With PointMassGravity force model
    forces: list[ForceModel] = [PointMassGravity(mu)]
    s1_custom = propagate_two_body(s0, dt_seconds=600.0, forces=forces)

    # Should be identical
    assert np.allclose(s1_standard.position, s1_custom.position)
    assert np.allclose(s1_standard.velocity, s1_custom.velocity)
