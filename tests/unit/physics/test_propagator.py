import math

import numpy as np
from astra.physics.propagator import propagate_two_body
from astra.state.orbital_state import GM, CelestialBody, OrbitalState, ReferenceFrame


def iss_state() -> OrbitalState:
    mu = GM["EARTH"]
    r = 6371.0 + 408.0  # ISS altitude ~408 km
    v = math.sqrt(mu / r)
    return OrbitalState(
        epoch=0.0,
        position=np.array([r, 0.0, 0.0]),
        velocity=np.array([0.0, v, 0.0]),
        frame=ReferenceFrame.ICRF,
        central_body=CelestialBody.EARTH,
    )


def test_energy_conservation() -> None:
    """Specific energy must be conserved to 1e-8 relative error."""
    s0 = iss_state()
    s1 = propagate_two_body(s0, dt_seconds=5400.0)  # ~1 orbit
    rel_err = abs(s1.specific_energy - s0.specific_energy) / abs(s0.specific_energy)
    assert rel_err < 1e-8, f"Energy drift: {rel_err:.2e}"


def test_angular_momentum_conservation() -> None:
    s0 = iss_state()
    s1 = propagate_two_body(s0, dt_seconds=5400.0)
    h0 = s0.specific_angular_momentum
    h1 = s1.specific_angular_momentum
    rel_err = np.linalg.norm(h1 - h0) / np.linalg.norm(h0)
    assert rel_err < 1e-9, f"Angular momentum drift: {rel_err:.2e}"


def test_circular_orbit_period() -> None:
    """After one Keplerian period the spacecraft returns to start."""
    s0 = iss_state()
    mu = GM["EARTH"]
    T = 2.0 * math.pi * math.sqrt(s0.semi_major_axis**3 / mu)
    s1 = propagate_two_body(s0, dt_seconds=T)
    pos_err = np.linalg.norm(s1.position - s0.position)
    assert pos_err < 0.1, f"Position after 1 period: {pos_err:.3f} km (must be < 100 m)"
