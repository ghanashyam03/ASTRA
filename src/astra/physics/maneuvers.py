"""Impulsive maneuver computation using correct sphere-of-influence patching.

All functions operate in the two-body approximation within each SOI.
"""
from __future__ import annotations

import math

import numpy as np

from astra.state.orbital_state import GM


def departure_delta_v(
    v_inf_departure: np.ndarray,
    parking_altitude_km: float,
    body: str = "EARTH",
) -> float:
    """Compute TMI (Trans-Mars Injection) Δv from a circular parking orbit.

    Physics:
      r_park = R_body + h_park
      v_park = sqrt(μ / r_park)           [circular orbit speed]
      v_inf  = ||v_inf_departure||         [hyperbolic excess speed]
      v_hyp  = sqrt(v_inf² + 2μ/r_park)  [speed at periapsis of departure hyperbola]
      Δv_TMI = v_hyp - v_park

    Parameters
    ----------
    v_inf_departure : heliocentric departure excess velocity vector [km/s]
                      = v_spacecraft_helio - v_body_helio at departure
    parking_altitude_km : altitude of circular parking orbit [km]
    body : central body name (must be in GM dict)

    Returns
    -------
    Δv_TMI in km/s
    """
    from astra.state.orbital_state import PHYSICAL_RADIUS, CelestialBody
    mu = GM[body.upper()]
    r_body = PHYSICAL_RADIUS[CelestialBody[body.upper()]]
    r_park = r_body + parking_altitude_km
    v_park = math.sqrt(mu / r_park)
    v_inf_mag = float(np.linalg.norm(v_inf_departure))
    v_hyp = math.sqrt(v_inf_mag**2 + 2.0 * mu / r_park)
    return v_hyp - v_park


def arrival_delta_v(
    v_inf_arrival: np.ndarray,
    capture_altitude_km: float,
    body: str = "MARS",
) -> float:
    """Compute MOI (Mars Orbit Insertion) Δv to reach a circular capture orbit.

    Physics:
      r_cap  = R_body + h_capture
      v_cap  = sqrt(μ / r_cap)             [circular capture orbit speed]
      v_inf  = ||v_inf_arrival||            [hyperbolic excess speed at arrival]
      v_hyp  = sqrt(v_inf² + 2μ/r_cap)    [speed at periapsis of arrival hyperbola]
      Δv_MOI = v_hyp - v_cap

    Parameters
    ----------
    v_inf_arrival : heliocentric arrival excess velocity vector [km/s]
                    = v_body_helio - v_spacecraft_helio at arrival
    capture_altitude_km : altitude of target circular orbit [km]
    body : destination body name

    Returns
    -------
    Δv_MOI in km/s (positive — always a deceleration)
    """
    from astra.state.orbital_state import PHYSICAL_RADIUS, CelestialBody
    mu = GM[body.upper()]
    r_body = PHYSICAL_RADIUS[CelestialBody[body.upper()]]
    r_cap = r_body + capture_altitude_km
    v_cap = math.sqrt(mu / r_cap)
    v_inf_mag = float(np.linalg.norm(v_inf_arrival))
    v_hyp = math.sqrt(v_inf_mag**2 + 2.0 * mu / r_cap)
    return v_hyp - v_cap


def c3_from_vinf(v_inf: np.ndarray) -> float:
    """C3 = v_inf · v_inf [km²/s²]. Launch vehicle performance metric."""
    return float(np.dot(v_inf, v_inf))


def hyperbolic_excess_speed(v_sc_helio: np.ndarray, v_body_helio: np.ndarray) -> float:
    """||v_inf|| = ||v_spacecraft - v_body|| [km/s]."""
    return float(np.linalg.norm(v_sc_helio - v_body_helio))
