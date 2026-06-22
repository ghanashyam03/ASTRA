"""Impulsive maneuver computation using correct sphere-of-influence patching.

All functions operate in the two-body approximation within each SOI.

Heliocentric vs Mission Delta-V:
--------------------------------
- Heliocentric Delta-V (sum of hyperbolic excess velocities v_inf) is the asymptotic
  velocity difference relative to the planet's orbital velocity in the heliocentric frame.
  This represents the delta-v required if the spacecraft were entering/leaving the SOI
  without taking advantage of the planet's gravitational well (no Oberth effect).
- Mission Delta-V (SOI-patched) accounts for the gravity of the departure and arrival bodies
  via hyperbolic patch-conics formulas. It computes the impulsive burn at the periapsis
  of the hyperbola within the planetary sphere of influence (SOI), using the Oberth effect.

Elliptical Capture MOI:
-----------------------
- The arrival delta-v (MOI) for an elliptical capture orbit is calculated as the periapsis
  insertion burn only. This decelerates the spacecraft from the hyperbolic approach trajectory
  into the capture ellipse at its periapsis.
- Circularization of this ellipse into a circular orbit at apoapsis requires a second, separate
  apoapsis kick burn, which is computed using circularization_delta_v.
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
    apoapsis_km: float | None = None,
) -> float:
    """Compute MOI (Mars Orbit Insertion) Δv to reach a circular or elliptical capture orbit.

    For circular capture (apoapsis_km is None), the burn decelerates the spacecraft
    to circular velocity at the capture altitude.
    For elliptical capture (apoapsis_km is provided), the burn inserts the spacecraft
    into the capture ellipse at periapsis. Note that apoapsis_km is the radius from the
    body center, not the altitude.

    Physics when apoapsis_km is None (circular, existing behavior — unchanged):
      r_cap = R_body + capture_altitude_km
      v_cap = sqrt(μ / r_cap)
      v_hyp = sqrt(v_inf² + 2μ/r_cap)
      dv = v_hyp - v_cap

    Physics when apoapsis_km is provided (elliptical capture):
      r_peri = R_body + capture_altitude_km  (periapsis radius)
      r_apo = apoapsis_km                    (apoapsis radius, already from center)
      a_capture = (r_peri + r_apo) / 2.0    (semi-major axis of capture ellipse)
      v_peri_ellipse = sqrt(μ * (2/r_peri - 1/a_capture))
        (speed at periapsis of the capture ellipse via vis-viva)
      v_hyp = sqrt(v_inf² + 2μ/r_peri)
        (speed at periapsis of the incoming hyperbola)
      dv = v_hyp - v_peri_ellipse
        (decelerate from hyperbola to ellipse periapsis speed)

    Parameters
    ----------
    v_inf_arrival : heliocentric arrival excess velocity vector [km/s]
                    = v_body_helio - v_spacecraft_helio at arrival
    capture_altitude_km : altitude of target periapsis orbit [km]
    body : destination body name
    apoapsis_km : apoapsis radius from body center [km] (if elliptical), or None (if circular)

    Returns
    -------
    Δv_MOI in km/s (positive — always a deceleration)
    """
    from astra.state.orbital_state import PHYSICAL_RADIUS, CelestialBody

    mu = GM[body.upper()]
    r_body = PHYSICAL_RADIUS[CelestialBody[body.upper()]]
    r_peri = r_body + capture_altitude_km
    v_inf_mag = float(np.linalg.norm(v_inf_arrival))
    v_hyp = math.sqrt(v_inf_mag**2 + 2.0 * mu / r_peri)

    if apoapsis_km is None:
        v_cap = math.sqrt(mu / r_peri)
        return v_hyp - v_cap
    else:
        r_apo = float(apoapsis_km)
        a_capture = (r_peri + r_apo) / 2.0
        v_peri_ellipse = math.sqrt(mu * (2.0 / r_peri - 1.0 / a_capture))
        return v_hyp - v_peri_ellipse


def circularization_delta_v(
    capture_periapsis_km: float,
    capture_apoapsis_km: float,
    body: str,
) -> float:
    """Compute the apoapsis kick burn to circularize from the capture ellipse.

    Parameters
    ----------
    capture_periapsis_km : float
        Altitude of periapsis of capture ellipse [km] (above body surface).
    capture_apoapsis_km : float
        Radius of apoapsis from body center [km].
    body : str
        Central body name.

    Returns
    -------
    float
        Circularization Δv in km/s.
    """
    from astra.state.orbital_state import PHYSICAL_RADIUS, CelestialBody

    mu = GM[body.upper()]
    r_body = PHYSICAL_RADIUS[CelestialBody[body.upper()]]
    r_peri = r_body + capture_periapsis_km
    r_apo = capture_apoapsis_km
    a = (r_peri + r_apo) / 2.0
    v_apo_ellipse = math.sqrt(mu * (2.0 / r_apo - 1.0 / a))
    v_circular = math.sqrt(mu / r_apo)
    return v_circular - v_apo_ellipse


def c3_from_vinf(v_inf: np.ndarray) -> float:
    """C3 = v_inf · v_inf [km²/s²]. Launch vehicle performance metric."""
    return float(np.dot(v_inf, v_inf))


def hyperbolic_excess_speed(v_sc_helio: np.ndarray, v_body_helio: np.ndarray) -> float:
    """||v_inf|| = ||v_spacecraft - v_body|| [km/s]."""
    return float(np.linalg.norm(v_sc_helio - v_body_helio))
