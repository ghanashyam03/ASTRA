"""Powered and unpowered planetary flyby (gravity assist) physics.
All computations use the patched-conics model within the planet's SOI.
Reference: Vallado, D.A. — Fundamentals of Astrodynamics and Applications, Ch. 6.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from astra.state.orbital_state import GM, PHYSICAL_RADIUS, CelestialBody


@dataclass
class FlybyResult:
    body: str
    periapsis_km: float              # closest approach distance from center
    periapsis_altitude_km: float     # closest approach altitude
    v_inf_in_km_s: float             # incoming hyperbolic excess speed
    v_inf_out_km_s: float            # outgoing hyperbolic excess speed (= in for unpowered)
    turn_angle_deg: float            # deflection angle of velocity vector
    powered_dv_km_s: float           # propulsion burn at periapsis (0 for unpowered)
    dv_helio_km_s: float             # effective heliocentric Δv gained
    is_valid: bool                   # periapsis above atmosphere
    min_safe_periapsis_km: float     # physical radius + atmosphere clearance

# Safe flyby altitudes (above physical surface) [km]
# Based on atmospheric extent or minimum safe altitude for navigation
SAFE_FLYBY_ALTITUDE_KM: dict[str, float] = {
    "MERCURY": 200.0,
    "VENUS": 300.0,    # Venus has a dense atmosphere (~70km) + margin
    "EARTH": 500.0,
    "MOON": 50.0,
    "MARS": 200.0,
    "JUPITER": 500.0,
    "SATURN": 500.0,
}

def compute_flyby_turn_angle(
    v_inf_km_s: float,
    periapsis_km: float,
    body: str,
) -> float:
    """Compute hyperbolic turn angle [radians] for unpowered flyby.

    δ/2 = arcsin(1 / e)   where e = 1 + r_p × v_inf² / μ
    Full turn angle δ = 2 × arcsin(1/e)
    """
    mu = GM[body.upper()]
    if v_inf_km_s <= 0 or periapsis_km <= 0:
        return 0.0
    e = 1.0 + periapsis_km * v_inf_km_s**2 / mu
    if e < 1.0:
        return 0.0  # not hyperbolic
    half_angle = math.asin(1.0 / e)
    return 2.0 * half_angle

def compute_flyby(
    v_inf_in: np.ndarray,           # incoming hyperbolic excess velocity [km/s]
    periapsis_km: float,             # closest approach radius from body center [km]
    body: str,                       # planet name ("VENUS", "MARS", etc.)
    powered_dv_km_s: float = 0.0,   # propulsion burn at periapsis [km/s]
    flyby_plane_normal: np.ndarray | None = None,  # orbit plane normal
) -> FlybyResult:
    """Compute flyby result: incoming → outgoing hyperbolic excess velocity.

    For unpowered flyby: |v_inf_out| = |v_inf_in| (energy conserved in planet frame)
    For powered flyby: v_inf_out = v_inf_in + Δv_periapsis_burn

    The outgoing v_inf direction is rotated by turn_angle about the flyby plane normal.

    Parameters
    ----------
    v_inf_in : incoming hyperbolic excess velocity vector [km/s] in planet frame
    periapsis_km : closest approach from planet center [km]
    body : planet name
    powered_dv_km_s : additional Δv burn at periapsis [km/s]
    flyby_plane_normal : unit vector normal to flyby plane (computed if None)

    NOTE
    ----
    The effective heliocentric Δv gained (dv_helio_km_s) is computed using
    the patched-conics approximation: it is the magnitude of the vector change in
    excess velocity, i.e., ||v_inf_out - v_inf_in||. This represents the gravity-assist
    deflection and periapsis burn combined, assuming an instantaneous transfer inside the
    planet's Sphere of Influence. This is a patched-conics approximation, not a true
    numerical integration of the trajectory under 3-body or heliocentric gravity.
    """
    body_upper = body.upper()
    mu = GM[body_upper]
    R_body = PHYSICAL_RADIUS[CelestialBody[body_upper]]
    safe_alt = SAFE_FLYBY_ALTITUDE_KM.get(body_upper, 300.0)
    min_safe_r = R_body + safe_alt

    v_inf_in_mag = float(np.linalg.norm(v_inf_in))
    is_valid = periapsis_km >= min_safe_r

    # Speed at periapsis and outgoing excess velocity computation
    if powered_dv_km_s > 0:
        # Speed at periapsis of incoming hyperbola
        v_peri_in = math.sqrt(v_inf_in_mag**2 + 2.0 * mu / periapsis_km)
        v_peri_out = v_peri_in + powered_dv_km_s
        v_inf_out_mag = math.sqrt(max(0.0, v_peri_out**2 - 2.0 * mu / periapsis_km))
        
        # Turn angle for powered flyby (asymmetric incoming & outgoing asymptotes)
        e_in = 1.0 + periapsis_km * v_inf_in_mag**2 / mu
        e_out = 1.0 + periapsis_km * v_inf_out_mag**2 / mu
        if e_in >= 1.0 and e_out >= 1.0:
            turn_angle_rad = math.asin(1.0 / e_in) + math.asin(1.0 / e_out)
        else:
            turn_angle_rad = 0.0
    else:
        v_inf_out_mag = v_inf_in_mag
        turn_angle_rad = compute_flyby_turn_angle(v_inf_in_mag, periapsis_km, body_upper)

    turn_angle_deg = math.degrees(turn_angle_rad)

    # Rotate v_inf_in by turn_angle to get v_inf_out direction
    v_inf_in_hat = v_inf_in / (v_inf_in_mag + 1e-10)

    # Compute rotation axis (perpendicular to v_inf_in, in the flyby plane)
    if flyby_plane_normal is None:
        # Choose arbitrary normal perpendicular to v_inf_in
        arbitrary = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(v_inf_in_hat, arbitrary)) > 0.9:
            arbitrary = np.array([0.0, 1.0, 0.0])
        rot_axis = np.cross(v_inf_in_hat, arbitrary)
        rot_axis /= np.linalg.norm(rot_axis) + 1e-10
    else:
        rot_axis = flyby_plane_normal / (np.linalg.norm(flyby_plane_normal) + 1e-10)

    # Rodrigues' rotation formula: rotate v_inf_in_hat by turn_angle about rot_axis
    c, s = math.cos(turn_angle_rad), math.sin(turn_angle_rad)
    v_inf_out_hat = (v_inf_in_hat * c
                     + np.cross(rot_axis, v_inf_in_hat) * s
                     + rot_axis * np.dot(rot_axis, v_inf_in_hat) * (1.0 - c))
    v_inf_out = v_inf_out_hat * v_inf_out_mag

    # Effective heliocentric Δv = |v_inf_out - v_inf_in| (in planet frame = heliocentric gain)
    dv_helio = float(np.linalg.norm(v_inf_out - v_inf_in))

    return FlybyResult(
        body=body_upper,
        periapsis_km=periapsis_km,
        periapsis_altitude_km=periapsis_km - R_body,
        v_inf_in_km_s=v_inf_in_mag,
        v_inf_out_km_s=v_inf_out_mag,
        turn_angle_deg=turn_angle_deg,
        powered_dv_km_s=powered_dv_km_s,
        dv_helio_km_s=dv_helio,
        is_valid=is_valid,
        min_safe_periapsis_km=min_safe_r,
    )
