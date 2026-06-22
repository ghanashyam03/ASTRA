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
    bplane_theta_rad: float | None = None,   # NEW
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
    bplane_theta_rad : B-plane angle in radians to determine the flyby plane rotation axis

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
    if flyby_plane_normal is not None:
        rot_axis = flyby_plane_normal / (np.linalg.norm(flyby_plane_normal) + 1e-10)
    elif bplane_theta_rad is not None:
        S, T, R = build_bplane_frame(v_inf_in)
        B_hat = math.cos(bplane_theta_rad) * T + math.sin(bplane_theta_rad) * R
        rot_axis = orbit_normal_from_bvector(S, B_hat)
    else:
        # Choose arbitrary normal perpendicular to v_inf_in
        arbitrary = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(v_inf_in_hat, arbitrary)) > 0.9:
            arbitrary = np.array([0.0, 1.0, 0.0])
        rot_axis = np.cross(v_inf_in_hat, arbitrary)
        rot_axis /= np.linalg.norm(rot_axis) + 1e-10

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


def build_bplane_frame(
    v_inf_in: np.ndarray,
    reference_pole: np.ndarray = np.array([0.0, 0.0, 1.0]),  # ecliptic north, matches
                                                               # ASTRA's ECLIPJ2000 frame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the (S, T, R) B-plane basis for an incoming hyperbolic asymptote.

    S = unit vector along v_inf_in (incoming asymptote direction).
    T = unit vector in the plane perpendicular to S, derived from reference_pole.
    R = S × T, completing a right-handed orthonormal frame.

    The B-vector (impact parameter vector) lies in the T-R plane:
        B = b·cos(theta)·T + b·sin(theta)·R
    where b is the impact parameter magnitude and theta is the B-plane angle.

    The orbit normal (rotation axis for the flyby) is:
        h_hat = normalize(S × B_hat)
    This is the physically correct replacement for the arbitrary perpendicular
    currently used as a fallback in compute_flyby — theta now determines which
    side of the body the spacecraft passes (leading vs trailing relative to the
    body's own orbital motion), which is the actual mechanism that determines
    whether a gravity assist adds or removes heliocentric energy.

    Degenerate case: if S is nearly parallel to reference_pole (|S·pole| > 0.9),
    fall back to reference_pole = [0,1,0] to avoid a near-zero cross product,
    exactly as the existing arbitrary-normal code already guards against.
    """
    v_inf_in_mag = float(np.linalg.norm(v_inf_in))
    S = v_inf_in / max(v_inf_in_mag, 1e-10)
    pole = reference_pole
    if abs(float(np.dot(S, pole))) > 0.9:
        pole = np.array([0.0, 1.0, 0.0])
    T = np.cross(S, pole)
    T = T / max(float(np.linalg.norm(T)), 1e-10)
    R = np.cross(S, T)
    return S, T, R


def bplane_vector(b: float, theta_rad: float, T: np.ndarray, R: np.ndarray) -> np.ndarray:  # noqa: N803
    """Construct the B-vector from impact parameter and B-plane angle."""
    return b * math.cos(theta_rad) * T + b * math.sin(theta_rad) * R


def orbit_normal_from_bvector(S: np.ndarray, B_hat: np.ndarray) -> np.ndarray:  # noqa: N803
    """h_hat = normalize(S × B_hat) — the rotation axis for the flyby."""
    h = np.cross(S, B_hat)
    h_norm = float(np.linalg.norm(h))
    if h_norm < 1e-10:
        # B nearly anti/parallel to S — degenerate, fall back to arbitrary perpendicular
        arbitrary = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(S, arbitrary))) > 0.9:
            arbitrary = np.array([0.0, 1.0, 0.0])
        h = np.cross(S, arbitrary)
        h_norm = float(np.linalg.norm(h))
    return h / h_norm


def periapsis_from_impact_parameter(b_km: float, v_inf_km_s: float, mu: float) -> float:
    """Closed-form: r_p = sqrt(a_h² + b²) - a_h, where a_h = mu / v_inf²."""
    a_h = mu / (v_inf_km_s ** 2)
    return float(math.sqrt(a_h ** 2 + b_km ** 2) - a_h)


def impact_parameter_from_periapsis(r_p_km: float, v_inf_km_s: float, mu: float) -> float:
    """Closed-form: b = sqrt(r_p² + 2·r_p·a_h)."""
    a_h = mu / (v_inf_km_s ** 2)
    return float(math.sqrt(r_p_km ** 2 + 2.0 * r_p_km * a_h))


def periapsis_from_turn_angle(
    target_turn_rad: float, v_inf_km_s: float, mu: float
) -> float:
    """Closed-form: e = 1/sin(δ/2), r_p = a_h·(e-1). No iteration.
    Caller MUST check feasibility (max_achievable_turn_angle) BEFORE calling this —
    this function does not validate; it computes the periapsis that WOULD be
    needed even if that periapsis violates the safe-altitude minimum. The gate
    function below performs the actual feasibility check."""
    a_h = mu / (v_inf_km_s ** 2)
    half_angle = target_turn_rad / 2.0
    if half_angle <= 0.0 or half_angle >= math.pi / 2.0:
        raise ValueError(f"Target turn angle {math.degrees(target_turn_rad):.2f}° "
                         f"out of valid (0°, 180°) range")
    e = 1.0 / math.sin(half_angle)
    return float(a_h * (e - 1.0))


def max_achievable_turn_angle(
    v_inf_km_s: float, r_min_km: float, mu: float
) -> float:
    """Maximum turn angle [radians] achievable by unpowered gravity assist alone,
    i.e. at the minimum safe periapsis r_min_km."""
    a_h = mu / (v_inf_km_s ** 2)
    e_min = 1.0 + r_min_km / a_h
    return float(2.0 * math.asin(1.0 / e_min))


def max_achievable_turn_angle_with_unlimited_burn(
    v_inf_in_km_s: float, r_min_km: float, mu: float
) -> float:
    """Hard physical ceiling on turn angle at r_min, even with unlimited powered Δv
    at periapsis. δ_ceiling = asin(1/e_in) + π/2. Beyond this, NO finite correction
    at this body can achieve the required turn — this is a true rejection boundary,
    not a budget-dependent one."""
    a_h = mu / (v_inf_in_km_s ** 2)
    e_in = 1.0 + r_min_km / a_h
    return float(math.asin(1.0 / e_in) + math.pi / 2.0)


@dataclass
class FlybyFeasibility:
    is_achievable_unpowered: bool      # pure gravity assist, no burn needed
    is_achievable_with_bounded_burn: bool   # achievable with SOME finite powered_dv
    is_achievable_at_all: bool          # False ⟹ hard rejection, no budget can fix it
    required_turn_rad: float
    max_unpowered_turn_rad: float
    max_turn_with_unlimited_burn_rad: float
    solved_periapsis_km: float | None   # None if infeasible even unpowered
    min_safe_periapsis_km: float
    rejection_reason: str | None


def check_flyby_feasibility(
    v_inf_in_mag_km_s: float,
    required_turn_rad: float,
    body: str,
) -> FlybyFeasibility:
    """The mandatory gate. Given an incoming excess speed and a REQUIRED turn
    angle, determine whether this flyby can be achieved at all, and if so,
    whether it needs a powered burn. Never silently substitutes a different
    turn angle — every outcome is one of: unpowered-feasible, burn-feasible,
    or hard-rejected with an explicit reason string.
    """
    from astra.state.orbital_state import GM, PHYSICAL_RADIUS, CelestialBody
    body_upper = body.upper()
    mu = GM[body_upper]
    R_body = PHYSICAL_RADIUS[CelestialBody[body_upper]]
    r_min = R_body + SAFE_FLYBY_ALTITUDE_KM.get(body_upper, 300.0)

    max_unpowered = max_achievable_turn_angle(v_inf_in_mag_km_s, r_min, mu)
    max_with_burn = max_achievable_turn_angle_with_unlimited_burn(
        v_inf_in_mag_km_s, r_min, mu
    )

    if required_turn_rad <= max_unpowered:
        r_p = periapsis_from_turn_angle(required_turn_rad, v_inf_in_mag_km_s, mu)
        return FlybyFeasibility(
            is_achievable_unpowered=True,
            is_achievable_with_bounded_burn=True,
            is_achievable_at_all=True,
            required_turn_rad=required_turn_rad,
            max_unpowered_turn_rad=max_unpowered,
            max_turn_with_unlimited_burn_rad=max_with_burn,
            solved_periapsis_km=r_p,
            min_safe_periapsis_km=r_min,
            rejection_reason=None,
        )
    elif required_turn_rad <= max_with_burn:
        return FlybyFeasibility(
            is_achievable_unpowered=False,
            is_achievable_with_bounded_burn=True,
            is_achievable_at_all=True,
            required_turn_rad=required_turn_rad,
            max_unpowered_turn_rad=max_unpowered,
            max_turn_with_unlimited_burn_rad=max_with_burn,
            solved_periapsis_km=r_min,  # closest approach gives the best leverage
            min_safe_periapsis_km=r_min,
            rejection_reason=None,
        )
    else:
        return FlybyFeasibility(
            is_achievable_unpowered=False,
            is_achievable_with_bounded_burn=False,
            is_achievable_at_all=False,
            required_turn_rad=required_turn_rad,
            max_unpowered_turn_rad=max_unpowered,
            max_turn_with_unlimited_burn_rad=max_with_burn,
            solved_periapsis_km=None,
            min_safe_periapsis_km=r_min,
            rejection_reason=(
                f"Required turn {math.degrees(required_turn_rad):.2f}° exceeds the "
                f"physical ceiling {math.degrees(max_with_burn):.2f}° at {body_upper} "
                f"even with unlimited powered correction at periapsis {r_min:.0f} km. "
                f"This geometry is impossible at this body — not a budget problem."
            ),
        )