"""Numerical validation of the instantaneous-flyby (closed-form Rodrigues
rotation) approximation against direct two-body propagation through the
actual hyperbolic encounter. This module measures, it does not assume.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from astra.physics.propagator import propagate_two_body
from astra.state.orbital_state import GM, CelestialBody, OrbitalState, ReferenceFrame


@dataclass
class FlybyApproximationCheck:
    body: str
    v_inf_km_s: float
    periapsis_km: float
    r_stop_km: float
    numerical_v_inf_in: np.ndarray
    numerical_v_inf_out: np.ndarray
    closed_form_v_inf_out: np.ndarray
    angular_discrepancy_deg: float
    speed_convergence_error_fraction: float


def numerical_flyby_check(
    v_inf_km_s: float,
    periapsis_km: float,
    body: str,
    tolerance: float = 1e-8,
    stopping_mode: str = "speed",
) -> FlybyApproximationCheck:
    """Measure the instantaneous-flyby approximation error for one (v_inf,
    periapsis, body) case via direct numerical propagation.
    """
    from astra.physics.flyby import compute_flyby

    mu = GM[body.upper()]
    cb = CelestialBody[body.upper()]

    v_peri = math.sqrt(v_inf_km_s**2 + 2.0 * mu / periapsis_km)

    if stopping_mode == "direction":
        h = periapsis_km * v_peri
        r_stop = (mu * h / (v_inf_km_s * tolerance)) ** (1.0 / 3.0)
    else:
        r_stop = 2.0 * mu / (v_inf_km_s**2 * tolerance)

    # Periapsis state: position along local x, velocity along local y
    pos0 = np.array([periapsis_km, 0.0, 0.0])
    vel0 = np.array([0.0, v_peri, 0.0])
    state0 = OrbitalState(
        epoch=0.0, position=pos0, velocity=vel0, frame=ReferenceFrame.ICRF, central_body=cb
    )

    # Propagate FORWARD to find the outgoing asymptote
    dt = 3600.0
    state_fwd = state0
    while float(np.linalg.norm(state_fwd.position)) < r_stop and dt < 1e16:
        state_fwd = propagate_two_body(state0, dt)
        dt *= 1.5
    v_out_numerical = state_fwd.velocity.copy()
    speed_error = abs(float(np.linalg.norm(v_out_numerical)) - v_inf_km_s) / v_inf_km_s

    # Propagate BACKWARD (negative dt) to find the incoming asymptote
    dt_back = -3600.0
    state_back = state0
    while float(np.linalg.norm(state_back.position)) < r_stop and abs(dt_back) < 1e16:
        state_back = propagate_two_body(state0, dt_back)
        dt_back *= 1.5
    v_in_numerical = state_back.velocity.copy()

    # Closed-form prediction, fed the NUMERICALLY-DERIVED incoming vector for
    # an apples-to-apples comparison
    plane_normal = np.cross(pos0, vel0)
    plane_normal_norm = np.linalg.norm(plane_normal)
    if plane_normal_norm > 1e-10:
        plane_normal = plane_normal / plane_normal_norm
    else:
        plane_normal = np.array([0.0, 0.0, 1.0])

    closed_form_with_plane = compute_flyby(
        v_in_numerical,
        periapsis_km,
        body,
        powered_dv_km_s=0.0,
        flyby_plane_normal=plane_normal,
    )
    turn_rad = math.radians(closed_form_with_plane.turn_angle_deg)
    c, s = math.cos(turn_rad), math.sin(turn_rad)
    v_in_hat = v_in_numerical / np.linalg.norm(v_in_numerical)
    v_out_closed_hat = (
        v_in_hat * c
        + np.cross(plane_normal, v_in_hat) * s
        + plane_normal * np.dot(plane_normal, v_in_hat) * (1.0 - c)
    )
    v_out_closed = v_out_closed_hat * float(np.linalg.norm(v_out_numerical))

    v_out_num_hat = v_out_numerical / np.linalg.norm(v_out_numerical)
    cos_disc = float(np.clip(np.dot(v_out_num_hat, v_out_closed_hat), -1.0, 1.0))
    angular_discrepancy_deg = math.degrees(math.acos(cos_disc))

    return FlybyApproximationCheck(
        body=body.upper(),
        v_inf_km_s=v_inf_km_s,
        periapsis_km=periapsis_km,
        r_stop_km=r_stop,
        numerical_v_inf_in=v_in_numerical,
        numerical_v_inf_out=v_out_numerical,
        closed_form_v_inf_out=v_out_closed,
        angular_discrepancy_deg=angular_discrepancy_deg,
        speed_convergence_error_fraction=speed_error,
    )
