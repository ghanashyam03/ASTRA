"""Structurally correct multi-leg trajectory chain resolver.

Replaces ad hoc per-step flyby reconciliation. Every flyby in a chain is
checked for physical feasibility BEFORE any Trajectory is constructed —
infeasibility is always an explicit, early return, never a silent substitution.
See the algorithm specification in the prompt that generated this module for
the full derivation of each case.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from astra.dsl.compiler import CompiledMission
from astra.physics.flyby import check_flyby_feasibility
from astra.physics.kernel import PhysicsKernel
from astra.physics.lambert import find_best_transfer
from astra.physics.maneuvers import arrival_delta_v, departure_delta_v
from astra.state.orbital_state import GM, PHYSICAL_RADIUS, CelestialBody, OrbitalState
from astra.state.trajectory import Maneuver, Trajectory


class RejectionReason(StrEnum):
    LAMBERT_FAILED = "lambert_failed"
    UNKNOWN_BODY = "unknown_body"
    ZERO_V_INF = "zero_v_inf"
    IMPOSSIBLE_GEOMETRY = "impossible_geometry"
    BUDGET_EXCEEDED = "budget_exceeded"


@dataclass
class ChainResult:
    feasible: bool
    trajectory: Trajectory | None
    reason: str | None
    leg_details: list[dict[str, Any]] = field(default_factory=list)
    reason_code: RejectionReason | None = None


def _achieved_turn(r_p: float, v_inf_in_mag: float, v_inf_out_mag: float, mu: float) -> float:
    e_in = 1.0 + r_p * v_inf_in_mag**2 / mu
    e_out = 1.0 + r_p * v_inf_out_mag**2 / mu
    return math.asin(min(1.0, 1.0 / e_in)) + math.asin(min(1.0, 1.0 / e_out))


def _solve_periapsis_bisection(
    required_turn_rad: float,
    v_inf_in_mag: float,
    v_inf_out_mag: float,
    mu: float,
    r_min: float,
    r_max: float,
    tol: float = 1e-9,
    max_iter: int = 100,
) -> float | None:
    """Exact bisection exploiting the proven monotonicity of achieved_turn(r_p).

    Returns None if required_turn_rad is outside the achievable range at the
    ENDPOINTS [r_min, r_max] — i.e. no root exists in the search interval,
    which is itself meaningful information (not a numerical failure).
    """
    turn_at_min = _achieved_turn(r_min, v_inf_in_mag, v_inf_out_mag, mu)
    turn_at_max = _achieved_turn(r_max, v_inf_in_mag, v_inf_out_mag, mu)

    # achieved_turn is decreasing in r_p, so turn_at_min >= turn_at_max
    # Add a tiny tolerance for the boundary check to prevent rejecting
    # valid edge solutions due to floating-point noise.
    eps = 1e-12
    if not (turn_at_max - eps <= required_turn_rad <= turn_at_min + eps):
        return None  # no root in [r_min, r_max] — genuinely infeasible here

    # Clip required_turn_rad to the valid range [turn_at_max, turn_at_min] to ensure
    # that the bisection interval remains mathematically sound.
    required_turn_rad = max(turn_at_max, min(turn_at_min, required_turn_rad))

    lo, hi = r_min, r_max
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        turn_mid = _achieved_turn(mid, v_inf_in_mag, v_inf_out_mag, mu)
        if abs(turn_mid - required_turn_rad) < tol:
            return mid
        # achieved_turn decreasing: if turn_mid > required, need LARGER r_p
        if turn_mid > required_turn_rad:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def resolve_single_flyby_segment(
    body: str,
    v_inf_in: np.ndarray,
    v_inf_out_required: np.ndarray,
    min_alt_km: float,
    max_alt_km: float,
    powered_allowed: bool,
    max_powered_km_s: float,
    dsm_budget_available: float = 0.0,
) -> tuple[dict[str, Any] | None, str | None]:
    """Resolves a single flyby segment, checking physical feasibility.

    Returns:
        (result_dict, error_string):
        If feasible, result_dict is populated and error_string is None.
        If infeasible or budget exceeded, result_dict is None and error_string is the reason.
    """
    body_upper = body.upper()
    try:
        body_enum = CelestialBody[body_upper]
    except KeyError:
        return None, f"Unknown celestial body: {body_upper}"

    mu_body = GM[body_upper]
    R_body = PHYSICAL_RADIUS[body_enum]
    r_min = R_body + min_alt_km
    r_max = R_body + max_alt_km

    v_inf_in_mag = float(np.linalg.norm(v_inf_in))
    v_inf_out_mag = float(np.linalg.norm(v_inf_out_required))

    if v_inf_in_mag <= 1e-6 or v_inf_out_mag <= 1e-6:
        return None, f"Excess velocity magnitudes at {body_upper} must be non-zero."

    cos_req = float(np.dot(v_inf_in, v_inf_out_required) / (v_inf_in_mag * v_inf_out_mag))
    cos_req = max(-1.0, min(1.0, cos_req))
    required_turn_rad = math.acos(cos_req)
    magnitude_mismatch = abs(v_inf_in_mag - v_inf_out_mag)

    feas = check_flyby_feasibility(v_inf_in_mag, required_turn_rad, body_upper)

    # Case A: Pure gravity assist (zero cost)
    # Check if turn is unpowered achievable and magnitudes coincide to numerical precision (1e-3)
    if feas.is_achievable_unpowered and magnitude_mismatch < 1e-3:
        r_p = feas.solved_periapsis_km if feas.solved_periapsis_km is not None else r_min
        return {
            "dv_km_s": 0.0,
            "periapsis_km": r_p,
            "resolution": "unpowered",
            "from_powered": 0.0,
            "from_dsm": 0.0,
            "required_turn_deg": math.degrees(required_turn_rad),
            "max_unpowered_turn_deg": math.degrees(feas.max_unpowered_turn_rad),
            "max_turn_with_unlimited_burn_deg": math.degrees(feas.max_turn_with_unlimited_burn_rad),
            "magnitude_mismatch_km_s": magnitude_mismatch,
        }, None

    # Case B: Powered periapsis burn needed (magnitude mismatch and/or turn angle beyond ceiling)
    best_r_p = _solve_periapsis_bisection(
        required_turn_rad, v_inf_in_mag, v_inf_out_mag, mu_body, r_min, r_max
    )

    if best_r_p is not None:
        v_peri_in = math.sqrt(v_inf_in_mag**2 + 2.0 * mu_body / best_r_p)
        v_peri_out_needed = math.sqrt(v_inf_out_mag**2 + 2.0 * mu_body / best_r_p)
        best_powered_dv = abs(v_peri_out_needed - v_peri_in)
    else:
        best_powered_dv = float("inf")

    if best_r_p is None:
        err = (
            f"Flyby at {body_upper}: no periapsis in [{r_min:.0f},{r_max:.0f}] km "
            f"achieves required turn {math.degrees(required_turn_rad):.2f}° "
            f"(unpowered ceiling {math.degrees(feas.max_unpowered_turn_rad):.2f}°, "
            f"unlimited-burn ceiling {math.degrees(feas.max_turn_with_unlimited_burn_rad):.2f}°). "
            f"This geometry is impossible at this body."
        )
        return None, err

    budget_available = (max_powered_km_s if powered_allowed else 0.0) + dsm_budget_available
    if best_powered_dv > budget_available:
        err = (
            f"Flyby at {body_upper} requires {best_powered_dv:.4f} km/s correction "
            f"(periapsis {best_r_p:.0f} km) but only {budget_available:.4f} km/s "
            f"is available (powered budget + remaining DSM budget). Increase "
            f"the declared budget or relax the trajectory geometry."
        )
        return None, err

    # Allocate costs
    powered_budget = max_powered_km_s if powered_allowed else 0.0
    from_powered = min(best_powered_dv, powered_budget)
    from_dsm = best_powered_dv - from_powered

    resolution = "powered" if from_dsm < 1e-6 else "powered+dsm"

    return {
        "dv_km_s": best_powered_dv,
        "periapsis_km": best_r_p,
        "resolution": resolution,
        "from_powered": from_powered,
        "from_dsm": from_dsm,
        "required_turn_deg": math.degrees(required_turn_rad),
        "max_unpowered_turn_deg": math.degrees(feas.max_unpowered_turn_rad),
        "max_turn_with_unlimited_burn_deg": math.degrees(feas.max_turn_with_unlimited_burn_rad),
        "magnitude_mismatch_km_s": magnitude_mismatch,
    }, None


def resolve_flyby_chain(
    mission: CompiledMission,
    kernel: PhysicsKernel,
    chain_bodies: list[str],  # [origin_name, flyby1, flyby2, ..., dest_name]
    departure_epoch: float,
    leg_tofs: list[float],  # len = len(chain_bodies) - 1
    flyby_specs: dict[str, dict[str, Any]],  # body_name -> {min_alt_km, max_alt_km,
    #               powered_allowed, max_powered_km_s}
) -> ChainResult:
    """Resolve a full multi-leg chain with mandatory per-flyby feasibility checking."""
    assert len(leg_tofs) == len(chain_bodies) - 1

    mu_sun = GM["SUN"]
    epochs = [departure_epoch]
    for tof in leg_tofs:
        epochs.append(epochs[-1] + tof)

    bodies_enum = [CelestialBody[b.upper()] for b in chain_bodies]
    body_states = [
        kernel.get_body_state(bodies_enum[i], epochs[i]) for i in range(len(chain_bodies))
    ]

    # STEP 1: solve every leg's Lambert problem
    leg_solutions = []
    for i in range(len(leg_tofs)):
        try:
            sol = find_best_transfer(
                r1=body_states[i].position,
                v1_body=body_states[i].velocity,
                r2=body_states[i + 1].position,
                v2_body=body_states[i + 1].velocity,
                tof=leg_tofs[i],
                mu=mu_sun,
                max_revs=mission.max_revs_per_leg,
            )
            leg_solutions.append(sol)
        except Exception as e:
            return ChainResult(
                False,
                None,
                f"Leg {i} Lambert solve failed: {e}",
                reason_code=RejectionReason.LAMBERT_FAILED,
            )

    maneuvers: list[Maneuver] = []
    dsm_remaining = mission.dsm_budget_km_s
    leg_details: list[dict[str, Any]] = []

    # STEP 2: departure burn
    v_inf_dep = leg_solutions[0].v1 - body_states[0].velocity
    dv_tmi = departure_delta_v(v_inf_dep, mission.parking_altitude_km, chain_bodies[0])
    dv_tmi_vec = (v_inf_dep / max(float(np.linalg.norm(v_inf_dep)), 1e-10)) * dv_tmi
    maneuvers.append(Maneuver(epoch=epochs[0], delta_v=dv_tmi_vec, label="TMI"))

    # STEP 3: each intermediate flyby
    for k in range(1, len(chain_bodies) - 1):
        body = chain_bodies[k]
        spec = flyby_specs.get(
            body.upper(),
            {
                "min_alt_km": 300.0,
                "max_alt_km": 50000.0,
                "powered_allowed": False,
                "max_powered_km_s": 0.0,
            },
        )
        v_inf_in = leg_solutions[k - 1].v2 - body_states[k].velocity
        v_inf_out_required = leg_solutions[k].v1 - body_states[k].velocity

        res, err = resolve_single_flyby_segment(
            body=body,
            v_inf_in=v_inf_in,
            v_inf_out_required=v_inf_out_required,
            min_alt_km=spec["min_alt_km"],
            max_alt_km=spec["max_alt_km"],
            powered_allowed=spec["powered_allowed"],
            max_powered_km_s=spec["max_powered_km_s"],
            dsm_budget_available=dsm_remaining,
        )
        if res is None:
            reason_code = RejectionReason.IMPOSSIBLE_GEOMETRY
            if err is not None:
                if "requires" in err and "correction" in err:
                    reason_code = RejectionReason.BUDGET_EXCEEDED
                elif "Unknown celestial body" in err:
                    reason_code = RejectionReason.UNKNOWN_BODY
                elif "Excess velocity magnitudes" in err:
                    reason_code = RejectionReason.ZERO_V_INF
            return ChainResult(False, None, err, leg_details, reason_code=reason_code)

        dsm_remaining -= res["from_dsm"]
        v_inf_in_mag = float(np.linalg.norm(v_inf_in))
        dv_vec = (v_inf_in / max(v_inf_in_mag, 1e-10)) * res["dv_km_s"]

        is_powered = res["from_dsm"] < 1e-6
        label = f"FLY_{body.upper()}_POWERED" if is_powered else f"FLY_{body.upper()}_DSM"
        if res["dv_km_s"] < 1e-6:
            label = f"FLY_{body.upper()}"

        maneuvers.append(Maneuver(epoch=epochs[k], delta_v=dv_vec, label=label))

        leg_detail = {
            "body": body.upper(),
            "required_turn_deg": res["required_turn_deg"],
            "magnitude_mismatch_km_s": res["magnitude_mismatch_km_s"],
            "max_unpowered_turn_deg": res["max_unpowered_turn_deg"],
            "resolution": res["resolution"],
            "dv_km_s": res["dv_km_s"],
            "periapsis_km": res["periapsis_km"],
        }
        leg_details.append(leg_detail)

    # STEP 4: arrival burn
    v_inf_arr = body_states[-1].velocity - leg_solutions[-1].v2
    dv_moi = arrival_delta_v(
        v_inf_arr,
        mission.capture_altitude_km,
        chain_bodies[-1],
        apoapsis_km=mission.capture_apoapsis_km,
    )
    dv_moi_vec = (v_inf_arr / max(float(np.linalg.norm(v_inf_arr)), 1e-10)) * dv_moi
    maneuvers.append(Maneuver(epoch=epochs[-1], delta_v=dv_moi_vec, label="MOI"))

    states = [
        OrbitalState(
            epoch=epochs[i],
            position=body_states[i].position.copy(),
            velocity=(
                leg_solutions[i].v1 if i < len(leg_solutions) else leg_solutions[-1].v2
            ).copy(),
            central_body=CelestialBody.SUN,
        )
        for i in range(len(chain_bodies))
    ]

    trajectory = Trajectory(
        states=states,
        maneuvers=maneuvers,
        metadata={"chain": chain_bodies, "leg_details": leg_details},
    )
    return ChainResult(True, trajectory, None, leg_details, reason_code=None)
