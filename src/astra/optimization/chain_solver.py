"""Structurally correct multi-leg trajectory chain resolver.

Replaces ad hoc per-step flyby reconciliation. Every flyby in a chain is
checked for physical feasibility BEFORE any Trajectory is constructed —
infeasibility is always an explicit, early return, never a silent substitution.
See the algorithm specification in the prompt that generated this module for
the full derivation of each case.
"""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)  # noqa: E402


class RejectionReason(StrEnum):
    LAMBERT_FAILED = "lambert_failed"
    UNKNOWN_BODY = "unknown_body"
    ZERO_V_INF = "zero_v_inf"
    IMPOSSIBLE_GEOMETRY = "impossible_geometry"
    BUDGET_EXCEEDED = "budget_exceeded"


@dataclass
class TrajectoryRejectionRecord:
    trial_id: int
    departure_epoch: float
    leg_tofs: list[float]

    current_leg: int
    current_body: str

    rejection_stage: str

    reason_code: str

    incoming_v_inf: np.ndarray | None
    outgoing_v_inf: np.ndarray | None

    required_turn_angle: float | None
    maximum_turn_angle: float | None

    periapsis_radius: float | None
    minimum_allowed_radius: float | None

    max_revs_used: int | None

    delta_v_cost: float | None


@dataclass
class ChainResult:
    feasible: bool
    trajectory: Trajectory | None
    reason: str | None = None
    leg_details: list[dict[str, Any]] = field(default_factory=list)
    reason_code: RejectionReason | None = None
    rejection_records: list[TrajectoryRejectionRecord] = field(default_factory=list)
    total_continuity_violation_km_s: float = 0.0


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


def resolve_flyby_high_fidelity(
    v_inf_in: np.ndarray,
    periapsis_km: float,
    body: str,
    encounter_epoch: float,
    kernel: PhysicsKernel,
) -> np.ndarray:
    """Resolve a flyby's outgoing v_inf using three-body propagation
    (Sun + the actual, time-varying ephemeris position of the flyby body)
    instead of the closed-form, planet-frozen Rodrigues rotation.

    Returns the outgoing v_inf vector (heliocentric, relative to the
    flyby body's velocity AT THE EXIT epoch — not the encounter epoch,
    since that distinction is exactly what this function corrects for).
    """
    import math

    from astra.physics.forces.gravity import PointMassGravity
    from astra.physics.forces.third_body import EphemerisThirdBodyPerturbation
    from astra.physics.propagator import propagate_two_body
    from astra.physics.soi import compute_soi_radius
    from astra.state.orbital_state import GM, CelestialBody, OrbitalState, ReferenceFrame

    cb = CelestialBody[body.upper()]
    mu_body = GM[body.upper()]
    mu_sun = GM["SUN"]
    v_inf_mag = float(np.linalg.norm(v_inf_in))

    v_peri = math.sqrt(v_inf_mag**2 + 2.0 * mu_body / periapsis_km)
    planet_state = kernel.get_body_state(cb, encounter_epoch)

    from astra.physics.flyby import (
        build_bplane_frame,
        build_geometrically_consistent_periapsis,
        compute_flyby_turn_angle,
        orbit_normal_from_bvector,
    )

    body_upper = body.upper()
    # Turn angle for this approach geometry (exact hyperbolic formula)
    turn_rad = compute_flyby_turn_angle(v_inf_mag, periapsis_km, body_upper)

    # Orbit normal using T-axis as B-vector (θ=0 convention; reproducible and consistent)
    S_hat, T_hat, _ = build_bplane_frame(v_inf_in)
    h_hat = orbit_normal_from_bvector(S_hat, T_hat)

    # Geometrically self-consistent periapsis position and velocity unit vectors
    r_peri_hat, v_peri_hat, _ = build_geometrically_consistent_periapsis(S_hat, h_hat, turn_rad)

    # Periapsis state in planet-centered inertial frame
    pos_planet = r_peri_hat * periapsis_km
    vel_planet = v_peri_hat * v_peri  # v_peri already computed from vis-viva above

    r_helio_peri = planet_state.position + pos_planet
    v_helio_peri = planet_state.velocity + vel_planet
    state0 = OrbitalState(
        epoch=encounter_epoch,
        position=r_helio_peri,
        velocity=v_helio_peri,
        frame=ReferenceFrame.ECLIPJ2000,
        central_body=CelestialBody.SUN,
    )

    r_soi = compute_soi_radius(body)
    dt_half = r_soi / v_inf_mag

    forces = [
        PointMassGravity(mu_sun),
        EphemerisThirdBodyPerturbation(kernel, body, encounter_epoch),
    ]
    state_exit = propagate_two_body(state0, dt_half, forces=forces)

    exit_planet_state = kernel.get_body_state(cb, encounter_epoch + dt_half)
    v_inf_out = state_exit.velocity - exit_planet_state.velocity
    return v_inf_out  # type: ignore[no-any-return]


def resolve_single_flyby_segment(
    body: str,
    v_inf_in: np.ndarray,
    v_inf_out_required: np.ndarray,
    min_alt_km: float,
    max_alt_km: float,
    powered_allowed: bool,
    max_powered_km_s: float,
    dsm_budget_available: float = 0.0,
    high_fidelity_ratio_threshold: float = 5.0,
    encounter_epoch: float | None = None,
    kernel: PhysicsKernel | None = None,
) -> tuple[dict[str, Any] | None, str | None, dict[str, Any] | None]:
    """Resolves a single flyby segment, checking physical feasibility.

    Returns:
        (result_dict, error_string, failure_dict):
        If feasible, result_dict is populated and error_string/failure_dict are None.
        If infeasible or budget exceeded, result_dict is None, error_string is the reason,
        and failure_dict contains detailed diagnostic information.
    """
    body_upper = body.upper()
    try:
        body_enum = CelestialBody[body_upper]
    except KeyError:
        fail_cb: dict[str, Any] = {
            "rejection_stage": "unknown_body",
            "reason_code": "unknown_body",
            "incoming_v_inf": v_inf_in,
            "outgoing_v_inf": v_inf_out_required,
            "required_turn_angle": None,
            "maximum_turn_angle": None,
            "periapsis_radius": None,
            "minimum_allowed_radius": None,
            "delta_v_cost": None,
        }
        return None, f"Unknown celestial body: {body_upper}", fail_cb

    mu_body = GM[body_upper]
    R_body = PHYSICAL_RADIUS[body_enum]
    r_min = R_body + min_alt_km
    r_max = R_body + max_alt_km

    v_inf_in_mag = float(np.linalg.norm(v_inf_in))
    v_inf_out_mag = float(np.linalg.norm(v_inf_out_required))

    if v_inf_in_mag <= 1e-6 or v_inf_out_mag <= 1e-6:
        fail_zero: dict[str, Any] = {
            "rejection_stage": "zero_v_inf",
            "reason_code": "zero_v_inf",
            "incoming_v_inf": v_inf_in,
            "outgoing_v_inf": v_inf_out_required,
            "required_turn_angle": None,
            "maximum_turn_angle": None,
            "periapsis_radius": None,
            "minimum_allowed_radius": None,
            "delta_v_cost": None,
        }
        return (
            None,
            f"Excess velocity magnitudes at {body_upper} must be non-zero.",
            fail_zero,
        )

    is_hf = False
    if encounter_epoch is not None and kernel is not None:
        from astra.physics.soi_passage_estimate import soi_crossing_displacement_ratio

        ratio_res = soi_crossing_displacement_ratio(
            body_upper, v_inf_in_mag, encounter_epoch, kernel
        )
        if ratio_res.ratio > high_fidelity_ratio_threshold:
            is_hf = True

    cos_req = float(np.dot(v_inf_in, v_inf_out_required) / (v_inf_in_mag * v_inf_out_mag))
    cos_req = max(-1.0, min(1.0, cos_req))
    required_turn_rad = math.acos(cos_req)
    magnitude_mismatch = abs(v_inf_in_mag - v_inf_out_mag)

    feas = check_flyby_feasibility(v_inf_in_mag, required_turn_rad, body_upper)

    # Case A: Pure gravity assist (zero cost)
    # Check if turn is unpowered achievable and magnitudes coincide to numerical precision (1e-3)
    if feas.is_achievable_unpowered and magnitude_mismatch < 1e-3:
        r_p = feas.solved_periapsis_km if feas.solved_periapsis_km is not None else r_min
        res_dict = {
            "dv_km_s": 0.0,
            "periapsis_km": r_p,
            "resolution": "unpowered",
            "from_powered": 0.0,
            "from_dsm": 0.0,
            "required_turn_deg": math.degrees(required_turn_rad),
            "max_unpowered_turn_deg": math.degrees(feas.max_unpowered_turn_rad),
            "max_turn_with_unlimited_burn_deg": math.degrees(feas.max_turn_with_unlimited_burn_rad),
            "magnitude_mismatch_km_s": magnitude_mismatch,
        }
        if is_hf:
            assert encounter_epoch is not None
            assert kernel is not None
            v_inf_out_hf = resolve_flyby_high_fidelity(
                v_inf_in=v_inf_in,
                periapsis_km=r_p,
                body=body_upper,
                encounter_epoch=encounter_epoch,
                kernel=kernel,
            )
            res_dict["is_hf"] = True
            res_dict["v_inf_out"] = v_inf_out_hf
        return res_dict, None, None

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
        fail_geom: dict[str, Any] = {
            "rejection_stage": "flyby_impossible_geometry",
            "reason_code": "impossible_geometry",
            "incoming_v_inf": v_inf_in,
            "outgoing_v_inf": v_inf_out_required,
            "required_turn_angle": required_turn_rad,
            "maximum_turn_angle": feas.max_turn_with_unlimited_burn_rad,
            "periapsis_radius": None,
            "minimum_allowed_radius": r_min,
            "delta_v_cost": None,
        }
        return None, err, fail_geom

    budget_available = (max_powered_km_s if powered_allowed else 0.0) + dsm_budget_available
    if best_powered_dv > budget_available:
        err = (
            f"Flyby at {body_upper} requires {best_powered_dv:.4f} km/s correction "
            f"(periapsis {best_r_p:.0f} km) but only {budget_available:.4f} km/s "
            f"is available (powered budget + remaining DSM budget). Increase "
            f"the declared budget or relax the trajectory geometry."
        )
        fail_budget: dict[str, Any] = {
            "rejection_stage": "flyby_budget_exceeded",
            "reason_code": "budget_exceeded",
            "incoming_v_inf": v_inf_in,
            "outgoing_v_inf": v_inf_out_required,
            "required_turn_angle": required_turn_rad,
            "maximum_turn_angle": feas.max_turn_with_unlimited_burn_rad,
            "periapsis_radius": best_r_p,
            "minimum_allowed_radius": r_min,
            "delta_v_cost": best_powered_dv,
        }
        return None, err, fail_budget

    # Allocate costs
    powered_budget = max_powered_km_s if powered_allowed else 0.0
    from_powered = min(best_powered_dv, powered_budget)
    from_dsm = best_powered_dv - from_powered

    resolution = "powered" if from_dsm < 1e-6 else "powered+dsm"

    res_dict = {
        "dv_km_s": best_powered_dv,
        "periapsis_km": best_r_p,
        "resolution": resolution,
        "from_powered": from_powered,
        "from_dsm": from_dsm,
        "required_turn_deg": math.degrees(required_turn_rad),
        "max_unpowered_turn_deg": math.degrees(feas.max_unpowered_turn_rad),
        "max_turn_with_unlimited_burn_deg": math.degrees(feas.max_turn_with_unlimited_burn_rad),
        "magnitude_mismatch_km_s": magnitude_mismatch,
    }
    if is_hf:
        assert encounter_epoch is not None
        assert kernel is not None
        v_inf_out_hf = resolve_flyby_high_fidelity(
            v_inf_in=v_inf_in,
            periapsis_km=best_r_p,
            body=body_upper,
            encounter_epoch=encounter_epoch,
            kernel=kernel,
        )
        res_dict["is_hf"] = True
        res_dict["v_inf_out"] = v_inf_out_hf
    return res_dict, None, None


@dataclass
class DSMResolution:
    dsm_epoch: float
    dsm_position: np.ndarray
    dsm_delta_v_km_s: float
    dsm_delta_v_vector: np.ndarray
    effective_arrival_velocity: np.ndarray  # replaces the original leg's v2
    v_after_dsm: np.ndarray


def resolve_leg_with_dsm(
    r1: np.ndarray,
    v1_original: np.ndarray,
    r2: np.ndarray,
    v2_destination_body: np.ndarray,
    t_start: float,
    tof_leg: float,
    dsm_fraction: float,
    mu_sun: float,
    max_revs: int = 0,
) -> DSMResolution:
    """Implements the exact 5-step composition above. dsm_fraction must be
    in (0, 1) — values at or near the endpoints are degenerate (DSM at the
    very start or very end of a leg provides no benefit over adjusting the
    original Lambert solve directly) and should be avoided by the caller's
    search range, not specially handled here."""
    from astra.physics.lambert import find_best_transfer
    from astra.physics.propagator import propagate_two_body
    from astra.state.orbital_state import CelestialBody, OrbitalState

    t_dsm_offset = dsm_fraction * tof_leg
    state_before = propagate_two_body(
        OrbitalState(
            epoch=t_start, position=r1, velocity=v1_original, central_body=CelestialBody.SUN
        ),
        dt_seconds=t_dsm_offset,
    )
    remaining_tof = (1.0 - dsm_fraction) * tof_leg
    sol_dsm = find_best_transfer(
        r1=state_before.position,
        v1_body=state_before.velocity,
        r2=r2,
        v2_body=v2_destination_body,
        tof=remaining_tof,
        mu=mu_sun,
        max_revs=max_revs,
    )
    dsm_vec = sol_dsm.v1 - state_before.velocity
    return DSMResolution(
        dsm_epoch=t_start + t_dsm_offset,
        dsm_position=state_before.position.copy(),
        dsm_delta_v_km_s=float(np.linalg.norm(dsm_vec)),
        dsm_delta_v_vector=dsm_vec,
        effective_arrival_velocity=sol_dsm.v2,
        v_after_dsm=sol_dsm.v1,
    )


def resolve_flyby_chain(
    mission: CompiledMission,
    kernel: PhysicsKernel,
    chain_bodies: list[str],  # [origin_name, flyby1, flyby2, ..., dest_name]
    departure_epoch: float,
    leg_tofs: list[float],  # len = len(chain_bodies) - 1
    flyby_specs: dict[str, dict[str, Any]],  # body_name -> {min_alt_km, max_alt_km,
    #               powered_allowed, max_powered_km_s}
    dsm_fractions: list[float | None] | None = None,
    high_fidelity_ratio_threshold: float = 5.0,
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
    required_departure_vinf: float | None = None  # set after each flyby leg

    for i in range(len(leg_tofs)):
        leg_max_revs = (
            mission.leg_max_revs[i]
            if hasattr(mission, "leg_max_revs") and i < len(mission.leg_max_revs)
            else mission.max_revs_per_leg
        )
        logger.debug(f"Solving Leg {i} Lambert with max_revs={leg_max_revs}")
        try:
            sol = find_best_transfer(
                r1=body_states[i].position,
                v1_body=body_states[i].velocity,
                r2=body_states[i + 1].position,
                v2_body=body_states[i + 1].velocity,
                tof=leg_tofs[i],
                mu=mu_sun,
                max_revs=leg_max_revs,
                target_departure_vinf_km_s=required_departure_vinf,
                vinf_continuity_weight=10.0,
            )
            logger.debug(f"Leg {i} solved with n_revs={sol.n_revs}, branch={sol.branch}")
            leg_solutions.append(sol)
        except Exception as e:
            rec = TrajectoryRejectionRecord(
                trial_id=-1,
                departure_epoch=departure_epoch,
                leg_tofs=leg_tofs,
                current_leg=i,
                current_body=chain_bodies[i + 1],
                rejection_stage="lambert",
                reason_code="lambert_failed",
                incoming_v_inf=None,
                outgoing_v_inf=None,
                required_turn_angle=None,
                maximum_turn_angle=None,
                periapsis_radius=None,
                minimum_allowed_radius=None,
                max_revs_used=leg_max_revs,
                delta_v_cost=None,
            )
            return ChainResult(
                False,
                None,
                f"Leg {i} Lambert solve failed: {e}",
                leg_details=[],
                reason_code=RejectionReason.LAMBERT_FAILED,
                rejection_records=[rec],
                total_continuity_violation_km_s=0.0,
            )

        # After solving leg i (which arrives at body i+1):
        # If body i+1 is an intermediate unpowered flyby body, the DEPARTURE v∞ for
        # leg i+1 must equal the arrival v∞ at body i+1 (unpowered flyby continuity).
        # Compute this constraint now, before solving leg i+1.
        is_last_leg = i == len(leg_tofs) - 1
        if not is_last_leg:
            next_body_name = chain_bodies[i + 1].upper()
            next_spec = (
                flyby_specs.get(next_body_name) or flyby_specs.get(chain_bodies[i + 1]) or {}
            )
            is_intermediate_unpowered_flyby = (
                next_spec is not None
                and not next_spec.get("powered_allowed", False)
                and next_body_name not in (chain_bodies[0].upper(), chain_bodies[-1].upper())
            )
            if is_intermediate_unpowered_flyby:
                v_inf_arrival = sol.v2 - body_states[i + 1].velocity
                required_departure_vinf = float(np.linalg.norm(v_inf_arrival))
                logger.debug(
                    f"Leg {i + 1} continuity target: departure |v∞| = "
                    f"{required_departure_vinf:.4f} km/s "
                    f"(arrival at {next_body_name} from leg {i})"
                )
            else:
                required_departure_vinf = None  # powered flyby or destination — no constraint
        else:
            required_departure_vinf = None

    maneuvers: list[Maneuver] = []
    dsm_remaining = mission.dsm_budget_km_s
    leg_details: list[dict[str, Any]] = []

    # STEP 2: departure burn
    v_inf_dep = leg_solutions[0].v1 - body_states[0].velocity
    dv_tmi = departure_delta_v(v_inf_dep, mission.parking_altitude_km, chain_bodies[0])
    dv_tmi_vec = (v_inf_dep / max(float(np.linalg.norm(v_inf_dep)), 1e-10)) * dv_tmi
    maneuvers.append(Maneuver(epoch=epochs[0], delta_v=dv_tmi_vec, label="TMI"))

    # Resolve DSMs on legs
    dsm_resolutions = {}
    arrival_velocities = []

    for i in range(len(leg_tofs)):
        sol_original = leg_solutions[i]
        dsm_frac = dsm_fractions[i] if dsm_fractions is not None else None

        if dsm_frac is not None:
            leg_max_revs = (
                mission.leg_max_revs[i]
                if hasattr(mission, "leg_max_revs") and i < len(mission.leg_max_revs)
                else mission.max_revs_per_leg
            )
            logger.debug(f"Solving Leg {i} DSM Lambert with max_revs={leg_max_revs}")
            try:
                dsm_res = resolve_leg_with_dsm(
                    r1=body_states[i].position,
                    v1_original=sol_original.v1,
                    r2=body_states[i + 1].position,
                    v2_destination_body=body_states[i + 1].velocity,
                    t_start=epochs[i],
                    tof_leg=leg_tofs[i],
                    dsm_fraction=dsm_frac,
                    mu_sun=mu_sun,
                    max_revs=leg_max_revs,
                )
                logger.debug(f"Leg {i} DSM resolved: cost={dsm_res.dsm_delta_v_km_s:.4f} km/s")
                dsm_resolutions[i] = dsm_res
                dsm_remaining -= dsm_res.dsm_delta_v_km_s
                arrival_velocities.append(dsm_res.effective_arrival_velocity)
            except Exception as e:
                partial_violation = sum(
                    ld.get("magnitude_mismatch_km_s", 0.0) for ld in leg_details
                )
                return ChainResult(
                    False,
                    None,
                    f"Leg {i} DSM resolve failed: {e}",
                    leg_details=leg_details,
                    reason_code=RejectionReason.LAMBERT_FAILED,
                    total_continuity_violation_km_s=partial_violation,
                )

            if dsm_remaining < -1e-9:
                err_msg = (
                    f"Leg {i} DSM cost {dsm_res.dsm_delta_v_km_s:.4f} km/s "
                    f"exceeds remaining DSM budget "
                    f"{dsm_remaining + dsm_res.dsm_delta_v_km_s:.4f} km/s"
                )
                rec = TrajectoryRejectionRecord(
                    trial_id=-1,
                    departure_epoch=departure_epoch,
                    leg_tofs=leg_tofs,
                    current_leg=i,
                    current_body=chain_bodies[i + 1],
                    rejection_stage="dsm_budget",
                    reason_code="budget_exceeded",
                    incoming_v_inf=None,
                    outgoing_v_inf=None,
                    required_turn_angle=None,
                    maximum_turn_angle=None,
                    periapsis_radius=None,
                    minimum_allowed_radius=None,
                    max_revs_used=leg_max_revs,
                    delta_v_cost=dsm_res.dsm_delta_v_km_s,
                )
                partial_violation = sum(
                    ld.get("magnitude_mismatch_km_s", 0.0) for ld in leg_details
                )
                return ChainResult(
                    False,
                    None,
                    err_msg,
                    leg_details=leg_details,
                    reason_code=RejectionReason.BUDGET_EXCEEDED,
                    rejection_records=[rec],
                    total_continuity_violation_km_s=partial_violation,
                )
        else:
            arrival_velocities.append(sol_original.v2)

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
        v_inf_in = arrival_velocities[k - 1] - body_states[k].velocity
        v_inf_out_required = leg_solutions[k].v1 - body_states[k].velocity

        res, err, fail_dict = resolve_single_flyby_segment(
            body=body,
            v_inf_in=v_inf_in,
            v_inf_out_required=v_inf_out_required,
            min_alt_km=spec["min_alt_km"],
            max_alt_km=spec["max_alt_km"],
            powered_allowed=spec["powered_allowed"],
            max_powered_km_s=spec["max_powered_km_s"],
            dsm_budget_available=dsm_remaining,
            high_fidelity_ratio_threshold=high_fidelity_ratio_threshold,
            encounter_epoch=epochs[k],
            kernel=kernel,
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

            rec_rejection: TrajectoryRejectionRecord | None = None
            if fail_dict is not None:
                rec_rejection = TrajectoryRejectionRecord(
                    trial_id=-1,
                    departure_epoch=departure_epoch,
                    leg_tofs=leg_tofs,
                    current_leg=k,
                    current_body=body.upper(),
                    rejection_stage=fail_dict["rejection_stage"],
                    reason_code=fail_dict["reason_code"],
                    incoming_v_inf=fail_dict["incoming_v_inf"],
                    outgoing_v_inf=fail_dict["outgoing_v_inf"],
                    required_turn_angle=fail_dict["required_turn_angle"],
                    maximum_turn_angle=fail_dict["maximum_turn_angle"],
                    periapsis_radius=fail_dict["periapsis_radius"],
                    minimum_allowed_radius=fail_dict["minimum_allowed_radius"],
                    max_revs_used=None,
                    delta_v_cost=fail_dict["delta_v_cost"],
                )
            failed_mismatch = abs(
                float(np.linalg.norm(v_inf_in)) - float(np.linalg.norm(v_inf_out_required))
            )
            partial_violation = (
                sum(ld.get("magnitude_mismatch_km_s", 0.0) for ld in leg_details) + failed_mismatch
            )
            return ChainResult(
                False,
                None,
                err,
                leg_details,
                reason_code=reason_code,
                rejection_records=[rec_rejection] if rec_rejection is not None else [],
                total_continuity_violation_km_s=partial_violation,
            )

        dsm_remaining -= res["from_dsm"]
        v_inf_in_mag = float(np.linalg.norm(v_inf_in))
        dv_vec = (v_inf_in / max(v_inf_in_mag, 1e-10)) * res["dv_km_s"]

        is_powered = res["from_dsm"] < 1e-6
        label = f"FLY_{body.upper()}_POWERED" if is_powered else f"FLY_{body.upper()}_DSM"
        if res["dv_km_s"] < 1e-6:
            label = f"FLY_{body.upper()}"
        if res.get("is_hf", False):
            label += "_HF"

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
        if "is_hf" in res:
            leg_detail["is_hf"] = res["is_hf"]
            leg_detail["v_inf_out"] = res["v_inf_out"]
        leg_details.append(leg_detail)

    # STEP 4: arrival burn
    v_inf_arr = body_states[-1].velocity - arrival_velocities[-1]
    dv_moi = arrival_delta_v(
        v_inf_arr,
        mission.capture_altitude_km,
        chain_bodies[-1],
        apoapsis_km=mission.capture_apoapsis_km,
    )
    dv_moi_vec = (v_inf_arr / max(float(np.linalg.norm(v_inf_arr)), 1e-10)) * dv_moi
    maneuvers.append(Maneuver(epoch=epochs[-1], delta_v=dv_moi_vec, label="MOI"))

    # Add DSM maneuvers
    for i in range(len(leg_tofs)):
        if i in dsm_resolutions:
            dsm_res = dsm_resolutions[i]
            maneuvers.append(
                Maneuver(
                    epoch=dsm_res.dsm_epoch,
                    delta_v=dsm_res.dsm_delta_v_vector,
                    label=f"DSM_LEG_{i}",
                )
            )

    maneuvers.sort(key=lambda m: m.epoch)

    # Build states list chronologically
    states = []
    for i in range(len(leg_tofs)):
        states.append(
            OrbitalState(
                epoch=epochs[i],
                position=body_states[i].position.copy(),
                velocity=leg_solutions[i].v1.copy(),
                central_body=CelestialBody.SUN,
            )
        )
        if i in dsm_resolutions:
            dsm_res = dsm_resolutions[i]
            states.append(
                OrbitalState(
                    epoch=dsm_res.dsm_epoch,
                    position=dsm_res.dsm_position.copy(),
                    velocity=dsm_res.v_after_dsm.copy(),
                    central_body=CelestialBody.SUN,
                )
            )
    states.append(
        OrbitalState(
            epoch=epochs[-1],
            position=body_states[-1].position.copy(),
            velocity=arrival_velocities[-1].copy(),
            central_body=CelestialBody.SUN,
        )
    )

    trajectory = Trajectory(
        states=states,
        maneuvers=maneuvers,
        metadata={"chain": chain_bodies, "leg_details": leg_details},
    )
    total_violation = sum(ld.get("magnitude_mismatch_km_s", 0.0) for ld in leg_details)
    return ChainResult(
        True,
        trajectory,
        None,
        leg_details,
        reason_code=None,
        total_continuity_violation_km_s=total_violation,
    )
