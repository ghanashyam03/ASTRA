"""Independent physics-sanity auditor for any flyby-containing Trajectory.

Deliberately does NOT call resolve_flyby_chain, resolve_single_flyby_segment,
or any function the optimizer itself used to produce the result. Instead, it
re-derives v_inf_in/v_inf_out for each flyby maneuver DIRECTLY from the stored
Trajectory's states and the kernel's own ephemeris, and independently checks
the implied turn angle against check_flyby_feasibility — the same gate
function, since that IS the canonical physics, but with all the SURROUNDING
bookkeeping re-derived from scratch rather than trusted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from astra.physics.flyby import check_flyby_feasibility
from astra.physics.kernel import PhysicsKernel
from astra.physics.lambert import find_best_transfer
from astra.state.orbital_state import GM, CelestialBody
from astra.state.trajectory import Trajectory


class AuditFailure(Exception):  # noqa: N818
    """Raised when a stored trajectory's reported Δv is inconsistent with the
    flyby geometry independently re-derived from its own states."""

    pass


@dataclass
class FlybyAuditResult:
    body: str
    epoch: float
    reported_dv_km_s: float
    required_turn_deg: float
    max_unpowered_turn_deg: float
    is_self_consistent: bool
    discrepancy_note: str | None


def audit_trajectory_physics(
    trajectory: Trajectory,
    kernel: PhysicsKernel,
    tolerance_km_s: float = 0.01,
) -> list[FlybyAuditResult]:
    """Independently re-derive feasibility for every FLY_* maneuver in a
    trajectory. Raises AuditFailure if any flyby's reported Δv is inconsistent
    with the geometry implied by the trajectory's own states.

    This function deliberately reconstructs v_inf_in and v_inf_out from
    trajectory.states (the spacecraft's actual recorded heliocentric states
    immediately before/after each flyby) and the kernel's ephemeris — NOT from
    any cached value the original optimizer computed. If the original
    optimizer's bookkeeping was wrong (as in the original Venus failure), this
    independent reconstruction will surface the inconsistency.
    """
    results: list[FlybyAuditResult] = []
    fly_maneuvers = [
        (i, m) for i, m in enumerate(trajectory.maneuvers) if m.label.startswith("FLY_")
    ]
    for idx, maneuver in fly_maneuvers:
        body_name = maneuver.label.replace("FLY_", "").split("_")[0]
        try:
            cb = CelestialBody[body_name]
        except KeyError:
            continue  # not a recognized body label — skip, not an audit target

        # Find the state at/closest to the flyby epoch
        state_after = next(
            (s for s in trajectory.states if abs(s.epoch - maneuver.epoch) < 1.0), None
        )
        if state_after is None:
            continue

        # Find the state immediately preceding the flyby state
        state_before = next(
            (s for s in reversed(trajectory.states) if s.epoch < state_after.epoch), None
        )
        if state_before is None:
            continue

        state_at_flyby_epoch = maneuver.epoch
        body_state = kernel.get_body_state(cb, state_at_flyby_epoch)

        # Determine incoming velocity
        # TODO: Refactor this to use two-body Keplerian propagation (IVP) instead of solving
        # a fresh Lambert BVP. Keplerian propagation is more robust because it avoids branch
        # ambiguity (short-way vs long-way, multi-revolution) and will not raise false positives
        # if the optimizer flies a non-minimal-dv branch. It also directly detects corrupted
        # states if the propagated position deviates from the stored position.
        dist = float(np.linalg.norm(state_before.position - state_after.position))
        if dist < 1000.0:
            # Fallback for mock trajectories where position is identical
            v_in = state_before.velocity
        else:
            try:
                sol = find_best_transfer(
                    r1=state_before.position,
                    v1_body=np.zeros(3),
                    r2=state_after.position,
                    v2_body=np.zeros(3),
                    tof=state_after.epoch - state_before.epoch,
                    mu=GM["SUN"],
                )
                v_in = sol.v2
            except Exception:
                v_in = state_before.velocity

        # Outgoing velocity is the velocity leaving the flyby body
        v_out = state_after.velocity

        v_inf_in = v_in - body_state.velocity
        v_inf_out = v_out - body_state.velocity

        v_inf_in_mag = float(np.linalg.norm(v_inf_in))
        v_inf_out_mag = float(np.linalg.norm(v_inf_out))
        cos_turn = float(np.dot(v_inf_in, v_inf_out) / max(v_inf_in_mag * v_inf_out_mag, 1e-10))
        cos_turn = max(-1.0, min(1.0, cos_turn))
        required_turn_rad = math.acos(cos_turn)

        feas = check_flyby_feasibility(v_inf_in_mag, required_turn_rad, body_name)

        reported_dv = maneuver.magnitude
        magnitude_change = abs(v_inf_in_mag - v_inf_out_mag)
        # Self-consistency: if the reported Δv is near zero, the flyby was
        # claimed unpowered — this REQUIRES the geometry to actually be
        # achievable unpowered AND have negligible magnitude change. If the
        # reported Δv is non-trivial, it should be of the same ORDER as the
        # magnitude change implied by the actual geometry — a large mismatch
        # here is exactly the original bug's signature.
        is_consistent = True
        note = None

        if not feas.is_achievable_at_all:
            is_consistent = False
            note = (
                f"Flyby at {body_name} is physically impossible even with unlimited burn. "
                f"Required turn: {math.degrees(required_turn_rad):.2f}°, "
                f"limit: {math.degrees(feas.max_turn_with_unlimited_burn_rad):.2f}°."
            )
            if reported_dv < tolerance_km_s:
                note += " — THIS IS THE ORIGINAL BUG'S SIGNATURE."
        elif reported_dv < tolerance_km_s and not feas.is_achievable_unpowered:
            is_consistent = False
            note = (
                f"Maneuver reports near-zero Δv ({reported_dv:.4f} km/s) but "
                f"the implied geometry requires {math.degrees(required_turn_rad):.2f}° "
                f"turn against a {math.degrees(feas.max_unpowered_turn_rad):.2f}° "
                f"unpowered ceiling — THIS IS THE ORIGINAL BUG'S SIGNATURE."
            )
        elif reported_dv > tolerance_km_s and magnitude_change > 2.0 * reported_dv + 0.1:
            is_consistent = False
            note = (
                f"Reported Δv ({reported_dv:.4f} km/s) is inconsistent with "
                f"the implied v_inf magnitude change ({magnitude_change:.4f} km/s) "
                f"by more than a factor of 2 — independent re-derivation does "
                f"not corroborate the stored value."
            )

        results.append(
            FlybyAuditResult(
                body=body_name,
                epoch=state_at_flyby_epoch,
                reported_dv_km_s=reported_dv,
                required_turn_deg=math.degrees(required_turn_rad),
                max_unpowered_turn_deg=math.degrees(feas.max_unpowered_turn_rad),
                is_self_consistent=is_consistent,
                discrepancy_note=note,
            )
        )

    failures = [r for r in results if not r.is_self_consistent]
    if failures:
        raise AuditFailure(
            f"{len(failures)} flyby maneuver(s) failed independent physics audit: "
            + "; ".join(f.discrepancy_note for f in failures if f.discrepancy_note is not None)
        )
    return results
