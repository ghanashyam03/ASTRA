"""Compile MissionDSL into strongly-typed domain objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC
from typing import TYPE_CHECKING, Any

from astra.dsl.schema import ConstraintType, MissionDSL
from astra.state.orbital_state import CelestialBody
from astra.state.spacecraft import PropulsionSystem, PropulsionType, Spacecraft

if TYPE_CHECKING:
    from astra.physics.ephemeris import EphemerisEngine


@dataclass
class CompiledConstraint:
    type: ConstraintType
    limit: float  # the numeric threshold
    hard: bool
    body: str | None = None


@dataclass
class CompiledObjective:
    metric: str
    direction: str  # "minimize" or "maximize"
    weight: float


@dataclass
class CompiledMission:
    mission_id: str
    spacecraft: Spacecraft
    origin_body: CelestialBody
    destination_body: CelestialBody
    departure_epoch_start: float  # J2000 seconds
    departure_epoch_end: float
    tof_min_seconds: float
    tof_max_seconds: float
    tof_step_seconds: float
    constraints: list[CompiledConstraint]
    objectives: list[CompiledObjective]
    seed: int
    max_evaluations: int
    parking_altitude_km: float = 200.0
    capture_altitude_km: float = 300.0
    capture_apoapsis_km: float | None = None
    flyby_sequence: list[dict[str, Any]] = field(default_factory=list)
    dsm_budget_km_s: float = 0.0
    max_revs_per_leg: int = 0
    leg_tof_bounds: list[tuple[float, float]] = field(default_factory=list)
    # list length = len(flyby_sequence) + 1 (one bound per leg, including final leg to destination)
    leg_max_revs: list[int] = field(default_factory=list)
    # list length = len(flyby_sequence) + 1


def compile_mission(dsl: MissionDSL, ephemeris: EphemerisEngine | None = None) -> CompiledMission:
    """Compile MissionDSL → CompiledMission domain object.
    ephemeris: optional EphemerisEngine for epoch conversion.
    If None, uses Python datetime → J2000 conversion."""
    # Epoch conversion
    if ephemeris is not None:
        dep_start = ephemeris.epoch_from_date(dsl.launch_window.start.strftime("%Y-%m-%dT%H:%M:%S"))
        dep_end = ephemeris.epoch_from_date(dsl.launch_window.end.strftime("%Y-%m-%dT%H:%M:%S"))
    else:
        # Fallback: compute J2000 seconds from datetime
        from datetime import datetime

        J2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=UTC)
        dep_start = (dsl.launch_window.start.replace(tzinfo=UTC) - J2000).total_seconds()
        dep_end = (dsl.launch_window.end.replace(tzinfo=UTC) - J2000).total_seconds()

    # Spacecraft
    prop = PropulsionSystem(
        type=PropulsionType(dsl.spacecraft.propulsion.type.value),
        isp_seconds=dsl.spacecraft.propulsion.isp_seconds,
        thrust_newtons=dsl.spacecraft.propulsion.thrust_newtons,
        propellant_mass_kg=dsl.spacecraft.fuel_mass_kg,
    )
    sc = Spacecraft(
        name=dsl.spacecraft.name,
        dry_mass_kg=dsl.spacecraft.dry_mass_kg,
        propulsion=prop,
    )

    # Trajectory schema choice (precedence to multi_body_trajectory)
    traj = dsl.multi_body_trajectory if dsl.multi_body_trajectory is not None else dsl.trajectory

    # Bodies
    origin = CelestialBody[traj.origin.body.upper()]
    destination = CelestialBody[traj.destination.body.upper()]

    # Constraints
    constraints = []
    for c in dsl.constraints:
        if c.type == ConstraintType.MAX_DELTA_V:
            constraints.append(
                CompiledConstraint(type=c.type, limit=c.value_km_s or 99.0, hard=c.hard)
            )
        elif c.type == ConstraintType.MAX_DURATION:
            constraints.append(
                CompiledConstraint(type=c.type, limit=(c.value_days or 999) * 86400.0, hard=c.hard)
            )
        elif c.type == ConstraintType.MIN_PERIAPSIS:
            constraints.append(
                CompiledConstraint(type=c.type, limit=c.value_km or 0.0, hard=c.hard, body=c.body)
            )

    objectives = [
        CompiledObjective(
            metric=o.metric.value,
            direction=o.direction.value,
            weight=o.weight,
        )
        for o in dsl.objectives
    ]

    from astra.physics.soi import get_default_parking_altitude

    h_park = get_default_parking_altitude(traj.origin.body)
    if traj.origin.orbit is not None:
        if traj.origin.orbit.altitude_km is not None:
            h_park = traj.origin.orbit.altitude_km
        elif traj.origin.orbit.periapsis_km is not None:
            h_park = traj.origin.orbit.periapsis_km

    h_cap = get_default_parking_altitude(traj.destination.body)
    capture_apoapsis_km = None
    if traj.destination.orbit is not None:
        orbit = traj.destination.orbit
        if traj.destination.orbit.altitude_km is not None:
            h_cap = traj.destination.orbit.altitude_km
        elif traj.destination.orbit.periapsis_km is not None:
            h_cap = traj.destination.orbit.periapsis_km

        if orbit.type == "elliptical" and orbit.apoapsis_km is not None:
            capture_apoapsis_km = orbit.apoapsis_km
        else:
            capture_apoapsis_km = None

    day = 86400.0
    compiled = CompiledMission(
        mission_id=dsl.mission_id,
        spacecraft=sc,
        origin_body=origin,
        destination_body=destination,
        departure_epoch_start=dep_start,
        departure_epoch_end=dep_end,
        tof_min_seconds=dsl.launch_window.tof_min_days * day,
        tof_max_seconds=dsl.launch_window.tof_max_days * day,
        tof_step_seconds=dsl.launch_window.resolution_days * day,
        constraints=constraints,
        objectives=objectives,
        seed=dsl.optimization.seed,
        max_evaluations=dsl.optimization.budget.max_evaluations,
        parking_altitude_km=h_park,
        capture_altitude_km=h_cap,
        capture_apoapsis_km=capture_apoapsis_km,
    )

    if dsl.multi_body_trajectory is not None:
        mbt = dsl.multi_body_trajectory
        compiled.flyby_sequence = [
            {
                "body": fb.body.upper(),
                "min_alt_km": fb.min_periapsis_alt_km,
                "max_alt_km": fb.max_periapsis_alt_km,
                "powered_allowed": fb.powered_burn_allowed,
                "max_powered_km_s": fb.max_powered_burn_km_s,
            }
            for fb in mbt.flyby_sequence
        ]
        compiled.dsm_budget_km_s = mbt.dsm_budget_km_s
        compiled.max_revs_per_leg = mbt.max_revs_per_leg

        day = 86400.0
        global_tof_min = dsl.launch_window.tof_min_days
        global_tof_max = dsl.launch_window.tof_max_days
        global_max_revs = mbt.max_revs_per_leg

        # Build leg-level TOF bounds and max_revs:
        # Leg i goes from body[i] to body[i+1].
        # flyby_sequence[i] carries the TOF-bounds for the approach INTO body[i+1].
        # The final leg (last flyby body → destination) has no explicit override;
        # use global bounds.
        leg_tof_bounds: list[tuple[float, float]] = []
        leg_max_revs_list: list[int] = []
        for fb in mbt.flyby_sequence:
            tmin = (fb.tof_min_days if fb.tof_min_days is not None else global_tof_min) * day
            tmax = (fb.tof_max_days if fb.tof_max_days is not None else global_tof_max) * day
            leg_tof_bounds.append((tmin, tmax))
            leg_max_revs_list.append(fb.max_revs if fb.max_revs is not None else global_max_revs)
        # Final leg: origin/last-flyby → destination, always use global
        leg_tof_bounds.append((global_tof_min * day, global_tof_max * day))
        leg_max_revs_list.append(global_max_revs)

        compiled.leg_tof_bounds = leg_tof_bounds
        compiled.leg_max_revs = leg_max_revs_list
        # Also update flyby_sequence entries with their max_revs for backward compat
        for i, fb in enumerate(mbt.flyby_sequence):
            compiled.flyby_sequence[i]["max_revs"] = leg_max_revs_list[i]

    return compiled
