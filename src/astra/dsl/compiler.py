"""Compile MissionDSL into strongly-typed domain objects."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC
from typing import TYPE_CHECKING

from astra.dsl.schema import ConstraintType, MissionDSL
from astra.state.orbital_state import CelestialBody
from astra.state.spacecraft import PropulsionSystem, PropulsionType, Spacecraft

if TYPE_CHECKING:
    from astra.physics.ephemeris import EphemerisEngine


@dataclass
class CompiledConstraint:
    type: ConstraintType
    limit: float        # the numeric threshold
    hard: bool
    body: str | None = None

@dataclass
class CompiledObjective:
    metric: str
    direction: str      # "minimize" or "maximize"
    weight: float

@dataclass
class CompiledMission:
    mission_id: str
    spacecraft: Spacecraft
    origin_body: CelestialBody
    destination_body: CelestialBody
    departure_epoch_start: float   # J2000 seconds
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

    # Bodies
    origin = CelestialBody[dsl.trajectory.origin.body.upper()]
    destination = CelestialBody[dsl.trajectory.destination.body.upper()]

    # Constraints
    constraints = []
    for c in dsl.constraints:
        if c.type == ConstraintType.MAX_DELTA_V:
            constraints.append(CompiledConstraint(
                type=c.type, limit=c.value_km_s or 99.0, hard=c.hard))
        elif c.type == ConstraintType.MAX_DURATION:
            constraints.append(CompiledConstraint(
                type=c.type, limit=(c.value_days or 999) * 86400.0, hard=c.hard))
        elif c.type == ConstraintType.MIN_PERIAPSIS:
            constraints.append(CompiledConstraint(
                type=c.type, limit=c.value_km or 0.0, hard=c.hard, body=c.body))

    objectives = [
        CompiledObjective(
            metric=o.metric.value,
            direction=o.direction.value,
            weight=o.weight,
        ) for o in dsl.objectives
    ]

    from astra.physics.soi import get_default_parking_altitude
    h_park = get_default_parking_altitude(dsl.trajectory.origin.body)
    if dsl.trajectory.origin.orbit is not None:
        if dsl.trajectory.origin.orbit.altitude_km is not None:
            h_park = dsl.trajectory.origin.orbit.altitude_km
        elif dsl.trajectory.origin.orbit.periapsis_km is not None:
            h_park = dsl.trajectory.origin.orbit.periapsis_km

    h_cap = get_default_parking_altitude(dsl.trajectory.destination.body)
    capture_apoapsis_km = None
    if dsl.trajectory.destination.orbit is not None:
        orbit = dsl.trajectory.destination.orbit
        if dsl.trajectory.destination.orbit.altitude_km is not None:
            h_cap = dsl.trajectory.destination.orbit.altitude_km
        elif dsl.trajectory.destination.orbit.periapsis_km is not None:
            h_cap = dsl.trajectory.destination.orbit.periapsis_km
        
        if orbit.type == "elliptical" and orbit.apoapsis_km is not None:
            capture_apoapsis_km = orbit.apoapsis_km
        else:
            capture_apoapsis_km = None

    day = 86400.0
    return CompiledMission(
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
