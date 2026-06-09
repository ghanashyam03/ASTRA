from __future__ import annotations

from dataclasses import dataclass
from astra.state.trajectory import Trajectory
from astra.state.orbital_state import PHYSICAL_RADIUS, CelestialBody


@dataclass
class PhysicalConstraintResult:
    constraint_type: str
    body: str | None
    limit_km: float
    actual_km: float
    satisfied: bool
    violation_km: float  # positive if violated, 0 if satisfied


def check_min_periapsis(
    trajectory: Trajectory,
    min_periapsis_km: float,
    body: str | None = None,
) -> PhysicalConstraintResult:
    """Checks that trajectory.maneuvers do not imply a periapsis below minimum.
    For the purpose of this check, look in trajectory.metadata for keys:
    "parking_altitude_km" and "capture_altitude_km".
    """
    metadata = trajectory.metadata or {}
    b_name = body if body else "EARTH"
    
    try:
        origin_body = CelestialBody[b_name.upper()]
    except (KeyError, AttributeError):
        if isinstance(body, CelestialBody):
            origin_body = body
        else:
            origin_body = CelestialBody.EARTH

    if "capture_altitude_km" in metadata and b_name.upper() != "EARTH":
        actual_periapsis = PHYSICAL_RADIUS[origin_body] + metadata["capture_altitude_km"]
    elif "parking_altitude_km" in metadata:
        actual_periapsis = PHYSICAL_RADIUS[origin_body] + metadata["parking_altitude_km"]
    else:
        actual_periapsis = 0.0

    satisfied = actual_periapsis >= min_periapsis_km
    violation_km = max(0.0, min_periapsis_km - actual_periapsis)

    return PhysicalConstraintResult(
        constraint_type="min_periapsis",
        body=body,
        limit_km=min_periapsis_km,
        actual_km=actual_periapsis,
        satisfied=satisfied,
        violation_km=violation_km,
    )


def check_max_delta_v(
    trajectory: Trajectory,
    max_dv_km_s: float,
) -> PhysicalConstraintResult:
    """Checks that total delta-v does not exceed maximum."""
    actual = trajectory.delta_v_total
    satisfied = actual <= max_dv_km_s
    violation_km = max(0.0, actual - max_dv_km_s)
    
    return PhysicalConstraintResult(
        constraint_type="max_delta_v",
        body=None,
        limit_km=max_dv_km_s,
        actual_km=actual,
        satisfied=satisfied,
        violation_km=violation_km,
    )
