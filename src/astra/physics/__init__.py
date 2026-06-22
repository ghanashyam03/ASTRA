"""ASTRA Physics Subsystem.
Provides SPICE ephemerides, Lambert solvers, numerical propagation, and integrators.
"""
from __future__ import annotations

from astra.physics.ephemeris import (
    EphemerisEngine,
    EphemerisTarget,
    resolve_central_body,
    TargetType,
)

from astra.physics.exceptions import (
    InvalidEphemerisError,
    LambertConvergenceError,
    LambertError,
    LambertSingularityError,
    PhysicsError,
    PropagationError,
)

from astra.physics.flyby import (
    bplane_vector,
    build_bplane_frame,
    check_flyby_feasibility,
    compute_flyby,
    compute_flyby_turn_angle,
    FlybyFeasibility,
    FlybyResult,
    impact_parameter_from_periapsis,
    max_achievable_turn_angle,
    max_achievable_turn_angle_with_unlimited_burn,
    orbit_normal_from_bvector,
    periapsis_from_impact_parameter,
    periapsis_from_turn_angle,
    SAFE_FLYBY_ALTITUDE_KM,
)

from astra.physics.kernel import PhysicsKernel

from astra.physics.lambert import (
    LambertSolution,
    find_best_transfer,
    lambert_izzo,
    lambert_izzo_multirev,
    lambert_min_tof_multirev,
)

from astra.physics.maneuvers import (
    arrival_delta_v,
    c3_from_vinf,
    departure_delta_v,
    hyperbolic_excess_speed,
)

from astra.physics.propagator import (
    IntegrationResult,
    Integrator,
    RK45Integrator,
    propagate_to_times,
    propagate_two_body,
)

from astra.physics.soi import (
    DEFAULT_PARKING_ALTITUDE_KM,
    SOIResult,
    compute_soi_radius,
    get_default_parking_altitude,
    is_in_soi,
)

__all__ = [
    # Upper‑case names (alphabetical)
    "DEFAULT_PARKING_ALTITUDE_KM",
    "EphemerisEngine",
    "EphemerisTarget",
    "FlybyFeasibility",
    "FlybyResult",
    "IntegrationResult",
    "LambertConvergenceError",
    "LambertError",
    "LambertSolution",
    "LambertSingularityError",
    "PhysicsError",
    "PhysicsKernel",
    "RK45Integrator",
    "SOIResult",
    "TargetType",
    "arrival_delta_v",
    "bplane_vector",
    "build_bplane_frame",
    "c3_from_vinf",
    "check_flyby_feasibility",
    "compute_flyby",
    "compute_flyby_turn_angle",
    "compute_soi_radius",
    "departure_delta_v",
    "find_best_transfer",
    "get_default_parking_altitude",
    "hyperbolic_excess_speed",
    "impact_parameter_from_periapsis",
    "is_in_soi",
    "max_achievable_turn_angle",
    "max_achievable_turn_angle_with_unlimited_burn",
    "periapsis_from_impact_parameter",
    "periapsis_from_turn_angle",
    "propagate_to_times",
    "propagate_two_body",
    "PropagationError",
    "SAFE_FLYBY_ALTITUDE_KM",
    "Integrator",
    "InvalidEphemerisError",
    "lambert_izzo",
    "lambert_izzo_multirev",
    "lambert_min_tof_multirev",
    "orbit_normal_from_bvector",
    "resolve_central_body",
]
