"""ASTRA Physics Subsystem.
Provides SPICE ephemerides, Lambert solvers, numerical propagation, and integrators.
"""
from __future__ import annotations

from astra.physics.ephemeris import (
    EphemerisEngine,
    EphemerisTarget,
    TargetType,
    resolve_central_body,
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
    SAFE_FLYBY_ALTITUDE_KM,
    FlybyResult,
    compute_flyby,
    compute_flyby_turn_angle,
    periapsis_from_impact_parameter,
    impact_parameter_from_periapsis,
    periapsis_from_turn_angle,
    max_achievable_turn_angle,
    max_achievable_turn_angle_with_unlimited_burn,
    check_flyby_feasibility,
    build_bplane_frame,
    orbit_normal_from_bvector,
    bplane_vector,
    FlybyFeasibility,
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
    "PhysicsKernel",
    "EphemerisEngine",
    "EphemerisTarget",
    "TargetType",
    "resolve_central_body",
    "lambert_izzo",
    "lambert_izzo_multirev",
    "lambert_min_tof_multirev",
    "find_best_transfer",
    "LambertSolution",
    "propagate_two_body",
    "propagate_to_times",
    "Integrator",
    "RK45Integrator",
    "IntegrationResult",
    "PhysicsError",
    "LambertError",
    "LambertConvergenceError",
    "LambertSingularityError",
    "PropagationError",
    "InvalidEphemerisError",
    "departure_delta_v",
    "arrival_delta_v",
    "c3_from_vinf",
    "hyperbolic_excess_speed",
    "compute_soi_radius",
    "is_in_soi",
    "get_default_parking_altitude",
    "DEFAULT_PARKING_ALTITUDE_KM",
    "SOIResult",
    "compute_flyby",
    "compute_flyby_turn_angle",
    "FlybyResult",
    "SAFE_FLYBY_ALTITUDE_KM",
    "periapsis_from_impact_parameter",
    "impact_parameter_from_periapsis",
    "periapsis_from_turn_angle",
    "max_achievable_turn_angle",
    "max_achievable_turn_angle_with_unlimited_burn",
    "check_flyby_feasibility",
    "build_bplane_frame",
    "orbit_normal_from_bvector",
    "bplane_vector",
    "FlybyFeasibility",
]
