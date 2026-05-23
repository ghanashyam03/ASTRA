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
from astra.physics.kernel import PhysicsKernel
from astra.physics.lambert import lambert_izzo
from astra.physics.propagator import (
    IntegrationResult,
    Integrator,
    RK45Integrator,
    propagate_to_times,
    propagate_two_body,
)

__all__ = [
    "PhysicsKernel",
    "EphemerisEngine",
    "EphemerisTarget",
    "TargetType",
    "resolve_central_body",
    "lambert_izzo",
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
]
