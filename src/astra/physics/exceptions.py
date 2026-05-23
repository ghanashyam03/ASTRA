"""Custom scientific exceptions for the ASTRA physics core.
Ensures that all physical and numerical failures are explicit and loud.
"""
from __future__ import annotations

class PhysicsError(Exception):
    """Base exception for all ASTRA physics core errors."""
    pass

class LambertError(PhysicsError):
    """Base exception for all Lambert solver errors."""
    pass

class LambertConvergenceError(LambertError):
    """Raised when the Lambert solver fails to converge within iteration limits."""
    pass

class LambertSingularityError(LambertError):
    """Raised when the requested Lambert transfer geometry is singular or collinear."""
    pass

class PropagationError(PhysicsError):
    """Raised when numerical propagation encounters a singularity, collision, or solver failure."""
    pass

class InvalidEphemerisError(PhysicsError):
    """Raised when ephemeris target, observer, or kernel data is invalid or missing."""
    pass
