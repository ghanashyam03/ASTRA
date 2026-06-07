"""Modular force models package for orbital perturbations."""
from __future__ import annotations

from typing import TYPE_CHECKING

from astra.physics.forces.drag import ATMOSPHERE_CONSTANTS, AtmosphericDrag
from astra.physics.forces.gravity import J2_CONSTANTS, ForceModel, J2Perturbation, PointMassGravity
from astra.physics.forces.srp import SolarRadiationPressure

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np


def build_ode(forces: list[ForceModel]) -> Callable[[float, np.ndarray], np.ndarray]:
    """Build a system of differential equations compatible with solve_ivp.

    This is a lazy wrapper to prevent circular import issues between
    the propagator and forces modules.
    """
    from astra.physics.propagator import build_ode as _build_ode
    return _build_ode(forces)


__all__ = [
    "ForceModel",
    "PointMassGravity",
    "J2Perturbation",
    "SolarRadiationPressure",
    "AtmosphericDrag",
    "J2_CONSTANTS",
    "ATMOSPHERE_CONSTANTS",
    "build_ode",
]
