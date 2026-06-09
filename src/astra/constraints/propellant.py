from __future__ import annotations

import math
from dataclasses import dataclass
from astra.state.trajectory import Trajectory
from astra.state.spacecraft import Spacecraft


@dataclass
class PropellantConstraintResult:
    required_dv_km_s: float
    available_dv_km_s: float
    required_propellant_kg: float
    available_propellant_kg: float
    satisfied: bool
    margin_kg: float  # available - required, negative if violated


def check_propellant_budget(
    trajectory: Trajectory,
    spacecraft: Spacecraft,
) -> PropellantConstraintResult:
    """Evaluates the propellant budget constraint using the Tsiolkovsky rocket equation."""
    ve = spacecraft.propulsion.exhaust_velocity  # km/s
    dv_required = trajectory.delta_v_total       # km/s
    dv_available = spacecraft.delta_v_budget()   # km/s
    
    if ve > 0:
        mass_ratio_required = math.exp(dv_required / ve)
        m0 = spacecraft.total_mass_kg
        m_prop_required = m0 * (1 - 1.0 / mass_ratio_required)
    else:
        m_prop_required = 0.0
        
    m_prop_available = spacecraft.propulsion.propellant_mass_kg
    
    satisfied = dv_required <= dv_available
    margin_kg = m_prop_available - m_prop_required

    return PropellantConstraintResult(
        required_dv_km_s=dv_required,
        available_dv_km_s=dv_available,
        required_propellant_kg=m_prop_required,
        available_propellant_kg=m_prop_available,
        satisfied=satisfied,
        margin_kg=margin_kg,
    )
