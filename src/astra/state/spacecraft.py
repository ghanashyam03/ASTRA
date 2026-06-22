from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class PropulsionType(StrEnum):
    CHEMICAL = "chemical"
    ELECTRIC = "electric"
    HYBRID = "hybrid"


@dataclass
class PropulsionSystem:
    type: PropulsionType
    isp_seconds: float  # specific impulse
    thrust_newtons: float  # 0.0 for impulsive model
    propellant_mass_kg: float

    @property
    def exhaust_velocity(self) -> float:
        """Ve = Isp × g0 [km/s]."""
        return self.isp_seconds * 9.80665e-3  # convert m/s → km/s


@dataclass
class Spacecraft:
    name: str
    dry_mass_kg: float
    propulsion: PropulsionSystem

    @property
    def total_mass_kg(self) -> float:
        return self.dry_mass_kg + self.propulsion.propellant_mass_kg

    @property
    def mass_ratio(self) -> float:
        return self.total_mass_kg / self.dry_mass_kg

    def delta_v_budget(self) -> float:
        """Tsiolkovsky rocket equation [km/s]."""
        import math

        return self.propulsion.exhaust_velocity * math.log(self.mass_ratio)
