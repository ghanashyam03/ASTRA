"""Solar Radiation Pressure force model."""
from __future__ import annotations

import numpy as np

from astra.physics.forces.gravity import ForceModel


class SolarRadiationPressure(ForceModel):
    """Solar Radiation Pressure (SRP) force model.

    Assumes the spacecraft is always in sunlight (no planetary shadow/eclipse model).
    """

    def __init__(
        self,
        area_m2: float,
        mass_kg: float,
        Cr: float = 1.8,  # noqa: N803
        AU: float = 1.496e8,  # noqa: N803
        P_solar: float = 4.56e-6,  # noqa: N803
    ) -> None:
        """Initialize SolarRadiationPressure model.

        Parameters
        ----------
        area_m2 : float
            Spacecraft cross-sectional area in m^2.
        mass_kg : float
            Spacecraft mass in kg.
        Cr : float, optional
            Reflectivity coefficient (1.0 = absorb, 2.0 = reflect), by default 1.8.
        AU : float, optional
            Astronomical Unit distance in km, by default 1.496e8.
        P_solar : float, optional
            Solar pressure at 1 AU in N/m^2, by default 4.56e-6.
        """
        self.area_m2 = float(area_m2)
        self.mass_kg = float(mass_kg)
        self.Cr = float(Cr)
        self.AU = float(AU)
        self.P_solar = float(P_solar)

    def acceleration(self, state_vec: np.ndarray, t: float) -> np.ndarray:
        """Compute SRP acceleration.

        Parameters
        ----------
        state_vec : np.ndarray
            6-element state vector [x, y, z, vx, vy, vz] in km and km/s.
            Position components represent the displacement relative to the Sun.
        t : float
            Current time in seconds.

        Returns
        -------
        np.ndarray
            3-element acceleration vector [ax, ay, az] in km/s^2.
        """
        r_sc = state_vec[:3]
        r_mag = float(np.linalg.norm(r_sc))
        if r_mag < 1e-6:
            return np.zeros(3, dtype=np.float64)

        sun_direction = -r_sc / r_mag
        pressure = self.P_solar * ((self.AU / r_mag) ** 2)
        a_srp = (self.Cr * self.area_m2 * pressure / self.mass_kg) * sun_direction * 1e-3
        return a_srp.astype(np.float64)
