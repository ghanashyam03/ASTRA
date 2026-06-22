"""Atmospheric drag force model."""

from __future__ import annotations

import numpy as np

from astra.physics.forces.gravity import ForceModel


class AtmosphericDrag(ForceModel):
    """Atmospheric drag force model assuming an exponential density profile."""

    def __init__(
        self,
        area_m2: float,
        mass_kg: float,
        Cd: float = 2.2,  # noqa: N803
        body: str = "EARTH",
    ) -> None:
        """Initialize AtmosphericDrag model.

        Parameters
        ----------
        area_m2 : float
            Spacecraft cross-sectional area in m^2.
        mass_kg : float
            Spacecraft mass in kg.
        Cd : float, optional
            Drag coefficient, by default 2.2.
        body : str, optional
            Name of central body (e.g. "EARTH", "MARS"), by default "EARTH".

        Raises
        ------
        ValueError
            If atmospheric parameters for the specified body are not available.
        """
        self.area_m2 = float(area_m2)
        self.mass_kg = float(mass_kg)
        self.Cd = float(Cd)

        body_upper = body.upper()
        if body_upper not in ATMOSPHERE_CONSTANTS:
            raise ValueError(f"Atmospheric drag parameters not available for body: {body}")

        from astra.state.orbital_state import PHYSICAL_RADIUS, CelestialBody

        try:
            self.body_enum = CelestialBody[body_upper]
        except KeyError:
            raise ValueError(f"Unknown celestial body name: {body}")

        self.R_body = PHYSICAL_RADIUS[self.body_enum]

        constants = ATMOSPHERE_CONSTANTS[body_upper]
        self.rho0 = constants["rho0"]
        self.H = constants["H"]
        self.cutoff_km = constants["cutoff_km"]

    def acceleration(self, state_vec: np.ndarray, t: float) -> np.ndarray:
        """Compute atmospheric drag acceleration.

        Parameters
        ----------
        state_vec : np.ndarray
            6-element state vector [x, y, z, vx, vy, vz] in km and km/s.
        t : float
            Current time in seconds.

        Returns
        -------
        np.ndarray
            3-element acceleration vector [ax, ay, az] in km/s^2.
        """
        r = state_vec[:3]
        v = state_vec[3:]
        r_mag = float(np.linalg.norm(r))
        if r_mag < 1e-6:
            return np.zeros(3, dtype=np.float64)

        altitude_km = r_mag - self.R_body
        if altitude_km > self.cutoff_km:
            return np.zeros(3, dtype=np.float64)

        # rho0 in ATMOSPHERE_CONSTANTS is defined in kg/m^3 divided by 1e9 (e.g., 1.225e-9).
        # We multiply by 1e9 to obtain the physical density in kg/m^3.
        rho_kg_m3 = (self.rho0 * 1e9) * np.exp(-altitude_km / self.H)
        v_mag = float(np.linalg.norm(v))

        # Perform the drag calculation in standard SI units (m and m/s)
        # to avoid unit conversion errors.
        v_m_s = v * 1e3
        v_mag_m_s = v_mag * 1e3

        a_drag_m_s2 = -0.5 * self.Cd * (self.area_m2 / self.mass_kg) * rho_kg_m3 * v_mag_m_s * v_m_s

        # Convert the result from m/s^2 to km/s^2.
        return np.asarray(a_drag_m_s2 * 1e-3, dtype=np.float64)


# Atmospheric constants per body
# rho0 values are physical sea level densities in kg/m^3 scaled by 1e-9
# (Earth: 1.225e-9, Mars: 2.0e-11)
ATMOSPHERE_CONSTANTS: dict[str, dict[str, float]] = {
    "EARTH": {"rho0": 1.225e-9, "H": 8.5, "cutoff_km": 1000.0},
    "MARS": {"rho0": 2.0e-11, "H": 11.1, "cutoff_km": 200.0},
}
