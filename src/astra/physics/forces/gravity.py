"""Gravity force models including point mass and J2 perturbations."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ForceModel(ABC):
    """Abstract base class for pluggable force models in the orbit propagator."""

    @abstractmethod
    def acceleration(self, state_vec: np.ndarray, t: float) -> np.ndarray:
        """Compute the acceleration vector [ax, ay, az] in km/s^2.

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
        pass


class PointMassGravity(ForceModel):
    """Standard central body point-mass gravity model."""

    def __init__(self, mu: float) -> None:
        """Initialize PointMassGravity.

        Parameters
        ----------
        mu : float
            Gravitational parameter of the central body in km^3/s^2.
        """
        self.mu = float(mu)

    def acceleration(self, state_vec: np.ndarray, t: float) -> np.ndarray:
        """Compute point-mass gravity acceleration."""
        r = state_vec[:3]
        r_mag = float(np.linalg.norm(r))
        if r_mag < 1e-6:
            return np.zeros(3, dtype=np.float64)

        return -self.mu * r / (r_mag**3)


class J2Perturbation(ForceModel):
    """J2 oblateness perturbation force model."""

    def __init__(self, mu: float, J2: float, R_body: float) -> None:  # noqa: N803
        """Initialize J2Perturbation.

        Parameters
        ----------
        mu : float
            Gravitational parameter of the central body in km^3/s^2.
        J2 : float
            J2 perturbation coefficient (dimensionless).
        R_body : float
            Equatorial radius of the central body in km.
        """
        self.mu = float(mu)
        self.J2 = float(J2)
        self.R_body = float(R_body)

    def acceleration(self, state_vec: np.ndarray, t: float) -> np.ndarray:
        """Compute J2 perturbation acceleration."""
        r = state_vec[:3]
        r_mag = float(np.linalg.norm(r))
        if r_mag < 1e-6:
            return np.zeros(3, dtype=np.float64)

        x, y, z = r[0], r[1], r[2]
        factor = 1.5 * self.J2 * self.mu * (self.R_body**2) / (r_mag**5)

        ax = factor * x * (5.0 * (z**2) / (r_mag**2) - 1.0)
        ay = factor * y * (5.0 * (z**2) / (r_mag**2) - 1.0)
        az = factor * z * (5.0 * (z**2) / (r_mag**2) - 3.0)

        return np.array([ax, ay, az], dtype=np.float64)


# J2 Perturbation constants
J2_CONSTANTS: dict[str, float] = {
    "EARTH": 1.08263e-3,
    "MARS": 1.96045e-3,
}
