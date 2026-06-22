from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np


class ReferenceFrame(StrEnum):
    ICRF = "ICRF"
    J2000 = "J2000"
    ECLIPJ2000 = "ECLIPJ2000"
    ITRF93 = "ITRF93"


class CelestialBody(StrEnum):
    SUN = "SUN"
    MERCURY = "MERCURY"
    VENUS = "VENUS"
    EARTH = "EARTH"
    MOON = "MOON"
    MARS = "MARS"
    JUPITER = "JUPITER"
    SATURN = "SATURN"
    URANUS = "URANUS"
    NEPTUNE = "NEPTUNE"
    PLUTO = "PLUTO"


# Standard gravitational parameters μ = GM [km³/s²]
GM: dict[str, float] = {
    "SUN": 1.32712440018e11,
    "EARTH": 3.986004418e5,
    "MOON": 4.9048695e3,
    "MARS": 4.282837e4,
    "VENUS": 3.24859e5,
    "MERCURY": 2.2032e4,
    "JUPITER": 1.26686534e8,
    "SATURN": 3.7931187e7,
    "URANUS": 5.793939e6,
    "NEPTUNE": 6.836529e6,
}

# Equatorial physical radii [km] for collision detection
PHYSICAL_RADIUS: dict[CelestialBody, float] = {
    CelestialBody.SUN: 696340.0,
    CelestialBody.MERCURY: 2439.7,
    CelestialBody.VENUS: 6051.8,
    CelestialBody.EARTH: 6378.137,
    CelestialBody.MOON: 1737.4,
    CelestialBody.MARS: 3389.5,
    CelestialBody.JUPITER: 71492.0,
    CelestialBody.SATURN: 60268.0,
    CelestialBody.URANUS: 25559.0,
    CelestialBody.NEPTUNE: 24764.0,
    CelestialBody.PLUTO: 1188.3,
}


@dataclass
class OrbitalState:
    """Cartesian orbital state in specified reference frame."""

    epoch: float  # J2000 seconds
    position: np.ndarray  # [x, y, z] km
    velocity: np.ndarray  # [vx, vy, vz] km/s
    frame: ReferenceFrame = ReferenceFrame.ICRF
    central_body: CelestialBody = CelestialBody.SUN
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)
        assert self.position.dtype == np.float64, "position must be float64"
        assert self.velocity.dtype == np.float64, "velocity must be float64"
        assert self.position.shape == (3,), "position must be shape (3,)"
        assert self.velocity.shape == (3,), "velocity must be shape (3,)"

    @property
    def mu(self) -> float:
        return GM[self.central_body.value]

    @property
    def r(self) -> float:
        """Scalar distance from central body [km]."""
        return float(np.linalg.norm(self.position))

    @property
    def v(self) -> float:
        """Scalar speed [km/s]."""
        return float(np.linalg.norm(self.velocity))

    @property
    def specific_energy(self) -> float:
        """Vis-viva specific orbital energy [km²/s²]."""
        return 0.5 * self.v**2 - self.mu / self.r

    @property
    def specific_angular_momentum(self) -> np.ndarray:
        """h = r × v [km²/s]."""
        return np.cross(self.position, self.velocity)

    @property
    def semi_major_axis(self) -> float:
        """a = -μ / (2ε) [km]."""
        return -self.mu / (2.0 * self.specific_energy)

    @property
    def eccentricity_vector(self) -> np.ndarray:
        h = self.specific_angular_momentum
        return np.cross(self.velocity, h) / self.mu - self.position / self.r

    @property
    def eccentricity(self) -> float:
        return float(np.linalg.norm(self.eccentricity_vector))

    def to_dict(self) -> dict[str, Any]:
        res = {
            "epoch_j2000": self.epoch,
            "position_km": self.position.tolist(),
            "velocity_km_s": self.velocity.tolist(),
            "frame": self.frame.value,
            "central_body": self.central_body.value,
            "r_km": self.r,
            "v_km_s": self.v,
            "sma_km": self.semi_major_axis,
            "eccentricity": self.eccentricity,
        }
        if self.metadata:
            res["metadata"] = self.metadata
        return res
