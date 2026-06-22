"""Sphere-of-influence computations for ASTRA patched-conics model."""

from __future__ import annotations

from dataclasses import dataclass

# Semi-major axes [km] for SOI computation (mean orbital radii)
SEMI_MAJOR_AXIS_KM: dict[str, float] = {
    "MERCURY": 5.791e7,
    "VENUS": 1.082e8,
    "EARTH": 1.496e8,
    "MARS": 2.279e8,
    "JUPITER": 7.784e8,
    "SATURN": 1.432e9,
    "URANUS": 2.867e9,
    "NEPTUNE": 4.515e9,
}

# Masses relative to Sun for SOI computation (m_body / m_sun)
MASS_RATIO: dict[str, float] = {
    "MERCURY": 1.660e-7,
    "VENUS": 2.448e-6,
    "EARTH": 3.003e-6,
    "MARS": 3.227e-7,
    "JUPITER": 9.545e-4,
    "SATURN": 2.858e-4,
    "URANUS": 4.366e-5,
    "NEPTUNE": 5.151e-5,
}


@dataclass
class SOIResult:
    body: str
    soi_radius_km: float
    is_inside: bool  # True if spacecraft is currently inside this SOI
    distance_km: float  # distance from body center


def compute_soi_radius(body: str) -> float:
    """Laplace SOI radius [km] = a_body × (m_body/m_sun)^(2/5).

    Reference: Bate, Mueller, White — Fundamentals of Astrodynamics, Ch. 6.
    """
    a = SEMI_MAJOR_AXIS_KM.get(body.upper())
    m_ratio = MASS_RATIO.get(body.upper())
    if a is None or m_ratio is None:
        raise ValueError(f"SOI data not available for body: {body}")
    return float(a * (m_ratio**0.4))


def is_in_soi(spacecraft_pos_helio: object, body_pos_helio: object, body: str) -> bool:
    """Return True if spacecraft is within body's SOI."""
    import numpy as np

    r_sc = np.asarray(spacecraft_pos_helio)
    r_body = np.asarray(body_pos_helio)
    dist = float(np.linalg.norm(r_sc - r_body))
    return dist < compute_soi_radius(body)


# Standard parking orbit altitudes [km] per body (for default mission planning)
DEFAULT_PARKING_ALTITUDE_KM: dict[str, float] = {
    "EARTH": 200.0,  # LEO
    "MARS": 300.0,  # LMO
    "VENUS": 300.0,
    "MERCURY": 200.0,
    "JUPITER": 500.0,
    "SATURN": 500.0,
}


def get_default_parking_altitude(body: str) -> float:
    return DEFAULT_PARKING_ALTITUDE_KM.get(body.upper(), 300.0)
