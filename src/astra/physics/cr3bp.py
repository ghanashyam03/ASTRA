"""Circular Restricted Three-Body Problem (CR3BP) physical propagator interface stubs.
Part of the ASTRA physics core.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CR3BPSystem:
    """Pre-computed constants and scale factors for a CR3BP Sun-planet system."""

    body: str  # planet name (e.g. "VENUS")
    mu_star: float  # mass parameter (dimensionless)
    L_star_km: float  # length unit [km]
    n_rad_per_s: float  # mean motion [rad/s]
    T_star_s: float  # time unit [s]


# CR3BP system constants populated from astronomical data (DE440 mass parameters)
CR3BP_SYSTEMS: dict[str, CR3BPSystem] = {
    "MERCURY": CR3BPSystem(
        body="MERCURY",
        mu_star=1.6601e-7,
        L_star_km=57909335.748,
        n_rad_per_s=8.2667152e-07,
        T_star_s=1209670.311,
    ),
    "VENUS": CR3BPSystem(
        body="VENUS",
        mu_star=2.4478e-6,
        L_star_km=108208627.813,
        n_rad_per_s=3.2364098e-07,
        T_star_s=3089843.522,
    ),
    "EARTH": CR3BPSystem(
        body="EARTH",
        mu_star=3.0034e-6,
        L_star_km=149597870.700,
        n_rad_per_s=1.9909867e-07,
        T_star_s=5022635.349,
    ),
    "MARS": CR3BPSystem(
        body="MARS",
        mu_star=3.2271e-7,
        L_star_km=227936291.671,
        n_rad_per_s=1.0586092e-07,
        T_star_s=9446356.476,
    ),
    "JUPITER": CR3BPSystem(
        body="JUPITER",
        mu_star=9.5370e-4,
        L_star_km=778411576.486,
        n_rad_per_s=1.6782207e-08,
        T_star_s=59586918.054,
    ),
    "SATURN": CR3BPSystem(
        body="SATURN",
        mu_star=2.8578e-4,
        L_star_km=1426725364.717,
        n_rad_per_s=6.7609434e-09,
        T_star_s=147908353.100,
    ),
    "URANUS": CR3BPSystem(
        body="URANUS",
        mu_star=4.3667e-5,
        L_star_km=2870977615.965,
        n_rad_per_s=2.3682121e-09,
        T_star_s=422259484.022,
    ),
    "NEPTUNE": CR3BPSystem(
        body="NEPTUNE",
        mu_star=5.1503e-5,
        L_star_km=4498258374.078,
        n_rad_per_s=1.2075369e-09,
        T_star_s=828132050.006,
    ),
}


def cr3bp_eom(t: float, q: np.ndarray, mu: float) -> np.ndarray:
    """CR3BP equations of motion. Returns dq/dt.

    Parameters
    ----------
    t : float
        Dimensionless time.
    q : np.ndarray
        Dimensionless state vector [x, y, z, vx, vy, vz].
    mu : float
        Dimensionless mass parameter.

    Returns
    -------
    np.ndarray
        Derivative state vector dq/dt.
    """
    raise NotImplementedError("Implemented in P53")


def propagate_cr3bp(
    system: CR3BPSystem,
    q0: np.ndarray,  # initial state in rotating frame (non-dimensional)
    t_span: tuple[float, float],  # integration time span (non-dimensional)
    t_eval: np.ndarray | None = None,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate CR3BP EOM. Returns (t_array, q_array)."""
    raise NotImplementedError("Implemented in P53")


def nondimensionalize_state(
    pos_km: np.ndarray,
    vel_km_s: np.ndarray,
    epoch_s: float,
    system: CR3BPSystem,
    planet_pos_km: np.ndarray,
) -> np.ndarray:
    """Convert inertial ECLIPJ2000 state to rotating CR3BP non-dimensional state."""
    raise NotImplementedError("Implemented in P53")


def dimensionalize_state(
    q_nd: np.ndarray, epoch_s: float, system: CR3BPSystem, planet_pos_km: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert rotating CR3BP state back to inertial km/km/s."""
    raise NotImplementedError("Implemented in P53")
