"""Circular Restricted Three-Body Problem (CR3BP) propagator for ASTRA.

Implements the Sun-planet CR3BP as the high-fidelity alternative to patched-conics
for inner-planet flyby trajectories. See docs/design/cr3bp_design.md for the
full mathematical specification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

# ── System definitions ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CR3BPSystem:
    """Pre-computed CR3BP parameters for a Sun-planet system."""

    body: str
    mu_star: float  # m_planet / (m_sun + m_planet)
    L_star_km: float  # semi-major axis of planet's orbit [km]
    n_rad_per_s: float  # mean orbital motion of planet [rad/s]

    @property
    def T_star_s(self) -> float:  # noqa: N802
        """Non-dimensional time unit [s]."""
        return 1.0 / self.n_rad_per_s

    @property
    def V_star_km_s(self) -> float:  # noqa: N802
        """Non-dimensional velocity unit [km/s]."""
        return self.L_star_km * self.n_rad_per_s


# Physical constants
_AU_KM = 1.495978707e8  # 1 AU in km
_G_MU_SUN_KM3S2 = 1.32712440018e11  # GM_sun [km³/s²]

# From DE440; mass ratios μ* = GM_planet / (GM_sun + GM_planet)
_GM_PLANETS_KM3S2 = {
    "MERCURY": 2.2032e4,
    "VENUS": 3.24859e5,
    "EARTH": 3.98600e5,
    "MARS": 4.28284e4,
    "JUPITER": 1.26687e8,
    "SATURN": 3.79313e7,
    "URANUS": 5.79397e6,
    "NEPTUNE": 6.83510e6,
}
_SEMI_MAJOR_AU = {
    "MERCURY": 0.38710,
    "VENUS": 0.72333,
    "EARTH": 1.00000,
    "MARS": 1.52366,
    "JUPITER": 5.20336,
    "SATURN": 9.53707,
    "URANUS": 19.1913,
    "NEPTUNE": 30.0690,
}


def _build_system(body: str) -> CR3BPSystem:
    b = body.upper()
    gm_planet = _GM_PLANETS_KM3S2[b]
    gm_total = _G_MU_SUN_KM3S2 + gm_planet
    mu_star = gm_planet / gm_total
    a_km = _SEMI_MAJOR_AU[b] * _AU_KM
    n = math.sqrt(gm_total / a_km**3)
    return CR3BPSystem(body=b, mu_star=mu_star, L_star_km=a_km, n_rad_per_s=n)


CR3BP_SYSTEMS: dict[str, CR3BPSystem] = {b: _build_system(b) for b in _GM_PLANETS_KM3S2}


# ── Equations of motion ───────────────────────────────────────────────────────


def cr3bp_eom(t: float, q: np.ndarray, mu: float) -> np.ndarray:
    """CR3BP equations of motion in the rotating frame (non-dimensional).

    State q = [x, y, z, ẋ, ẏ, ż].
    Returns dq/dt = [ẋ, ẏ, ż, ẍ, ÿ, z̈].
    """
    x, y, z, vx, vy, vz = q
    mu1 = 1.0 - mu  # mass fraction of primary (Sun)

    r1 = math.sqrt((x + mu) ** 2 + y**2 + z**2)  # distance from Sun
    r2 = math.sqrt((x - mu1) ** 2 + y**2 + z**2)  # distance from planet

    if r1 < 1e-15 or r2 < 1e-15:
        # Singularity guard: should never be reached in normal trajectories
        raise RuntimeError(
            f"CR3BP singularity: r1={r1:.2e}, r2={r2:.2e}. "
            "Spacecraft is too close to a primary body."
        )

    r1_3 = r1**3
    r2_3 = r2**3

    ax = 2.0 * vy + x - mu1 * (x + mu) / r1_3 - mu * (x - mu1) / r2_3
    ay = -2.0 * vx + y - mu1 * y / r1_3 - mu * y / r2_3
    az = -mu1 * z / r1_3 - mu * z / r2_3

    return np.array([vx, vy, vz, ax, ay, az])


def propagate_cr3bp(
    system: CR3BPSystem,
    q0_nd: np.ndarray,
    t_span_nd: tuple[float, float],
    t_eval_nd: np.ndarray | None = None,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate CR3BP EOM using scipy RK45.

    Parameters
    ----------
    system     : CR3BPSystem instance (provides mu_star)
    q0_nd      : initial state [x,y,z,vx,vy,vz] in non-dimensional rotating frame
    t_span_nd  : (t_start, t_end) in non-dimensional time
    t_eval_nd  : optional output times (non-dimensional)
    rtol, atol : integrator tolerances (defaults are tight for precision work)

    Returns
    -------
    t_nd   : (N,) time array [non-dimensional]
    q_nd   : (6, N) state array [non-dimensional]
    """
    sol = solve_ivp(
        fun=cr3bp_eom,
        t_span=t_span_nd,
        y0=q0_nd,
        method="RK45",
        t_eval=t_eval_nd,
        rtol=rtol,
        atol=atol,
        args=(system.mu_star,),
        dense_output=False,
    )
    if not sol.success:
        raise RuntimeError(f"CR3BP integration failed: {sol.message}")
    return sol.t, sol.y


def jacobi_constant(q_nd: np.ndarray, mu: float) -> float:
    """Compute the Jacobi constant C = 2Ω − v² (conserved quantity in CR3BP).

    C is conserved along any trajectory (to integrator precision). Use this
    to validate integration quality: |C(t) − C(t=0)| / C(t=0) should be < rtol.
    """
    x, y, z, vx, vy, vz = q_nd
    mu1 = 1.0 - mu
    r1 = math.sqrt((x + mu) ** 2 + y**2 + z**2)
    r2 = math.sqrt((x - mu1) ** 2 + y**2 + z**2)
    omega = 0.5 * (x**2 + y**2) + mu1 / r1 + mu / r2
    v2 = vx**2 + vy**2 + vz**2
    return float(2.0 * omega - v2)


# ── Frame transformations ─────────────────────────────────────────────────────


def nondimensionalize_state(
    pos_inertial_km: np.ndarray,
    vel_inertial_km_s: np.ndarray,
    epoch_s: float,
    system: CR3BPSystem,
    planet_pos_inertial_km: np.ndarray,
) -> np.ndarray:
    """Convert inertial ECLIPJ2000 state → CR3BP non-dimensional rotating-frame state.

    Parameters
    ----------
    pos_inertial_km        : spacecraft position in ECLIPJ2000 [km]
    vel_inertial_km_s      : spacecraft velocity in ECLIPJ2000 [km/s]
    epoch_s                : current epoch [J2000 seconds]
    system                 : CR3BPSystem for the relevant Sun-planet pair
    planet_pos_inertial_km : planet position in ECLIPJ2000 at epoch [km]
    """
    theta = system.n_rad_per_s * epoch_s  # current angle of P2 around barycentre

    # Barycentric shift: Sun is located at -mu_star * planet_pos
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    R = np.array([[cos_t, sin_t, 0.0], [-sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]])

    r_bary = pos_inertial_km - system.mu_star * planet_pos_inertial_km
    r_rot = R @ r_bary

    # Velocity: v_rot = R @ v_inertial − ω × r_rot
    omega_vec = np.array([0.0, 0.0, system.n_rad_per_s])
    v_rot = R @ vel_inertial_km_s - np.cross(omega_vec, r_rot)

    # Non-dimensionalize
    r_nd = r_rot / system.L_star_km
    v_nd = v_rot / system.V_star_km_s

    return np.array([r_nd[0], r_nd[1], r_nd[2], v_nd[0], v_nd[1], v_nd[2]])


def dimensionalize_state(
    q_nd: np.ndarray,
    epoch_s: float,
    system: CR3BPSystem,
    planet_pos_inertial_km: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert CR3BP rotating-frame non-dimensional state → inertial ECLIPJ2000 [km, km/s]."""
    theta = system.n_rad_per_s * epoch_s
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    R_inv = np.array([[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]])

    r_nd = q_nd[:3]
    v_nd = q_nd[3:]

    r_rot = r_nd * system.L_star_km
    v_rot = v_nd * system.V_star_km_s

    omega_vec = np.array([0.0, 0.0, system.n_rad_per_s])
    v_inertial = R_inv @ (v_rot + np.cross(omega_vec, r_rot))
    r_inertial = R_inv @ r_rot + system.mu_star * planet_pos_inertial_km

    return r_inertial, v_inertial


def lagrange_l1_x(mu: float) -> float:
    """L1 x-coordinate (non-dimensional) via cubic approximation."""
    alpha = (mu / 3.0) ** (1.0 / 3.0)
    return float(1.0 - alpha)


def lagrange_l2_x(mu: float) -> float:
    """L2 x-coordinate (non-dimensional) via cubic approximation."""
    alpha = (mu / 3.0) ** (1.0 / 3.0)
    return float(1.0 + alpha)
