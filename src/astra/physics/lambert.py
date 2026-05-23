"""Lambert problem solver using the Izzo (2015) algorithm.
Given two position vectors and time of flight, returns the
initial and final velocity vectors for the connecting conic arc.
Reference: Izzo, D. (2015). Revisiting Lambert's problem.
Celestial Mechanics and Dynamical Astronomy, 121(1), 1-15."""
from __future__ import annotations

import math

import numpy as np


def lambert_izzo(
    r1: np.ndarray,
    r2: np.ndarray,
    tof: float,
    mu: float,
    retrograde: bool = False,
    max_iter: int = 50,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Solve Lambert's problem via Izzo universal variable method.

    Parameters
    ----------
    r1      : initial position vector [km]
    r2      : final position vector [km]
    tof     : time of flight [seconds]
    mu      : gravitational parameter [km³/s²]
    retrograde : if True, use retrograde transfer
    max_iter: maximum Householder iterations
    tol     : convergence tolerance

    Returns
    -------
    v1      : initial velocity [km/s]
    v2      : final velocity [km/s]
    converged: True if solution converged within tolerance
    """
    r1_norm = float(np.linalg.norm(r1))
    r2_norm = float(np.linalg.norm(r2))

    # Check collinearity of r1 and r2
    cross_r = np.cross(r1, r2)
    cross_norm = float(np.linalg.norm(cross_r))
    if cross_norm < 1e-10:
        return np.zeros(3), np.zeros(3), False

    c = r2 - r1
    c_norm = float(np.linalg.norm(c))

    # Semiperimeter
    s = (r1_norm + r2_norm + c_norm) * 0.5

    # Versors
    i_r1 = r1 / r1_norm
    i_r2 = r2 / r2_norm
    i_h = cross_r / cross_norm

    # Geometry parameter lambda
    ll = math.sqrt(1.0 - min(1.0, c_norm / s))

    # Tangential directions
    if i_h[2] < 0:
        ll = -ll
        i_t1 = np.cross(i_r1, i_h)
        i_t2 = np.cross(i_r2, i_h)
    else:
        i_t1 = np.cross(i_h, i_r1)
        i_t2 = np.cross(i_h, i_r2)

    # Adjust for retrograde
    if retrograde:
        ll = -ll
        i_t1 = -i_t1
        i_t2 = -i_t2

    # Non-dimensional time of flight
    T = math.sqrt(2.0 * mu / s**3) * tof

    # Initial guess
    T_0 = math.acos(ll) + ll * math.sqrt(1.0 - ll**2)
    T_1 = 2.0 * (1.0 - ll**3) / 3.0

    if T >= T_0:
        x0 = (T_0 / T) ** (2.0 / 3.0) - 1.0
    elif T < T_1:
        x0 = 2.5 * T_1 / T * (T_1 - T) / (1.0 - ll**5) + 1.0
    else:
        x0 = math.exp(math.log(2.0) * math.log(T / T_0) / math.log(T_1 / T_0)) - 1.0

    # Householder iterations
    x = x0
    converged = False

    for _ in range(max_iter):
        y = math.sqrt(1.0 - ll**2 * (1.0 - x**2))

        # compute psi
        if -1.0 <= x < 1.0:
            arg = x * y + ll * (1.0 - x**2)
            arg = max(-1.0, min(1.0, arg))
            psi = math.acos(arg)
        elif x > 1.0:
            arg = (y - x * ll) * math.sqrt(x**2 - 1.0)
            psi = math.asinh(arg)
        else:
            psi = 0.0

        denom = 1.0 - x**2
        if abs(denom) < 1e-12:
            break

        if x < 1.0:
            T_ = (psi / math.sqrt(1.0 - x**2) - x + ll * y) / denom
        else:
            T_ = (psi / math.sqrt(x**2 - 1.0) - x + ll * y) / denom

        fval = T_ - T

        # derivatives
        fder = (3.0 * T_ * x - 2.0 + 2.0 * ll**3 * x / y) / denom
        fder2 = (3.0 * T_ + 5.0 * x * fder + 2.0 * (1.0 - ll**2) * ll**3 / y**3) / denom
        fder3 = (7.0 * x * fder2 + 8.0 * fder - 6.0 * (1.0 - ll**2) * ll**5 * x / y**5) / denom

        # Householder step
        num = fder**2 - fval * fder2 / 2.0
        den = fder * (fder**2 - fval * fder2) + fder3 * fval**2 / 6.0
        if abs(den) < 1e-15:
            break
        dx = fval * (num / den)

        x_new = x - dx
        if x_new < -0.999:
            x_new = -0.999
        x = x_new

        if abs(dx) < tol:
            converged = True
            break

    # Final y computation
    y = math.sqrt(1.0 - ll**2 * (1.0 - x**2))

    # Reconstruct velocities
    gamma = math.sqrt(mu * s / 2.0)
    rho = (r1_norm - r2_norm) / c_norm
    sigma = math.sqrt(1.0 - rho**2)

    V_r1 = gamma * ((ll * y - x) - rho * (ll * y + x)) / r1_norm
    V_r2 = -gamma * ((ll * y - x) + rho * (ll * y + x)) / r2_norm
    V_t1 = gamma * sigma * (y + ll * x) / r1_norm
    V_t2 = gamma * sigma * (y + ll * x) / r2_norm

    v1 = V_r1 * i_r1 + V_t1 * i_t1
    v2 = V_r2 * i_r2 + V_t2 * i_t2

    return v1, v2, converged
