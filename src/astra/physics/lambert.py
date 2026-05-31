"""Lambert problem solver using the Izzo (2015) algorithm.
Given two position vectors and time of flight, returns the
initial and final velocity vectors for the connecting conic arc.
Reference: Izzo, D. (2015). Revisiting Lambert's problem.
Celestial Mechanics and Dynamical Astronomy, 121(1), 1-15."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from astra.physics.exceptions import LambertConvergenceError, LambertSingularityError


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
    v1      : initial velocity [km/s] (float64)
    v2      : final velocity [km/s] (float64)
    converged: True if solution converged within tolerance

    Raises
    ------
    LambertSingularityError : if transfer geometry is singular or time of flight is invalid.
    LambertConvergenceError : if the solver fails to converge.
    """
    if tof <= 0.0:
        raise LambertSingularityError(f"Time of flight must be strictly positive. Got: {tof}")

    r1_norm = float(np.linalg.norm(r1))
    r2_norm = float(np.linalg.norm(r2))

    # Check collinearity of r1 and r2
    cross_r = np.cross(r1, r2)
    cross_norm = float(np.linalg.norm(cross_r))
    if cross_norm < 1e-10:
        raise LambertSingularityError(
            f"Collinear transfer geometry (cross norm = {cross_norm:.2e} < 1e-10). "
            f"Lambert solver cannot uniquely resolve orbital plane."
        )

    c = r2 - r1
    c_norm = float(np.linalg.norm(c))

    # Semiperimeter
    s = (r1_norm + r2_norm + c_norm) * 0.5
    if s <= 0.0:
        raise LambertSingularityError(f"Invalid semiperimeter: {s} <= 0")

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
    dx = 0.0

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

    if not converged:
        raise LambertConvergenceError(
            f"Lambert solver failed to converge within {max_iter} iterations "
            f"with tolerance {tol}. Last step size: {abs(dx):.2e}"
        )

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

    v1 = np.asarray(V_r1 * i_r1 + V_t1 * i_t1, dtype=np.float64)
    v2 = np.asarray(V_r2 * i_r2 + V_t2 * i_t2, dtype=np.float64)

    assert v1.dtype == np.float64, "v1 must be float64"
    assert v2.dtype == np.float64, "v2 must be float64"

    return v1, v2, converged


@dataclass
class LambertSolution:
    """Represents the optimal transfer trajectory solution."""
    v1: np.ndarray        # Transfer departure velocity [km/s] (float64)
    v2: np.ndarray        # Transfer arrival velocity [km/s] (float64)
    n_revs: int           # Number of revolutions
    branch: str           # "low", "high", or "single"
    delta_v: float        # Total Δv in km/s


def _compute_t_and_derivatives(x: float, ll: float, n: int) -> tuple[float, float, float, float]:
    """Compute non-dimensional time of flight and its first three derivatives w.r.t x."""
    y = math.sqrt(1.0 - ll**2 * (1.0 - x**2))
    arg = x * y + ll * (1.0 - x**2)
    arg = max(-1.0, min(1.0, arg))
    psi = math.acos(arg) + n * math.pi
    
    denom = 1.0 - x**2
    if abs(denom) < 1e-12:
        denom = 1e-12 if denom >= 0 else -1e-12
        
    T_ = (psi / math.sqrt(1.0 - x**2) - x + ll * y) / denom
    
    fder = (3.0 * T_ * x - 2.0 + 2.0 * ll**3 * x / y) / denom
    fder2 = (3.0 * T_ + 5.0 * x * fder + 2.0 * (1.0 - ll**2) * ll**3 / y**3) / denom
    fder3 = (7.0 * x * fder2 + 8.0 * fder - 6.0 * (1.0 - ll**2) * ll**5 * x / y**5) / denom
    
    return T_, fder, fder2, fder3


def lambert_min_tof_multirev(
    ll: float,
    n: int,
    max_iter: int = 15,
    tol: float = 1e-12,
) -> tuple[float, float]:
    """Compute x_min and non-dimensional T_min for N-revolution transfer.
    
    Uses Halley iterations on fder = 0 to find the turning point.
    """
    x = -0.5  # Robust initial guess for x_min
    for _ in range(max_iter):
        T_, fder, fder2, fder3 = _compute_t_and_derivatives(x, ll, n)
        num = 2.0 * fder2**2 - fder * fder3
        if abs(num) < 1e-15:
            break
        dx = 2.0 * fder * fder2 / num
        x_new = x - dx
        if x_new <= -0.999:
            x_new = -0.999
        elif x_new >= 0.999:
            x_new = 0.999
        
        x = x_new
        if abs(dx) < tol:
            break
            
    T_min, _, _, _ = _compute_t_and_derivatives(x, ll, n)
    return x, T_min


def lambert_izzo_multirev(
    r1: np.ndarray,
    r2: np.ndarray,
    tof: float,
    mu: float,
    n: int,
    branch: str,
    retrograde: bool = False,
    max_iter: int = 50,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Solve the multi-revolution Lambert problem via Izzo's universal variable method.
    
    Raises
    ------
    LambertSingularityError : if transfer geometry is singular or T < T_min.
    LambertConvergenceError : if the solver fails to converge.
    """
    if tof <= 0.0:
        raise LambertSingularityError(f"Time of flight must be strictly positive. Got: {tof}")
    if n < 1:
        raise ValueError(f"lambert_izzo_multirev requires n >= 1, got {n}")
    if branch not in ("low", "high"):
        raise ValueError(f"branch must be 'low' or 'high', got {branch}")

    r1_norm = float(np.linalg.norm(r1))
    r2_norm = float(np.linalg.norm(r2))

    # Check collinearity of r1 and r2
    cross_r = np.cross(r1, r2)
    cross_norm = float(np.linalg.norm(cross_r))
    if cross_norm < 1e-10:
        raise LambertSingularityError(
            f"Collinear transfer geometry (cross norm = {cross_norm:.2e} < 1e-10). "
            f"Lambert solver cannot uniquely resolve orbital plane."
        )

    c = r2 - r1
    c_norm = float(np.linalg.norm(c))

    # Semiperimeter
    s = (r1_norm + r2_norm + c_norm) * 0.5
    if s <= 0.0:
        raise LambertSingularityError(f"Invalid semiperimeter: {s} <= 0")

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

    # Find turning point (x_min, T_min) for singularity check
    x_min, T_min = lambert_min_tof_multirev(ll, n, tol=tol)
    if T < T_min:
        raise LambertSingularityError(
            f"No multi-revolution solution exists for N={n} with non-dimensional TOF={T:.4f}. "
            f"Minimum non-dimensional TOF is {T_min:.4f}."
        )

    # Choose initial guess x0 based on branch
    if branch == "low":
        x0 = x_min + 0.4 * (1.0 - x_min)
    else:
        x0 = x_min - 0.4 * (x_min + 1.0)

    x = x0
    converged = False
    dx = 0.0

    for _ in range(max_iter):
        y = math.sqrt(1.0 - ll**2 * (1.0 - x**2))

        # compute psi for -1 < x < 1 (always elliptic for n >= 1)
        arg = x * y + ll * (1.0 - x**2)
        arg = max(-1.0, min(1.0, arg))
        psi = math.acos(arg) + n * math.pi

        denom = 1.0 - x**2
        if abs(denom) < 1e-12:
            break

        T_ = (psi / math.sqrt(denom) - x + ll * y) / denom
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

        # Enforce branch boundaries
        if branch == "low":
            if x_new <= x_min:
                x_new = x_min + 0.1 * (x - x_min)
            if x_new >= 0.999:
                x_new = 0.999
        else:  # high branch
            if x_new >= x_min:
                x_new = x_min - 0.1 * (x_min - x)
            if x_new <= -0.999:
                x_new = -0.999

        x = x_new

        if abs(dx) < tol:
            converged = True
            break

    if not converged:
        raise LambertConvergenceError(
            f"Lambert solver failed to converge within {max_iter} iterations "
            f"with tolerance {tol} for N={n}, branch={branch}."
        )

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

    v1 = np.asarray(V_r1 * i_r1 + V_t1 * i_t1, dtype=np.float64)
    v2 = np.asarray(V_r2 * i_r2 + V_t2 * i_t2, dtype=np.float64)

    return v1, v2, converged


def find_best_transfer(
    r1: np.ndarray,
    v1_body: np.ndarray,
    r2: np.ndarray,
    v2_body: np.ndarray,
    tof: float,
    mu: float,
    max_revs: int = 2,
    retrograde: bool = False,
    max_iter: int = 50,
    tol: float = 1e-12,
) -> LambertSolution:
    """Find the optimal transfer (minimum delta-v) across N=0 and N >= 1 multi-rev branches.
    
    Parameters
    ----------
    r1       : initial position vector [km]
    v1_body  : departure planet/body velocity vector [km/s]
    r2       : final position vector [km]
    v2_body  : arrival planet/body velocity vector [km/s]
    tof      : time of flight [seconds]
    mu       : gravitational parameter [km³/s²]
    max_revs : maximum number of revolutions to search
    retrograde: prograde/retrograde toggle
    max_iter : max iterations for root-finders
    tol      : convergence tolerance
    
    Returns
    -------
    LambertSolution
        The best trajectory solution found.
        
    Raises
    ------
    LambertSingularityError : if no valid transfer can be found.
    """
    best_sol: LambertSolution | None = None

    # 1. Evaluate standard single-rev N=0 solution
    try:
        v1_0, v2_0, converged_0 = lambert_izzo(
            r1, r2, tof, mu, retrograde=retrograde, max_iter=max_iter, tol=tol
        )
        if converged_0:
            dv1 = v1_0 - v1_body
            dv2 = v2_body - v2_0
            dv_total = float(np.linalg.norm(dv1) + np.linalg.norm(dv2))
            best_sol = LambertSolution(
                v1=v1_0,
                v2=v2_0,
                n_revs=0,
                branch="single",
                delta_v=dv_total,
            )
    except Exception:
        pass

    # Calculate semiperimeter and geometry to check T_min early for multi-revolutions
    r1_norm = float(np.linalg.norm(r1))
    r2_norm = float(np.linalg.norm(r2))
    c_norm = float(np.linalg.norm(r2 - r1))
    s = (r1_norm + r2_norm + c_norm) * 0.5
    
    # Non-dimensional time of flight
    T = math.sqrt(2.0 * mu / s**3) * tof

    # 2. Iterate through multi-revolution solutions N = 1, ..., max_revs
    for n in range(1, max_revs + 1):
        # Calculate ll geometry parameter
        cross_r = np.cross(r1, r2)
        cross_norm = float(np.linalg.norm(cross_r))
        if cross_norm < 1e-10:
            continue
        ll = math.sqrt(1.0 - min(1.0, c_norm / s))
        if cross_r[2] < 0:
            ll = -ll
        if retrograde:
            ll = -ll

        # Check early if N-rev solution exists
        try:
            _, T_min = lambert_min_tof_multirev(ll, n, tol=tol)
            if T < T_min:
                break
        except Exception:
            continue

        # Evaluate both branches
        for branch in ("low", "high"):
            try:
                v1, v2, converged = lambert_izzo_multirev(
                    r1, r2, tof, mu, n, branch, retrograde, max_iter, tol
                )
                if converged:
                    dv1 = v1 - v1_body
                    dv2 = v2_body - v2
                    dv_total = float(np.linalg.norm(dv1) + np.linalg.norm(dv2))
                    if best_sol is None or dv_total < best_sol.delta_v:
                        best_sol = LambertSolution(
                            v1=v1,
                            v2=v2,
                            n_revs=n,
                            branch=branch,
                            delta_v=dv_total,
                        )
            except Exception:
                continue

    if best_sol is None:
        raise LambertSingularityError(
            f"No valid single or multi-revolution Lambert transfer could be found "
            f"for time of flight {tof:.1f}s."
        )

    return best_sol
