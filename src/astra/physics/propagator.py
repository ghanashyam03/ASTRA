"""Two-body and n-body numerical orbital propagator."""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from astra.state.orbital_state import OrbitalState


def two_body_ode(
    t: float, y: np.ndarray, mu: float
) -> np.ndarray:
    """Two-body equations of motion: ÿ = -μ/|r|³ × r."""
    r = y[:3]
    r_norm = np.linalg.norm(r)
    r_norm3 = r_norm**3
    ax, ay, az = -mu * r / r_norm3
    return np.array([y[3], y[4], y[5], ax, ay, az])

def propagate_two_body(
    state: OrbitalState,
    dt_seconds: float,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> OrbitalState:
    """Propagate state forward by dt_seconds using RK45 integrator.

    Returns the new OrbitalState. Raises RuntimeError if integration fails.
    """
    mu = state.mu
    y0 = np.concatenate([state.position, state.velocity])
    sol = solve_ivp(
        two_body_ode,
        t_span=(0.0, dt_seconds),
        y0=y0,
        args=(mu,),
        method="RK45",
        rtol=rtol,
        atol=atol,
        dense_output=False,
    )
    if not sol.success:
        raise RuntimeError(f"Propagation failed: {sol.message}")
    y_final = sol.y[:, -1]
    return OrbitalState(
        epoch=state.epoch + dt_seconds,
        position=y_final[:3],
        velocity=y_final[3:],
        frame=state.frame,
        central_body=state.central_body,
    )

def propagate_to_times(
    state: OrbitalState,
    times_seconds: np.ndarray,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> list[OrbitalState]:
    """Propagate to multiple time points. Returns list of OrbitalStates."""
    mu = state.mu
    y0 = np.concatenate([state.position, state.velocity])
    sol = solve_ivp(
        two_body_ode,
        t_span=(0.0, float(times_seconds[-1])),
        y0=y0,
        args=(mu,),
        method="RK45",
        t_eval=times_seconds,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"Propagation failed: {sol.message}")
    states = []
    for i, t in enumerate(times_seconds):
        states.append(OrbitalState(
            epoch=state.epoch + t,
            position=sol.y[:3, i],
            velocity=sol.y[3:, i],
            frame=state.frame,
            central_body=state.central_body,
        ))
    return states
