"""Two-body and n-body numerical orbital propagator with a pluggable
integrator architecture and physics-hardened collision checks.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from astra.physics.exceptions import PropagationError
from astra.physics.forces.gravity import ForceModel
from astra.state.orbital_state import PHYSICAL_RADIUS, OrbitalState


@dataclass
class IntegrationResult:
    """Standardized results of numerical orbit integration."""
    t: np.ndarray
    y: np.ndarray
    success: bool
    message: str
    nfev: int
    njev: int
    nsteps: int


class Integrator(ABC):
    """Abstract base class for all pluggable orbit integrators (RK, symplectic, etc.)."""

    @abstractmethod
    def integrate(
        self,
        fun: Callable[..., np.ndarray],
        t_span: tuple[float, float],
        y0: np.ndarray,
        rtol: float,
        atol: float,
        t_eval: np.ndarray | None = None,
        **options: Any  # noqa: ANN401
    ) -> IntegrationResult:
        """Solve initial value problem for the given differential equations."""
        pass


class RK45Integrator(Integrator):
    """Standard Runge-Kutta 4(5) adaptive step numerical integrator."""

    def integrate(
        self,
        fun: Callable[..., np.ndarray],
        t_span: tuple[float, float],
        y0: np.ndarray,
        rtol: float,
        atol: float,
        t_eval: np.ndarray | None = None,
        **options: Any  # noqa: ANN401
    ) -> IntegrationResult:
        sol = solve_ivp(
            fun,
            t_span=t_span,
            y0=y0,
            method="RK45",
            rtol=rtol,
            atol=atol,
            t_eval=t_eval,
            dense_output=False,
            **options
        )
        nsteps = len(sol.t) if sol.success else 0
        return IntegrationResult(
            t=sol.t,
            y=sol.y,
            success=sol.success,
            message=sol.message,
            nfev=sol.nfev,
            njev=sol.njev,
            nsteps=nsteps
        )


def two_body_ode(
    t: float, y: np.ndarray, mu: float
) -> np.ndarray:
    """Two-body equations of motion: ÿ = -μ/|r|³ × r.
    Enforces np.float64 precision and checks for singularity limits.
    """
    r = y[:3]
    r_norm = float(np.linalg.norm(r))
    if r_norm < 1e-6:
        # Prevent division by zero / singularity crash in derivatives
        return np.zeros(6, dtype=np.float64)
        
    r_norm3 = r_norm**3
    ax, ay, az = -mu * r / r_norm3
    res = np.array([y[3], y[4], y[5], ax, ay, az], dtype=np.float64)
    return res


def build_ode(forces: list[ForceModel]) -> Callable[[float, np.ndarray], np.ndarray]:
    """Build a system of differential equations compatible with solve_ivp.

    Parameters
    ----------
    forces : list[ForceModel]
        A list of force model components to include in the ODE.

    Returns
    -------
    Callable[[float, np.ndarray], np.ndarray]
        The derivative function dy/dt = f(t, y) for integration.
    """
    def ode(t: float, y: np.ndarray) -> np.ndarray:
        accel = np.zeros(3, dtype=np.float64)
        for f in forces:
            accel += f.acceleration(y, t)
        return np.array([y[3], y[4], y[5], accel[0], accel[1], accel[2]], dtype=np.float64)
    return ode


def propagate_two_body(
    state: OrbitalState,
    dt_seconds: float,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    integrator: Integrator | None = None,
    forces: list[ForceModel] | None = None,
) -> OrbitalState:
    """Propagate state forward by dt_seconds using pluggable integrator.
    
    Includes safety checks for central body collisions and numerical singularities.
    Returns the new OrbitalState containing detailed integration metadata.
    """
    if integrator is None:
        integrator = RK45Integrator()

    r_min = PHYSICAL_RADIUS.get(state.central_body, 0.0)
    
    # 1. Pre-propagation collision check
    r0_norm = float(np.linalg.norm(state.position))
    if r0_norm < r_min:
        raise PropagationError(
            f"Initial state is inside the collision boundary of {state.central_body.value}: "
            f"r = {r0_norm:.3f} km, physical radius = {r_min:.3f} km"
        )

    # Setup terminal event for collision detection during integration
    def collision_event(t: float, y: np.ndarray, *args: Any) -> float:  # noqa: ANN401
        r = y[:3]
        return float(np.linalg.norm(r) - r_min)
        
    collision_event.terminal = True  # type: ignore[attr-defined]
    collision_event.direction = -1  # type: ignore[attr-defined]

    y0 = np.concatenate([state.position, state.velocity]).astype(np.float64)
    assert y0.dtype == np.float64, "Initial state vector must be np.float64"

    mu = state.mu
    start_time = time.perf_counter()
    
    # Execute integration
    if forces is None:
        res = integrator.integrate(
            two_body_ode,
            t_span=(0.0, float(dt_seconds)),
            y0=y0,
            rtol=rtol,
            atol=atol,
            args=(mu,),
            events=collision_event
        )
    else:
        res = integrator.integrate(
            build_ode(forces),
            t_span=(0.0, float(dt_seconds)),
            y0=y0,
            rtol=rtol,
            atol=atol,
            events=collision_event
        )
    
    elapsed_time = time.perf_counter() - start_time

    # 2. Check for integration success and collision triggers
    if not res.success:
        raise PropagationError(f"Numerical propagation failed: {res.message}")

    y_final = res.y[:, -1].astype(np.float64)
    r_final = float(np.linalg.norm(y_final[:3]))

    # Check if the collision event halted the integration
    if r_final < r_min:
        raise PropagationError(
            f"Collision detected with central body {state.central_body.value} "
            f"during propagation: r = {r_final:.3f} km (physical radius = {r_min:.3f} km)"
        )

    # Attach stats metadata to return state
    metadata = {
        "nfev": res.nfev,
        "njev": res.njev,
        "nsteps": res.nsteps,
        "elapsed_time": elapsed_time,
        "success": res.success,
        "integrator": integrator.__class__.__name__,
    }

    return OrbitalState(
        epoch=state.epoch + dt_seconds,
        position=y_final[:3],
        velocity=y_final[3:],
        frame=state.frame,
        central_body=state.central_body,
        metadata=metadata,
    )


def propagate_to_times(
    state: OrbitalState,
    times_seconds: np.ndarray,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    integrator: Integrator | None = None,
    forces: list[ForceModel] | None = None,
) -> list[OrbitalState]:
    """Propagate to multiple time points. Returns list of OrbitalStates."""
    if integrator is None:
        integrator = RK45Integrator()

    r_min = PHYSICAL_RADIUS.get(state.central_body, 0.0)
    
    # Pre-propagation collision check
    r0_norm = float(np.linalg.norm(state.position))
    if r0_norm < r_min:
        raise PropagationError(
            f"Initial state is inside the collision boundary of {state.central_body.value}: "
            f"r = {r0_norm:.3f} km, physical radius = {r_min:.3f} km"
        )

    def collision_event(t: float, y: np.ndarray, *args: Any) -> float:  # noqa: ANN401
        r = y[:3]
        return float(np.linalg.norm(r) - r_min)
        
    collision_event.terminal = True  # type: ignore[attr-defined]
    collision_event.direction = -1  # type: ignore[attr-defined]

    y0 = np.concatenate([state.position, state.velocity]).astype(np.float64)
    mu = state.mu
    times_eval = np.asarray(times_seconds, dtype=np.float64)

    start_time = time.perf_counter()
    if forces is None:
        res = integrator.integrate(
            two_body_ode,
            t_span=(0.0, float(times_eval[-1])),
            y0=y0,
            rtol=rtol,
            atol=atol,
            t_eval=times_eval,
            args=(mu,),
            events=collision_event
        )
    else:
        res = integrator.integrate(
            build_ode(forces),
            t_span=(0.0, float(times_eval[-1])),
            y0=y0,
            rtol=rtol,
            atol=atol,
            t_eval=times_eval,
            events=collision_event
        )
    elapsed_time = time.perf_counter() - start_time

    if not res.success:
        raise PropagationError(f"Numerical propagation failed: {res.message}")

    states = []
    # Check intermediate steps for collisions
    for i, t in enumerate(times_eval):
        y_step = res.y[:, i].astype(np.float64)
        r_step = float(np.linalg.norm(y_step[:3]))
        if r_step < r_min:
            raise PropagationError(
                f"Collision detected with central body {state.central_body.value} "
                f"at step time {t:.1f} s: r = {r_step:.3f} km (physical radius = {r_min:.3f} km)"
            )
            
        metadata = {
            "nfev": res.nfev,
            "njev": res.njev,
            "nsteps": res.nsteps,
            "elapsed_time": elapsed_time,
            "success": res.success,
            "integrator": integrator.__class__.__name__,
        }
        
        states.append(OrbitalState(
            epoch=state.epoch + t,
            position=y_step[:3],
            velocity=y_step[3:],
            frame=state.frame,
            central_body=state.central_body,
            metadata=metadata,
        ))
        
    return states
