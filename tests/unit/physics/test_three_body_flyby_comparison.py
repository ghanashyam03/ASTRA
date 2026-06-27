import math
import typing
from pathlib import Path

import numpy as np
import pytest

SPICE = (Path("data/spice_kernels") / "de440.bsp").exists()


@pytest.mark.skipif(not SPICE, reason="SPICE kernels required")
@pytest.mark.parametrize(
    "body,v_inf,periapsis_alt",
    [
        ("MERCURY", 5.0, 500.0),  # P44 flagged this as highest-ratio (~19)
        ("MARS", 5.5, 400.0),  # P44 flagged this as high-ratio (~8.8)
        ("NEPTUNE", 10.0, 50000.0),  # P44 flagged this as lowest-ratio (~1.1)
    ],
)
def test_planet_frozen_vs_moving_position_deviation(
    body: str, v_inf: float, periapsis_alt: float
) -> None:
    from astra.physics.forces.gravity import ForceModel, PointMassGravity
    from astra.physics.forces.third_body import EphemerisThirdBodyPerturbation
    from astra.physics.kernel import PhysicsKernel
    from astra.physics.propagator import propagate_two_body
    from astra.physics.soi import compute_soi_radius
    from astra.state.orbital_state import (
        GM,
        PHYSICAL_RADIUS,
        CelestialBody,
        OrbitalState,
        ReferenceFrame,
    )

    kernel = PhysicsKernel().load()
    cb = CelestialBody[body]
    mu_body = GM[body]
    mu_sun = GM["SUN"]
    R_body = PHYSICAL_RADIUS[cb]
    r_p = R_body + periapsis_alt
    t_peri = kernel.epoch_from_date("2030-06-01T00:00:00")

    v_peri = math.sqrt(v_inf**2 + 2.0 * mu_body / r_p)
    planet_state_at_peri = kernel.get_body_state(cb, t_peri)
    r_helio = planet_state_at_peri.position + np.array([r_p, 0.0, 0.0])
    v_helio = planet_state_at_peri.velocity + np.array([0.0, v_peri, 0.0])
    state0 = OrbitalState(
        epoch=t_peri,
        position=r_helio,
        velocity=v_helio,
        frame=ReferenceFrame.ECLIPJ2000,
        central_body=CelestialBody.SUN,
    )

    r_soi = compute_soi_radius(body)
    dt_half = r_soi / v_inf

    # "Frozen planet" propagation: planet's gravity as a FIXED point mass
    # at its t_peri position (literal patched-conics assumption)
    frozen_planet_pos = planet_state_at_peri.position.copy()

    class _FrozenThirdBody(ForceModel):
        def acceleration(self, state_vec: np.ndarray, t: float) -> np.ndarray:
            r_sc = state_vec[:3]
            r_rel = r_sc - frozen_planet_pos
            d = float(np.linalg.norm(r_rel))
            if d < 1e-6:
                return np.zeros(3)
            accel = -mu_body * r_rel / (d**3)
            return typing.cast(np.ndarray, accel)

    forces_frozen = [PointMassGravity(mu_sun), _FrozenThirdBody()]
    state_frozen_exit = propagate_two_body(state0, dt_half, forces=forces_frozen)

    # "Moving planet" propagation: real, time-varying ephemeris position
    forces_moving = [
        PointMassGravity(mu_sun),
        EphemerisThirdBodyPerturbation(kernel, body, t_peri),
    ]
    state_moving_exit = propagate_two_body(state0, dt_half, forces=forces_moving)

    pos_deviation_km = float(
        np.linalg.norm(state_frozen_exit.position - state_moving_exit.position)
    )
    vel_deviation_km_s = float(
        np.linalg.norm(state_frozen_exit.velocity - state_moving_exit.velocity)
    )

    print(
        f"\n{body} (v_inf={v_inf} km/s, periapsis alt={periapsis_alt} km): "
        f"position deviation={pos_deviation_km:.1f} km, "
        f"velocity deviation={vel_deviation_km_s:.6f} km/s, "
        f"over a {dt_half / 86400.0:.1f}-day SOI half-crossing"
    )
    assert pos_deviation_km >= 0.0  # always true, but ensures no NaN/inf
    assert np.isfinite(vel_deviation_km_s)


@pytest.mark.skipif(not SPICE, reason="SPICE kernels required")
def test_scientific_verifications() -> None:
    from astra.physics.forces.gravity import ForceModel, PointMassGravity
    from astra.physics.forces.third_body import EphemerisThirdBodyPerturbation
    from astra.physics.kernel import PhysicsKernel
    from astra.physics.propagator import propagate_two_body
    from astra.physics.soi import compute_soi_radius
    from astra.state.orbital_state import (
        GM,
        PHYSICAL_RADIUS,
        CelestialBody,
        OrbitalState,
        ReferenceFrame,
    )

    kernel = PhysicsKernel().load()
    body = "MARS"
    cb = CelestialBody[body]
    mu_body = GM[body]
    mu_sun = GM["SUN"]
    R_body = PHYSICAL_RADIUS[cb]
    r_p = R_body + 400.0
    t_peri = kernel.epoch_from_date("2030-06-01T00:00:00")
    v_inf = 5.5

    v_peri = math.sqrt(v_inf**2 + 2.0 * mu_body / r_p)
    planet_state_at_peri = kernel.get_body_state(cb, t_peri)
    r_helio = planet_state_at_peri.position + np.array([r_p, 0.0, 0.0])
    v_helio = planet_state_at_peri.velocity + np.array([0.0, v_peri, 0.0])

    # 1. Common Initial State Check
    state0_frozen = OrbitalState(
        epoch=t_peri,
        position=r_helio.copy(),
        velocity=v_helio.copy(),
        frame=ReferenceFrame.ECLIPJ2000,
        central_body=CelestialBody.SUN,
    )
    state0_moving = OrbitalState(
        epoch=t_peri,
        position=r_helio.copy(),
        velocity=v_helio.copy(),
        frame=ReferenceFrame.ECLIPJ2000,
        central_body=CelestialBody.SUN,
    )

    assert np.all(state0_frozen.position == state0_moving.position)
    assert np.all(state0_frozen.velocity == state0_moving.velocity)
    assert state0_frozen.epoch == state0_moving.epoch

    r_soi = compute_soi_radius(body)
    dt_half = r_soi / v_inf

    # Setup standard frozen force
    frozen_planet_pos = planet_state_at_peri.position.copy()

    class _FrozenThirdBody(ForceModel):
        def acceleration(self, state_vec: np.ndarray, t: float) -> np.ndarray:
            r_sc = state_vec[:3]
            r_rel = r_sc - frozen_planet_pos
            d = float(np.linalg.norm(r_rel))
            if d < 1e-6:
                return np.zeros(3)
            accel = -mu_body * r_rel / (d**3)
            return typing.cast(np.ndarray, accel)

    # 2. Convergence Check
    forces_frozen_def = [PointMassGravity(mu_sun), _FrozenThirdBody()]
    state_frozen_def = propagate_two_body(
        state0_frozen, dt_half, forces=forces_frozen_def, rtol=1e-10, atol=1e-12
    )
    forces_moving_def = [
        PointMassGravity(mu_sun),
        EphemerisThirdBodyPerturbation(kernel, body, t_peri),
    ]
    state_moving_def = propagate_two_body(
        state0_moving, dt_half, forces=forces_moving_def, rtol=1e-10, atol=1e-12
    )
    dev_def = float(np.linalg.norm(state_frozen_def.position - state_moving_def.position))

    # Tighter tolerances
    state_frozen_tight = propagate_two_body(
        state0_frozen, dt_half, forces=forces_frozen_def, rtol=1e-13, atol=1e-15
    )
    state_moving_tight = propagate_two_body(
        state0_moving, dt_half, forces=forces_moving_def, rtol=1e-13, atol=1e-15
    )
    dev_tight = float(np.linalg.norm(state_frozen_tight.position - state_moving_tight.position))

    # Verify deviation change is negligible (abs diff < 5.0 km or rel diff < 1e-4)
    assert abs(dev_def - dev_tight) < 5.0 or (abs(dev_def - dev_tight) / dev_def) < 1e-4

    # 3. Zero-Motion Control Case
    class MockStaticKernel:
        def __init__(self, static_state: OrbitalState) -> None:
            self.static_state = static_state

        def get_body_state(self, body: CelestialBody, epoch: float) -> OrbitalState:
            return self.static_state

    mock_kernel = MockStaticKernel(planet_state_at_peri)
    forces_moving_mock = [
        PointMassGravity(mu_sun),
        EphemerisThirdBodyPerturbation(mock_kernel, body, t_peri),  # type: ignore[arg-type]
    ]
    state_moving_mock = propagate_two_body(
        state0_moving, dt_half, forces=forces_moving_mock, rtol=1e-10, atol=1e-12
    )

    pos_diff = float(np.linalg.norm(state_frozen_def.position - state_moving_mock.position))
    vel_diff = float(np.linalg.norm(state_frozen_def.velocity - state_moving_mock.velocity))
    assert pos_diff < 1e-9, f"Zero-motion control position difference: {pos_diff} km"
    assert vel_diff < 1e-11, f"Zero-motion control velocity difference: {vel_diff} km/s"
