"""Multi-body benchmark coverage for Earth-Venus-Mars flyby planning."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from astra.dsl.compiler import compile_mission
from astra.dsl.parser import parse_mission_file
from astra.dsl.schema import ConstraintType, PhysicsModelType
from astra.physics.flyby import compute_flyby
from astra.physics.kernel import PhysicsKernel
from astra.state.orbital_state import PHYSICAL_RADIUS, CelestialBody

SPICE = (Path("data/spice_kernels") / "de440.bsp").exists()


@pytest.mark.skipif(not SPICE, reason="SPICE kernels required")
@pytest.mark.slow
def test_earth_venus_mars_2032_flyby_benchmark() -> None:
    """Earth-Venus-Mars 2032 validates multi-body DSL and Venus flyby physics."""
    kernel = PhysicsKernel().load()
    dsl = parse_mission_file("data/benchmarks/earth_venus_mars_2032.yaml")
    mission = compile_mission(dsl, kernel.ephemeris)

    assert mission.origin_body == CelestialBody.EARTH
    assert mission.destination_body == CelestialBody.MARS
    assert PhysicsModelType.N_BODY in dsl.physics.models
    assert any(
        c.type == ConstraintType.MIN_PERIAPSIS and c.body == "Venus" for c in dsl.constraints
    )

    venus_radius = PHYSICAL_RADIUS[CelestialBody.VENUS]
    min_periapsis_km = venus_radius + 300.0
    target_turn_rad = math.radians(30.0)
    v_inf_in = np.array([9.7, 0.0, 0.0])

    def turn_angle_error(periapsis_km: float) -> float:
        result = compute_flyby(v_inf_in, periapsis_km, "VENUS")
        return abs(math.radians(result.turn_angle_deg) - target_turn_rad)

    opt = minimize_scalar(
        turn_angle_error,
        bounds=(min_periapsis_km, venus_radius + 50_000.0),
        method="bounded",
        options={"xatol": 1e-6},
    )
    assert opt.success

    flyby = compute_flyby(v_inf_in, float(opt.x), "VENUS")
    assert flyby.is_valid
    assert flyby.periapsis_altitude_km > 300.0
    assert abs(flyby.turn_angle_deg - 30.0) < 1e-3
    assert flyby.dv_helio_km_s < 9.0
