"""End-to-end optimization integration tests. Requires SPICE kernels."""

from __future__ import annotations

from pathlib import Path

import pytest

SPICE_AVAILABLE = (Path("data/spice_kernels") / "de440.bsp").exists()


@pytest.mark.skipif(not SPICE_AVAILABLE, reason="SPICE kernels required")
def test_porkchop_grid_produces_finite_values() -> None:
    import numpy as np
    from astra.dsl.compiler import compile_mission
    from astra.dsl.parser import parse_mission_file
    from astra.optimization.engine import compute_porkchop
    from astra.physics.kernel import PhysicsKernel

    kernel = PhysicsKernel().load()
    dsl = parse_mission_file("data/benchmarks/earth_mars_2031.yaml")
    mission = compile_mission(dsl, kernel.ephemeris)
    dep_epochs, tof_days, dv_grid = compute_porkchop(mission, kernel, n_dep=20, n_tof=20)

    assert dep_epochs.shape == (20,)
    assert tof_days.shape == (20,)
    assert dv_grid.shape == (20, 20)
    finite_count = int(np.sum(np.isfinite(dv_grid)))
    assert finite_count > 50, f"Only {finite_count} finite values — porkchop mostly empty"
    min_dv = float(np.nanmin(dv_grid))
    assert 3.0 < min_dv < 12.0, f"Min porkchop Δv {min_dv:.2f} km/s is outside 3–12 range"


@pytest.mark.skipif(not SPICE_AVAILABLE, reason="SPICE kernels required")
def test_bayesian_optimization_finds_feasible() -> None:
    from astra.dsl.compiler import compile_mission
    from astra.dsl.parser import parse_mission_file
    from astra.optimization.engine import optimize_mission_bayesian
    from astra.physics.kernel import PhysicsKernel

    kernel = PhysicsKernel().load()
    dsl = parse_mission_file("data/benchmarks/earth_mars_2031.yaml")
    mission = compile_mission(dsl, kernel.ephemeris)
    # Use small budget for test speed
    mission.max_evaluations = 500
    result = optimize_mission_bayesian(mission, kernel, n_trials=500, time_limit=60.0)

    assert result.converged, "Optimizer must find at least 1 feasible trajectory"
    assert result.best_trajectory is not None
    dv = result.best_trajectory.delta_v_total
    days = result.best_trajectory.duration_days
    assert dv < 8.0, f"Best Δv {dv:.2f} km/s exceeds 8.0 km/s constraint"
    assert days < 250, f"Best duration {days:.1f} d exceeds 250-day constraint"
    # Physical plausibility check: Earth-Mars Δv always > 2.5 km/s
    assert dv > 2.5, f"Δv {dv:.2f} km/s is unphysically low"


@pytest.mark.skipif(not SPICE_AVAILABLE, reason="SPICE kernels required")
def test_pareto_front_is_nonempty() -> None:
    from astra.dsl.compiler import compile_mission
    from astra.dsl.parser import parse_mission_file
    from astra.optimization.engine import optimize_mission_bayesian
    from astra.physics.kernel import PhysicsKernel

    kernel = PhysicsKernel().load()
    dsl = parse_mission_file("data/benchmarks/earth_mars_2031.yaml")
    mission = compile_mission(dsl, kernel.ephemeris)
    result = optimize_mission_bayesian(mission, kernel, n_trials=300, time_limit=45.0)

    assert len(result.pareto_front) > 0, "Pareto front must be non-empty"
