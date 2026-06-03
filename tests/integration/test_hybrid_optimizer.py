import pytest
from pathlib import Path

from astra.dsl.compiler import compile_mission
from astra.dsl.parser import parse_mission_file
from astra.optimization.engine import (
    optimize_mission_bayesian,
    optimize_mission_hybrid,
)
from astra.physics.kernel import PhysicsKernel

SPICE = (Path("data/spice_kernels") / "de440.bsp").exists()

@pytest.mark.skipif(not SPICE, reason="SPICE kernels required")
def test_hybrid_not_worse_than_bayesian() -> None:
    """Hybrid optimizer Δv must be ≤ pure Bayesian Δv on same mission."""
    kernel = PhysicsKernel().load()
    dsl = parse_mission_file("data/benchmarks/earth_mars_2031.yaml")
    mission = compile_mission(dsl, kernel.ephemeris)

    r_bay = optimize_mission_bayesian(
        mission, kernel, n_trials=300, time_limit=45.0, seed=42)
    r_hyb = optimize_mission_hybrid(
        mission, kernel, n_trials_bayesian=300, time_limit=60.0, seed=42)

    assert r_bay.converged and r_hyb.converged
    dv_bay = r_bay.best_trajectory.delta_v_total
    dv_hyb = r_hyb.best_trajectory.delta_v_total
    print(f"\nBayesian: {dv_bay:.4f} km/s | Hybrid: {dv_hyb:.4f} km/s")
    assert dv_hyb <= dv_bay + 0.05, (
        f"Hybrid {dv_hyb:.4f} km/s is worse than Bayesian {dv_bay:.4f} by "
        f"> 0.05 km/s tolerance")

    # Assert that all required hybrid run metadata fields exist in result
    assert r_hyb.optimizer_strategy == "hybrid"
    assert r_hyb.phase1_best_dv is not None
    assert r_hyb.phase2_best_dv is not None
    assert r_hyb.refinement_improvement_km_s is not None
    assert r_hyb.refinement_evaluations is not None
    assert r_hyb.refinement_evaluations > 0

    # Assert that all required hybrid run metadata fields exist in best trajectory metadata
    meta = r_hyb.best_trajectory.metadata
    assert meta["optimizer_strategy"] == "hybrid"
    assert meta["phase1_best_dv"] == r_hyb.phase1_best_dv
    assert meta["phase2_best_dv"] == r_hyb.phase2_best_dv
    assert meta["refinement_improvement_km_s"] == r_hyb.refinement_improvement_km_s
    assert meta["refinement_evaluations"] == r_hyb.refinement_evaluations
