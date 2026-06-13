import pytest
from pathlib import Path

SPICE = (Path("data/spice_kernels") / "de440.bsp").exists()

@pytest.mark.skipif(not SPICE, reason="SPICE kernels required")
@pytest.mark.slow
def test_venus_mars_flyby_finds_feasible():
    from astra.physics.kernel import PhysicsKernel
    from astra.dsl.parser import parse_mission_file
    from astra.dsl.compiler import compile_mission
    from astra.optimization.engine import optimize_mission_with_flyby
    
    kernel = PhysicsKernel().load()
    dsl = parse_mission_file("data/benchmarks/earth_venus_mars_2032.yaml")
    mission = compile_mission(dsl, kernel.ephemeris)
    result = optimize_mission_with_flyby(
        mission, kernel, n_trials=500, time_limit=90.0, seed=42)
    
    assert result.converged, "Must find at least 1 feasible Venus flyby trajectory"
    best = result.best_trajectory
    
    # Physical sanity checks
    assert best.delta_v_total < 9.0, f"Δv {best.delta_v_total:.3f} exceeds 9.0 km/s"
    assert best.delta_v_total > 2.0, f"Δv {best.delta_v_total:.3f} unphysically low"
    assert len(best.maneuvers) == 3, "Venus flyby must have 3 maneuvers: TMI, FLY_VENUS, MOI"
    assert best.maneuvers[1].label == "FLY_VENUS"
    
    meta = best.metadata
    assert "periapsis_alt_km" in meta
    assert meta["periapsis_alt_km"] >= 300.0, "Periapsis below safe Venus altitude"
    assert meta["flyby_turn_angle_deg"] > 0, "Turn angle must be positive"
    
    print(f"\nEarth-Venus-Mars flyby result:")
    print(f"  Total Delta-v: {best.delta_v_total:.4f} km/s")
    print(f"  Duration: {best.duration_days:.1f} days")
    print(f"  Venus periapsis alt: {meta['periapsis_alt_km']:.0f} km")
    print(f"  Venus turn angle: {meta['flyby_turn_angle_deg']:.2f} deg")
    print(f"  Leg 1 (Earth->Venus): {meta['tof_leg1_days']:.0f} days")
    print(f"  Leg 2 (Venus->Mars): {meta['tof_leg2_days']:.0f} days")
