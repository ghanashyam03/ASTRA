"""END-TO-END BENCHMARK: Earth-Mars 2031 full mission optimization.
This is the ASTRA acceptance test. All assertions must pass.
Requires SPICE kernels. Takes ~2–5 minutes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

SPICE_AVAILABLE = (Path("data/spice_kernels") / "de440.bsp").exists()
MISSION_FILE = Path("data/benchmarks/earth_mars_2031.yaml")


@pytest.mark.skipif(not SPICE_AVAILABLE, reason="SPICE kernels required")
@pytest.mark.slow
def test_earth_mars_full_pipeline() -> None:
    """Full pipeline: parse → compile → optimize → explain.
    Validates every layer end-to-end."""
    from astra.dsl.compiler import compile_mission
    from astra.dsl.parser import parse_mission_file
    from astra.explainability.engine import explain
    from astra.optimization.engine import optimize_mission_bayesian
    from astra.physics.kernel import PhysicsKernel

    kernel = PhysicsKernel().load()
    dsl = parse_mission_file(MISSION_FILE)
    mission = compile_mission(dsl, kernel.ephemeris)

    # Use moderate budget for CI speed
    result = optimize_mission_bayesian(mission, kernel, n_trials=1000, time_limit=90.0, seed=42)

    # ─── Core assertions ───────────────────────────────────────────────
    assert result.converged, "Must find at least one feasible trajectory"
    assert result.best_trajectory is not None
    assert result.n_feasible > 0

    best = result.best_trajectory
    dv = best.delta_v_total
    days = best.duration_days

    # Physical plausibility: Earth-Mars Δv range is 3.5–7.5 km/s
    assert 3.0 < dv < 8.0, f"Δv {dv:.3f} km/s is outside 3–8 km/s range"
    # Duration constraint (DSL says max 250 days)
    assert days < 250.0, f"Duration {days:.1f} days exceeds 250-day constraint"
    # Must have exactly 2 maneuvers (TMI + MOI) for two-impulse transfer
    assert len(best.maneuvers) == 2, "Two-impulse transfer needs exactly 2 maneuvers"
    # Maneuver labels
    assert best.maneuvers[0].label == "TMI"
    assert best.maneuvers[1].label == "MOI"
    # Each individual maneuver < total Δv
    tmi = best.maneuvers[0].magnitude
    moi = best.maneuvers[1].magnitude
    assert abs(tmi + moi - dv) < 1e-6, "Maneuver magnitudes must sum to total Δv"
    # Pareto front
    assert len(result.pareto_front) >= 2, "Pareto front must have ≥ 2 solutions"

    # ─── Explainability assertions ─────────────────────────────────────
    trace = explain(
        best,
        mission,
        pareto_front=result.pareto_front,
        ephemeris=kernel.ephemeris,
    )
    trace_dict = trace.to_dict()

    # All top-level keys present
    for key in [
        "delta_v_decomposition",
        "constraint_analysis",
        "window_rationale",
        "pareto_analysis",
    ]:
        assert key in trace_dict, f"Missing explanation key: {key}"

    # Δv decomposition
    decomp = trace_dict["delta_v_decomposition"]
    assert abs(decomp["total_km_s"] - dv) < 1e-4
    assert len(decomp["components"]) == 2
    fractions = [c["fraction_pct"] for c in decomp["components"]]
    assert abs(sum(fractions) - 100.0) < 0.1

    # Constraint analysis
    ca = trace_dict["constraint_analysis"]
    assert ca["all_satisfied"] is True

    # Window rationale
    wr = trace_dict["window_rationale"]
    assert wr is not None
    assert "departure_date_utc" in wr
    assert wr["tof_days"] > 0
    assert 0 < wr["planet_phase_angle_deg"] < 360
    assert len(wr["rationale"]) >= 3

    # Pareto analysis
    pa = trace_dict["pareto_analysis"]
    assert pa is not None
    assert pa["n_pareto_solutions"] >= 2
    assert pa["avg_tradeoff_km_s_per_day"] >= 0

    print("\n" + "=" * 60)
    print("ASTRA EARTH-MARS 2031 ACCEPTANCE TEST PASSED")
    print("=" * 60)
    print(f"  Best Δv:           {dv:.4f} km/s")
    print(f"  Duration:          {days:.1f} days")
    print(f"  TMI burn:          {tmi:.4f} km/s")
    print(f"  MOI burn:          {moi:.4f} km/s")
    print(f"  Departure:         {wr['departure_date_utc']}")
    print(f"  Arrival:           {wr['arrival_date_utc']}")
    print(f"  Pareto solutions:  {len(result.pareto_front)}")
    print(f"  Evaluations:       {result.n_evaluations}")
    print(f"  Wall time:         {result.wall_time_s:.1f}s")
    print(f"  Tradeoff:          {pa['avg_tradeoff_km_s_per_day']:.5f} km/s per day saved")
    print("=" * 60)


@pytest.mark.skipif(not SPICE_AVAILABLE, reason="SPICE kernels required")
@pytest.mark.slow
def test_neural_accelerated_matches_standard() -> None:
    """Neural-accelerated optimizer must find same quality solution as standard."""
    from astra.dsl.compiler import compile_mission
    from astra.dsl.parser import parse_mission_file
    from astra.optimization.engine import (
        optimize_mission_bayesian,
        optimize_mission_neural_accelerated,
    )
    from astra.physics.kernel import PhysicsKernel

    kernel = PhysicsKernel().load()
    dsl = parse_mission_file(MISSION_FILE)
    mission = compile_mission(dsl, kernel.ephemeris)

    r_standard = optimize_mission_bayesian(mission, kernel, n_trials=500, time_limit=60.0, seed=42)
    r_neural = optimize_mission_neural_accelerated(
        mission,
        kernel,
        n_trials=500,
        time_limit=60.0,
        seed=42,
        pretrain_samples=300,
    )

    # Both must converge
    assert r_standard.converged
    assert r_neural.converged
    assert r_standard.best_trajectory is not None
    assert r_neural.best_trajectory is not None

    # Neural result must be within 10% of standard (same quality range)
    dv_std = r_standard.best_trajectory.delta_v_total
    dv_neu = r_neural.best_trajectory.delta_v_total
    assert dv_neu < dv_std * 1.10, (
        f"Neural result {dv_neu:.3f} km/s is >10% worse than standard {dv_std:.3f} km/s"
    )
    print(f"\nStandard: {dv_std:.4f} km/s | Neural: {dv_neu:.4f} km/s")


@pytest.mark.skipif(not SPICE_AVAILABLE, reason="SPICE kernels required")
def test_determinism() -> None:
    """Same seed + same DSL = bit-identical best Δv."""
    from astra.dsl.compiler import compile_mission
    from astra.dsl.parser import parse_mission_file
    from astra.optimization.engine import optimize_mission_bayesian
    from astra.physics.kernel import PhysicsKernel

    kernel = PhysicsKernel().load()
    dsl = parse_mission_file(MISSION_FILE)
    mission = compile_mission(dsl, kernel.ephemeris)

    r1 = optimize_mission_bayesian(mission, kernel, n_trials=100, time_limit=30.0, seed=99)
    r2 = optimize_mission_bayesian(mission, kernel, n_trials=100, time_limit=30.0, seed=99)

    if r1.converged and r2.converged:
        assert r1.best_trajectory is not None
        assert r2.best_trajectory is not None
        assert abs(r1.best_trajectory.delta_v_total - r2.best_trajectory.delta_v_total) < 1e-6, (
            "Same seed must produce identical best Δv"
        )
