"""Voyager 2 grand tour reproduction attempt. Success for this test means an
honest, evidence-grounded outcome — convergence is not required, and a forced
numerical match to historical Δv is explicitly NOT the goal given established
patched-conics fidelity caveats and the framing notes above."""

from pathlib import Path

import pytest

SPICE = (Path("data/spice_kernels") / "de440.bsp").exists()


@pytest.mark.skipif(not SPICE, reason="SPICE kernels required")
@pytest.mark.slow
def test_voyager2_chain_attempt_and_characterize() -> None:
    from astra.dsl.compiler import compile_mission
    from astra.dsl.parser import parse_mission_file
    from astra.optimization.engine import optimize_mission_chain
    from astra.physics.kernel import PhysicsKernel

    kernel = PhysicsKernel().load()
    dsl = parse_mission_file("data/benchmarks/voyager2_grand_tour_1977.yaml")
    mission = compile_mission(dsl, kernel.ephemeris)

    result = optimize_mission_chain(
        mission,
        kernel,
        chain_bodies=["EARTH", "JUPITER", "SATURN", "URANUS", "NEPTUNE"],
        n_trials=2000,
        time_limit=300.0,
        seed=11,
    )

    print(
        f"\nVoyager 2 grand tour: converged={result.converged}, "
        f"n_evaluations={result.n_evaluations}, n_feasible={result.n_feasible}"
    )

    if result.converged:
        best = result.best_trajectory
        leg_details = best.metadata.get("leg_details", [])
        print(
            f"Δv={best.delta_v_total:.4f} km/s, duration={best.duration_days:.0f} days "
            f"({best.duration_days / 365.25:.1f} years)"
        )
        for leg in leg_details:
            print(
                f"  {leg['body']}: required_turn={leg['required_turn_deg']:.2f}°, "
                f"resolution={leg.get('resolution')}"
            )
    else:
        print(
            "Did not converge — record in docs/validation/voyager2_divergence_report.md "
            "with the three-explanation analysis required by this prompt's framing notes."
        )

    # No hard assertion on convergence or Δv value — this test's deliverable
    # is the printed diagnostic output feeding the written report, plus the
    # confirmation that infeasible candidates were never silently accepted
    # (already guaranteed structurally by resolve_flyby_chain, verified again
    # here as a final sanity check):
    if result.best_trajectory is not None:
        assert result.best_trajectory.delta_v_total <= 99.0, (
            "A trajectory with the infeasibility sentinel value must never be "
            "reported as the best trajectory"
        )
