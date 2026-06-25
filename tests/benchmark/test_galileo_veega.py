"""Galileo VEEGA reproduction — Venus-Earth-Earth-Jupiter chain, 3 flybys.
First test of the chain solver beyond a single flyby. This is a fidelity-ceiling
test as much as a correctness test: patched-conics two-body chaining over a
multi-year, multi-flyby sequence WILL diverge from the historical trajectory at
some point, and this test's job is to characterize exactly where and why, not
to force an artificial match."""

from pathlib import Path

import pytest

SPICE = (Path("data/spice_kernels") / "de440.bsp").exists()


@pytest.mark.skipif(not SPICE, reason="SPICE kernels required")
@pytest.mark.slow
def test_galileo_veega_chain_resolves_self_consistently() -> None:
    from astra.dsl.compiler import compile_mission
    from astra.dsl.parser import parse_mission_file
    from astra.optimization.engine import optimize_mission_chain
    from astra.physics.kernel import PhysicsKernel

    kernel = PhysicsKernel().load()
    dsl = parse_mission_file("data/benchmarks/galileo_veega_1989.yaml")
    mission = compile_mission(dsl, kernel.ephemeris)

    result = optimize_mission_chain(
        mission,
        kernel,
        chain_bodies=["EARTH", "VENUS", "EARTH", "EARTH", "JUPITER"],
        n_trials=1500,
        time_limit=240.0,
        seed=7,
    )

    print(
        f"\nGalileo VEEGA chain optimization: converged={result.converged}, "
        f"n_evaluations={result.n_evaluations}, n_feasible={result.n_feasible}"
    )

    # Capture and print a histogram of the rejection causes
    print("\n=== Rejection Causes Histogram ===")
    rejections = result.rejection_reasons
    total_rejections = sum(rejections.values())
    for cause, count in sorted(rejections.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_rejections * 100) if total_rejections > 0 else 0
        bar = "#" * int(percentage // 5)
        print(f"  {cause:<25}: {count:>4} ({percentage:>5.1f}%) {bar}")
    print("==================================\n")

    assert len(rejections) > 0, "Should have captured at least one rejection reason"

    if result.converged:
        best = result.best_trajectory
        assert best is not None
        leg_details = best.metadata.get("leg_details", [])
        print(f"Best Δv: {best.delta_v_total:.4f} km/s over {best.duration_days:.1f} days")
        for leg in leg_details:
            print(
                f"  {leg['body']}: required_turn={leg['required_turn_deg']:.2f}°, "
                f"resolution={leg.get('resolution')}, dv={leg.get('dv_km_s', 0.0):.4f} km/s"
            )
        for leg in leg_details:
            assert leg["required_turn_deg"] >= 0.0
            if leg.get("resolution") == "unpowered":
                assert leg["required_turn_deg"] <= leg["max_unpowered_turn_deg"] + 0.5
    else:
        print(
            "Galileo VEEGA chain did not converge within budget at this fidelity "
            "level. This is a legitimate, reportable finding about the limits of "
            "patched-conics chaining over a 3-flyby, multi-year sequence — record "
            "it in docs/validation/galileo_veega_report.md rather than forcing "
            "convergence by loosening constraints arbitrarily."
        )
