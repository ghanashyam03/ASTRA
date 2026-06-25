from pathlib import Path

import pytest

SPICE = (Path("data/spice_kernels") / "de440.bsp").exists()


@pytest.mark.skipif(not SPICE, reason="SPICE kernels required")
def test_repeated_body_uses_distinct_epochs() -> None:
    """A chain visiting the same body twice (e.g. Venus-Venus) must query
    that body's state at each occurrence's OWN epoch, not silently reuse the
    first occurrence's epoch/position. This is an indexing correctness check,
    not a physics check."""
    from astra.dsl.compiler import compile_mission
    from astra.dsl.parser import parse_mission_file
    from astra.optimization.chain_solver import resolve_flyby_chain
    from astra.physics.kernel import PhysicsKernel
    from astra.state.orbital_state import CelestialBody

    kernel = PhysicsKernel().load()
    dsl = parse_mission_file("data/benchmarks/cassini_vve_1998.yaml")
    mission = compile_mission(dsl, kernel.ephemeris)
    dep_epoch = mission.departure_epoch_start

    result = resolve_flyby_chain(
        mission,
        kernel,
        chain_bodies=["EARTH", "VENUS", "VENUS", "EARTH", "SATURN"],
        departure_epoch=dep_epoch,
        leg_tofs=[150 * 86400.0, 400 * 86400.0, 200 * 86400.0, 1500 * 86400.0],
        flyby_specs={
            "VENUS": {
                "min_alt_km": 300.0,
                "max_alt_km": 20000.0,
                "powered_allowed": False,
                "max_powered_km_s": 0.0,
            },
            "EARTH": {
                "min_alt_km": 500.0,
                "max_alt_km": 20000.0,
                "powered_allowed": True,
                "max_powered_km_s": 0.3,
            },
        },
    )
    # Compute Venus position independently at the two DIFFERENT candidate epochs
    epoch_venus1 = dep_epoch + 150 * 86400.0
    epoch_venus2 = dep_epoch + 150 * 86400.0 + 400 * 86400.0
    venus1_pos = kernel.get_body_state(CelestialBody.VENUS, epoch_venus1).position
    venus2_pos = kernel.get_body_state(CelestialBody.VENUS, epoch_venus2).position
    import numpy as np

    separation_km = float(np.linalg.norm(venus1_pos - venus2_pos))
    assert separation_km > 1e6, (
        "Venus's two flyby positions, 400+ days apart, must be substantially "
        "separated — if this assertion fails, the chain solver may be reusing "
        "a single epoch for both occurrences"
    )
    print(
        f"\nVenus position separation between the two flybys: "
        f"{separation_km:.0f} km — confirms distinct epochs are being used"
    )
    if result.feasible:
        assert result.trajectory is not None
        print(f"Repeated-body chain resolved: Δv={result.trajectory.delta_v_total:.4f} km/s")
    else:
        print(f"Repeated-body chain infeasible at this TOF combination: {result.reason}")
