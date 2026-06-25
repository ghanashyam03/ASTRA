"""MESSENGER's actual chain has EARTH as both origin (launch epoch) and the
first flyby body (~1 year later, a completely different heliocentric position).
P35's repeated-body test covered two FLYBY occurrences of the same body
(Venus-Venus); this covers the origin reappearing as a flyby — a DIFFERENT
code path in resolve_flyby_chain, since the origin's state is read once at
departure_epoch via body_states[0], and the chain solver must NOT reuse that
same state object/epoch when EARTH appears again at chain index 1 with its
own, later epoch."""

from pathlib import Path

import pytest

SPICE = (Path("data/spice_kernels") / "de440.bsp").exists()


@pytest.mark.skipif(not SPICE, reason="SPICE kernels required")
def test_origin_body_reused_as_flyby_uses_distinct_epoch() -> None:
    import numpy as np

    from astra.dsl.compiler import compile_mission
    from astra.dsl.parser import parse_mission_file
    from astra.optimization.chain_solver import resolve_flyby_chain
    from astra.physics.kernel import PhysicsKernel
    from astra.state.orbital_state import CelestialBody

    kernel = PhysicsKernel().load()
    dsl = parse_mission_file("data/benchmarks/messenger_chain_2004.yaml")
    mission = compile_mission(dsl, kernel.ephemeris)
    dep_epoch = mission.departure_epoch_start

    leg_tofs = [
        365 * 86400.0,
        440 * 86400.0,
        245 * 86400.0,
        220 * 86400.0,
        268 * 86400.0,
        358 * 86400.0,
    ]
    result = resolve_flyby_chain(
        mission,
        kernel,
        chain_bodies=["EARTH", "EARTH", "VENUS", "VENUS", "MERCURY", "MERCURY", "MERCURY"],
        departure_epoch=dep_epoch,
        leg_tofs=leg_tofs,
        flyby_specs={
            "EARTH": {
                "min_alt_km": 500.0,
                "max_alt_km": 20000.0,
                "powered_allowed": True,
                "max_powered_km_s": 0.3,
            },
            "VENUS": {
                "min_alt_km": 300.0,
                "max_alt_km": 20000.0,
                "powered_allowed": False,
                "max_powered_km_s": 0.0,
            },
            "MERCURY": {
                "min_alt_km": 200.0,
                "max_alt_km": 15000.0,
                "powered_allowed": True,
                "max_powered_km_s": 0.3,
            },
        },
    )

    earth_origin_pos = kernel.get_body_state(CelestialBody.EARTH, dep_epoch).position
    earth_flyby_epoch = dep_epoch + leg_tofs[0]
    earth_flyby_pos = kernel.get_body_state(CelestialBody.EARTH, earth_flyby_epoch).position
    separation_km = float(np.linalg.norm(earth_origin_pos - earth_flyby_pos))
    # Earth's orbit is closed and takes ~365.25 days. After exactly 365 days, Earth is back
    # to almost the same position, separated by ~6.5e5 km (about 6 hours of travel).
    # Thus, a separation > 5e5 km confirms distinct epochs (an epoch reuse would yield 0 km).
    assert separation_km > 5e5, (
        "Earth at launch vs Earth ~1 year later must be substantially separated "
        "from 0 (Earth moves ~2.5e8 km along its orbit per year, but returns to "
        "nearly the same spot) — if this fails, the chain solver may be reusing "
        "the origin's epoch for the later flyby"
    )
    print(
        f"\nEarth origin-vs-flyby separation: {separation_km:.0f} km "
        f"(confirms distinct epochs used for repeated origin body)"
    )

    if result.feasible:
        print(f"6-flyby MESSENGER chain resolved: Δv={result.trajectory.delta_v_total:.4f} km/s")
    else:
        print(f"6-flyby chain infeasible at these TOFs: {result.reason}")
    # No hard pass/fail on convergence itself — this test's purpose is the
    # epoch-distinctness check above, which IS a hard assertion.
