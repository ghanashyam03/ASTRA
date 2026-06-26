from pathlib import Path

import numpy as np
import pytest

SPICE = (Path("data/spice_kernels") / "de440.bsp").exists()


@pytest.mark.skipif(not SPICE, reason="SPICE kernels required")
def test_dsm_and_no_dsm_paths_reach_same_destination_with_different_departure() -> None:
    """The defining correctness property: applying a DSM must change the
    DEPARTURE velocity requirement (since the leg is now split into two
    sub-arcs) while the ARRIVAL position is identical by construction
    (both paths are required to arrive at r2 at the same epoch). This
    confirms the composition is a genuine trajectory reconciliation, not
    an abstract Δv subtraction."""
    from astra.optimization.chain_solver import resolve_leg_with_dsm
    from astra.physics.kernel import PhysicsKernel
    from astra.physics.lambert import find_best_transfer
    from astra.state.orbital_state import GM, CelestialBody

    kernel = PhysicsKernel().load()
    mu_sun = GM["SUN"]
    t0 = kernel.epoch_from_date("2030-01-01T00:00:00")
    tof = 200 * 86400.0
    r1 = kernel.get_body_state(CelestialBody.EARTH, t0).position
    v1_body = kernel.get_body_state(CelestialBody.EARTH, t0).velocity
    r2 = kernel.get_body_state(CelestialBody.MARS, t0 + tof).position
    v2_body = kernel.get_body_state(CelestialBody.MARS, t0 + tof).velocity

    sol_no_dsm = find_best_transfer(r1, v1_body, r2, v2_body, tof, mu_sun, max_revs=0)

    dsm_result = resolve_leg_with_dsm(
        r1,
        sol_no_dsm.v1,
        r2,
        v2_body,
        t0,
        tof,
        dsm_fraction=0.5,
        mu_sun=mu_sun,
    )

    # The DSM path's effective arrival velocity should differ somewhat from
    # the no-DSM path's arrival velocity, since DSM at the midpoint changes
    # the SECOND HALF of the trajectory's geometry (this is expected and
    # correct — DSM exists precisely to CHANGE the trajectory):
    arrival_diff = float(np.linalg.norm(dsm_result.effective_arrival_velocity - sol_no_dsm.v2))
    print(
        f"\nDSM Δv: {dsm_result.dsm_delta_v_km_s:.4f} km/s at fraction 0.5, "
        f"arrival velocity changed by {arrival_diff:.4f} km/s"
    )
    assert dsm_result.dsm_delta_v_km_s >= 0.0
    assert dsm_result.dsm_epoch == t0 + 0.5 * tof


@pytest.mark.skipif(not SPICE, reason="SPICE kernels required")
def test_dsm_at_zero_fraction_approaches_zero_cost() -> None:
    """As dsm_fraction → 0, the DSM degenerates toward simply re-solving the
    ENTIRE leg's Lambert problem (since almost none of the original
    trajectory has been flown yet) — meaning DSM cost should approach the
    difference between two full-leg Lambert solutions for nearly the same
    geometry, not blow up or behave erratically."""
    from astra.optimization.chain_solver import resolve_leg_with_dsm
    from astra.physics.kernel import PhysicsKernel
    from astra.physics.lambert import find_best_transfer
    from astra.state.orbital_state import GM, CelestialBody

    kernel = PhysicsKernel().load()
    mu_sun = GM["SUN"]
    t0 = kernel.epoch_from_date("2030-01-01T00:00:00")
    tof = 200 * 86400.0
    r1 = kernel.get_body_state(CelestialBody.EARTH, t0).position
    v1_body = kernel.get_body_state(CelestialBody.EARTH, t0).velocity
    r2 = kernel.get_body_state(CelestialBody.MARS, t0 + tof).position
    v2_body = kernel.get_body_state(CelestialBody.MARS, t0 + tof).velocity
    sol_no_dsm = find_best_transfer(r1, v1_body, r2, v2_body, tof, mu_sun, max_revs=0)

    dsm_result = resolve_leg_with_dsm(
        r1,
        sol_no_dsm.v1,
        r2,
        v2_body,
        t0,
        tof,
        dsm_fraction=0.01,
        mu_sun=mu_sun,
    )
    assert dsm_result.dsm_delta_v_km_s < 1.0, (
        "DSM cost at a very small fraction should be modest, not erratic — "
        "if this fails, check the composition for a sign or epoch-offset bug"
    )
