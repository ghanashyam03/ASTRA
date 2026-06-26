import math
from pathlib import Path

import numpy as np
import pytest

SPICE = (Path("data/spice_kernels") / "de440.bsp").exists()


@pytest.mark.skipif(not SPICE, reason="SPICE kernels required")
def test_audit_catches_reconstructed_venus_bug_signature() -> None:
    """Construct a trajectory with the EXACT signature of the original bug:
    a FLY_VENUS maneuver reporting near-zero Δv, but states implying a
    turn angle far beyond Venus's physical ceiling. The auditor must raise."""
    from astra.optimization.audit import AuditFailure, audit_trajectory_physics
    from astra.physics.kernel import PhysicsKernel
    from astra.state.orbital_state import CelestialBody, OrbitalState
    from astra.state.trajectory import Maneuver, Trajectory

    kernel = PhysicsKernel().load()
    epoch = kernel.epoch_from_date("2032-06-01T00:00:00")
    venus_state = kernel.get_body_state(CelestialBody.VENUS, epoch)

    # Construct an implausible incoming/outgoing pair implying a huge turn
    v_inf_in_mag = 9.7
    v_inf_in = venus_state.velocity + np.array([v_inf_in_mag, 0.0, 0.0])
    # Outgoing direction rotated by an extreme angle relative to incoming
    v_inf_out = venus_state.velocity + np.array(
        [-v_inf_in_mag * 0.9, v_inf_in_mag * 0.4, 0.0]
    )  # large directional change

    state_before = OrbitalState(
        epoch=epoch - 86400.0,
        position=venus_state.position,
        velocity=v_inf_in,
        central_body=CelestialBody.SUN,
    )
    state_at = OrbitalState(
        epoch=epoch,
        position=venus_state.position,
        velocity=v_inf_out,
        central_body=CelestialBody.SUN,
    )

    bogus_maneuver = Maneuver(epoch=epoch, delta_v=np.zeros(3), label="FLY_VENUS")
    traj = Trajectory(states=[state_before, state_at], maneuvers=[bogus_maneuver])

    with pytest.raises(AuditFailure) as exc_info:
        audit_trajectory_physics(traj, kernel)
    assert "ORIGINAL BUG" in str(exc_info.value)


@pytest.mark.skipif(not SPICE, reason="SPICE kernels required")
def test_audit_passes_clean_mock_saturn_flyby() -> None:
    """Construct a trajectory with a physically feasible Saturn flyby.
    The auditor must pass it cleanly."""
    from astra.optimization.audit import audit_trajectory_physics
    from astra.physics.kernel import PhysicsKernel
    from astra.state.orbital_state import CelestialBody, OrbitalState
    from astra.state.trajectory import Maneuver, Trajectory

    kernel = PhysicsKernel().load()
    epoch = kernel.epoch_from_date("2032-06-01T00:00:00")
    saturn_state = kernel.get_body_state(CelestialBody.SATURN, epoch)

    v_inf_in_mag = 15.0
    # Incoming direction along X
    v_inf_in = saturn_state.velocity + np.array([v_inf_in_mag, 0.0, 0.0])

    # Outgoing rotated by 35 degrees in XY plane
    theta = math.radians(35.0)
    v_inf_out = saturn_state.velocity + np.array(
        [v_inf_in_mag * math.cos(theta), v_inf_in_mag * math.sin(theta), 0.0]
    )

    state_before = OrbitalState(
        epoch=epoch - 86400.0,
        position=saturn_state.position,
        velocity=v_inf_in,
        central_body=CelestialBody.SUN,
    )
    state_at = OrbitalState(
        epoch=epoch,
        position=saturn_state.position,
        velocity=v_inf_out,
        central_body=CelestialBody.SUN,
    )

    maneuver = Maneuver(epoch=epoch, delta_v=np.zeros(3), label="FLY_SATURN")
    traj = Trajectory(states=[state_before, state_at], maneuvers=[maneuver])

    audit_results = audit_trajectory_physics(traj, kernel)
    assert len(audit_results) == 1
    assert audit_results[0].is_self_consistent
    assert audit_results[0].body == "SATURN"
    assert abs(audit_results[0].required_turn_deg - 35.0) < 1e-3


@pytest.mark.skipif(not SPICE, reason="SPICE kernels required")
def test_audit_passes_clean_chain_solver_output() -> None:
    """Any trajectory produced by the GATED chain solver (Prompt 31+) must
    pass this audit cleanly — it was already proven feasible at construction
    time, so the independent re-derivation should agree."""
    from astra.dsl.compiler import compile_mission
    from astra.dsl.parser import parse_mission_file
    from astra.optimization.audit import audit_trajectory_physics
    from astra.optimization.chain_solver import resolve_flyby_chain
    from astra.physics.kernel import PhysicsKernel

    kernel = PhysicsKernel().load()
    dsl = parse_mission_file("data/benchmarks/earth_venus_mars_2032.yaml")
    mission = compile_mission(dsl, kernel.ephemeris)
    dep_epoch = mission.departure_epoch_start

    result = resolve_flyby_chain(
        mission,
        kernel,
        ["EARTH", "VENUS", "MARS"],
        dep_epoch,
        [150 * 86400.0, 180 * 86400.0],
        {
            "VENUS": {
                "min_alt_km": 300.0,
                "max_alt_km": 15000.0,
                "powered_allowed": True,
                "max_powered_km_s": 1.0,
            }
        },
    )
    if result.feasible:
        audit_results = audit_trajectory_physics(result.trajectory, kernel)
        assert all(r.is_self_consistent for r in audit_results)
        print(
            f"\nClean chain solver output passed audit: {len(audit_results)} "
            f"flyby maneuvers verified self-consistent"
        )
    else:
        pytest.skip(f"Chain infeasible at this TOF — no trajectory to audit: {result.reason}")


@pytest.mark.skipif(not SPICE, reason="SPICE kernels required")
def test_persistence_hook_triggers_audit_failure() -> None:
    """Verify that save_trajectory with audit_before_save=True propagates AuditFailure
    when saving a trajectory that fails the audit, and succeeds otherwise."""
    from astra.data.storage import TrajectoryStore
    from astra.optimization.audit import AuditFailure
    from astra.physics.kernel import PhysicsKernel
    from astra.state.orbital_state import CelestialBody, OrbitalState
    from astra.state.trajectory import Maneuver, Trajectory

    kernel = PhysicsKernel().load()
    epoch = kernel.epoch_from_date("2032-06-01T00:00:00")
    venus_state = kernel.get_body_state(CelestialBody.VENUS, epoch)

    # 1. Construct an infeasible Venus flyby (bogus)
    v_inf_in_mag = 9.7
    v_inf_in = venus_state.velocity + np.array([v_inf_in_mag, 0.0, 0.0])
    v_inf_out = venus_state.velocity + np.array([-v_inf_in_mag * 0.9, v_inf_in_mag * 0.4, 0.0])
    state_before_bad = OrbitalState(
        epoch=epoch - 86400.0,
        position=venus_state.position,
        velocity=v_inf_in,
        central_body=CelestialBody.SUN,
    )
    state_at_bad = OrbitalState(
        epoch=epoch,
        position=venus_state.position,
        velocity=v_inf_out,
        central_body=CelestialBody.SUN,
    )
    bogus_maneuver = Maneuver(epoch=epoch, delta_v=np.zeros(3), label="FLY_VENUS")
    bad_traj = Trajectory(states=[state_before_bad, state_at_bad], maneuvers=[bogus_maneuver])

    store = TrajectoryStore()

    # Saving without audit must succeed
    tid = store.save_trajectory(bad_traj, "test_mission", audit_before_save=False)
    assert tid is not None

    # Saving with audit must raise AuditFailure
    with pytest.raises(AuditFailure):
        store.save_trajectory(bad_traj, "test_mission", audit_before_save=True, kernel=kernel)

    # 2. Construct a feasible Saturn flyby
    saturn_state = kernel.get_body_state(CelestialBody.SATURN, epoch)
    v_saturn_in = saturn_state.velocity + np.array([15.0, 0.0, 0.0])
    theta = math.radians(35.0)
    v_saturn_out = saturn_state.velocity + np.array(
        [15.0 * math.cos(theta), 15.0 * math.sin(theta), 0.0]
    )
    state_before_good = OrbitalState(
        epoch=epoch - 86400.0,
        position=saturn_state.position,
        velocity=v_saturn_in,
        central_body=CelestialBody.SUN,
    )
    state_at_good = OrbitalState(
        epoch=epoch,
        position=saturn_state.position,
        velocity=v_saturn_out,
        central_body=CelestialBody.SUN,
    )
    good_maneuver = Maneuver(epoch=epoch, delta_v=np.zeros(3), label="FLY_SATURN")
    good_traj = Trajectory(states=[state_before_good, state_at_good], maneuvers=[good_maneuver])

    # Saving with audit must succeed
    tid_good = store.save_trajectory(
        good_traj, "test_mission", audit_before_save=True, kernel=kernel
    )
    assert tid_good is not None
