from pathlib import Path

import pytest

SPICE = (Path("data/spice_kernels") / "de440.bsp").exists()


@pytest.mark.skipif(not SPICE, reason="SPICE kernels required")
def test_pinn_dataset_generation() -> None:
    from astra.dsl.compiler import compile_mission
    from astra.dsl.parser import parse_mission_file
    from astra.neural.training.pipeline import generate_pinn_dataset
    from astra.physics.kernel import PhysicsKernel

    kernel = PhysicsKernel().load()
    dsl = parse_mission_file("data/benchmarks/earth_mars_2031.yaml")
    mission = compile_mission(dsl, kernel.ephemeris)

    X, dv_y, r1, r2, tof = generate_pinn_dataset(
        kernel,
        mission.origin_body,
        mission.destination_body,
        mission.departure_epoch_start,
        mission.departure_epoch_end,
        mission.tof_min_seconds,
        mission.tof_max_seconds,
        n_samples=100,
        seed=42,
    )
    assert X.shape == (100, 8)
    assert dv_y.shape == (100,)
    assert r1.shape == (100,)
    assert r2.shape == (100,)
    feasible_count = int((dv_y < 15.0).sum())
    assert feasible_count > 10, f"Only {feasible_count} feasible samples in 100"


@pytest.mark.skipif(not SPICE, reason="SPICE kernels required")
@pytest.mark.slow
def test_pinn_warmstart_quality() -> None:
    """PINN warm-start with 1000 trials must match standard 2000-trial quality."""
    from astra.dsl.compiler import compile_mission
    from astra.dsl.parser import parse_mission_file
    from astra.optimization.engine import (
        optimize_mission_bayesian,
        optimize_mission_pinn_accelerated,
    )
    from astra.physics.kernel import PhysicsKernel

    kernel = PhysicsKernel().load()
    dsl = parse_mission_file("data/benchmarks/earth_mars_2031.yaml")
    mission = compile_mission(dsl, kernel.ephemeris)

    r_standard = optimize_mission_bayesian(
        mission, kernel, n_trials=2000, time_limit=120.0, seed=42
    )
    r_pinn = optimize_mission_pinn_accelerated(
        mission,
        kernel,
        n_trials=1000,
        time_limit=120.0,
        seed=42,
        pinn_train_samples=300,
        pinn_epochs=30,
    )

    assert r_standard.converged and r_pinn.converged
    assert r_standard.best_trajectory is not None
    assert r_pinn.best_trajectory is not None
    dv_std = r_standard.best_trajectory.delta_v_total
    dv_pinn = r_pinn.best_trajectory.delta_v_total

    print(f"\nStandard (2000 trials): {dv_std:.4f} km/s")
    print(f"PINN warm-start (1000 trials): {dv_pinn:.4f} km/s")

    # PINN result must be within 5% of standard (same quality range)
    assert dv_pinn <= dv_std * 1.05, f"PINN {dv_pinn:.3f} is >5% worse than standard {dv_std:.3f}"
