from pathlib import Path

import pytest

from astra.dsl.compiler import compile_mission
from astra.dsl.parser import parse_mission_file
from astra.optimization.engine import optimize_mission_mcts
from astra.optimization.mcts import MCTSPlanner, MCTSResult
from astra.physics.kernel import PhysicsKernel

SPICE = (Path("data/spice_kernels") / "de440.bsp").exists()


@pytest.mark.skipif(not SPICE, reason="SPICE kernels required")
def test_mcts_planner_returns_mcts_result() -> None:
    """Verify that MCTS run returns an MCTSResult object with correct fields."""
    kernel = PhysicsKernel().load()
    dsl = parse_mission_file("data/benchmarks/earth_mars_2031.yaml")
    mission = compile_mission(dsl, kernel.ephemeris)

    planner = MCTSPlanner(
        mission,
        kernel,
        max_depth=3,
        n_iterations=20,
        dv_budget=30.0,
        seed=42,
        flyby_candidates=["VENUS"],
    )

    result = planner.run()
    assert isinstance(result, MCTSResult)
    assert isinstance(result.best_sequence, list)
    assert isinstance(result.best_dv_total, float)
    assert isinstance(result.all_paths, list)
    assert result.n_iterations == 20
    assert result.wall_time_s >= 0.0

    # Test dictionary serialization
    d = result.to_dict()
    assert "best_sequence" in d
    assert "best_dv_total" in d
    assert "all_paths" in d
    assert "n_iterations" in d
    assert "wall_time_s" in d
    assert "converged" in d
    assert isinstance(d["all_paths"], list)


@pytest.mark.skipif(not SPICE, reason="SPICE kernels required")
def test_optimize_mission_mcts() -> None:
    """Verify engine optimize_mission_mcts interface and Trajectory construction."""
    kernel = PhysicsKernel().load()
    dsl = parse_mission_file("data/benchmarks/earth_mars_2031.yaml")
    mission = compile_mission(dsl, kernel.ephemeris)

    opt_result = optimize_mission_mcts(
        mission=mission,
        kernel=kernel,
        flyby_candidates=["VENUS"],
        n_iterations=50,
        dv_budget=30.0,
        seed=42,
    )

    assert opt_result.optimizer_strategy is None  # default OptimizationResult field
    assert isinstance(opt_result.wall_time_s, float)
    assert isinstance(opt_result.n_evaluations, int)

    if opt_result.converged:
        traj = opt_result.best_trajectory
        assert traj is not None
        assert len(traj.states) == 2  # origin and destination
        assert len(traj.maneuvers) >= 1  # at least DEP and CAP

        # Validate maneuver labeling
        assert traj.maneuvers[0].label == "DEP"
        assert traj.maneuvers[-1].label == "CAP"

        if len(traj.maneuvers) > 2:
            # Check flyby labels
            for m in traj.maneuvers[1:-1]:
                assert m.label.startswith("FLY_")


@pytest.mark.skipif(not SPICE, reason="SPICE kernels required")
def test_surrogate_guided_mcts() -> None:
    """Verify that surrogate-guided MCTS executes and applies uncertainty penalties."""
    import numpy as np

    kernel = PhysicsKernel().load()
    dsl = parse_mission_file("data/benchmarks/earth_mars_2031.yaml")
    mission = compile_mission(dsl, kernel.ephemeris)

    from astra.neural.pinn import LambertPINNEnsemble

    surrogate = LambertPINNEnsemble(hidden_dims=[8], ensemble_size=2)
    x_dummy = np.random.randn(5, 8).astype(np.float32)
    y_dummy = np.random.randn(5, 6).astype(np.float32)
    r1_norms = np.ones(5, dtype=np.float32) * 1.5e8
    r2_norms = np.ones(5, dtype=np.float32) * 2.2e8
    tof_sec = np.ones(5, dtype=np.float32) * 200.0 * 86400.0
    surrogate.train_on_dataset(
        x_dummy, y_dummy, r1_norms, r2_norms, tof_sec, epochs=1, batch_size=2
    )

    planner = MCTSPlanner(
        mission=mission,
        kernel=kernel,
        max_depth=3,
        n_iterations=20,
        dv_budget=30.0,
        seed=42,
        flyby_candidates=["VENUS"],
        surrogate=surrogate,
        uncertainty_weight=0.5,
    )

    result = planner.run()
    assert isinstance(result, MCTSResult)
    # Check that children states store uncertainty and predicted_dv
    for child in planner.root.children:
        assert child.state.uncertainty >= 0.0
        assert child.state.predicted_dv >= 0.0
