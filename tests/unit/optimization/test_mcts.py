from pathlib import Path

import pytest
from astra.dsl.compiler import compile_mission
from astra.dsl.parser import parse_mission_file
from astra.optimization.mcts import MCTSPlanner, MCTSResult
from astra.physics.kernel import PhysicsKernel

SPICE = (Path("data/spice_kernels") / "de440.bsp").exists()


@pytest.mark.skipif(not SPICE, reason="SPICE kernels required")
def test_mcts_initialization_and_properties() -> None:
    """Verify MCTS components instantiate and have correct properties."""
    kernel = PhysicsKernel().load()
    dsl = parse_mission_file("data/benchmarks/earth_mars_2031.yaml")
    mission = compile_mission(dsl, kernel.ephemeris)

    planner = MCTSPlanner(
        mission,
        kernel,
        max_depth=3,
        n_iterations=50,
        dv_budget=20.0,
        seed=123,
    )

    # Check root properties
    assert planner.root.state.body == "EARTH"
    assert planner.root.state.dv_spent == 0.0
    assert len(planner.root.untried_actions) > 0
    assert planner.root.n_visits == 0
    assert planner.root.total_value == 0.0


@pytest.mark.skipif(not SPICE, reason="SPICE kernels required")
def test_mcts_planner_selection_and_expansion() -> None:
    """Verify select and expand methods operate as expected."""
    kernel = PhysicsKernel().load()
    dsl = parse_mission_file("data/benchmarks/earth_mars_2031.yaml")
    mission = compile_mission(dsl, kernel.ephemeris)

    planner = MCTSPlanner(
        mission,
        kernel,
        max_depth=2,
        n_iterations=10,
        dv_budget=15.0,
        seed=42,
    )

    # Select root (since it has untried actions, it shouldn't expand yet during select)
    selected = planner._select(planner.root)
    assert selected == planner.root

    # Expand root
    initial_actions_count = len(planner.root.untried_actions)
    child = planner._expand(planner.root)
    assert child is not None
    assert child.parent == planner.root
    assert len(planner.root.children) == 1
    assert len(planner.root.untried_actions) == initial_actions_count - 1

    # Check depth
    assert planner._get_node_depth(child) == 1


@pytest.mark.skipif(not SPICE, reason="SPICE kernels required")
def test_mcts_runs_successfully_for_earth_mars() -> None:
    """Verify that MCTS run converges and outputs paths reaching the destination."""
    kernel = PhysicsKernel().load()
    dsl = parse_mission_file("data/benchmarks/earth_mars_2031.yaml")
    mission = compile_mission(dsl, kernel.ephemeris)

    # Run with 100 iterations, shallow depth to find a path quickly
    planner = MCTSPlanner(
        mission,
        kernel,
        max_depth=3,
        n_iterations=100,
        dv_budget=30.0,
        seed=42,
        flyby_candidates=["VENUS"],
    )

    result = planner.run()
    assert isinstance(result, MCTSResult)
    paths = result.all_paths
    assert isinstance(paths, list)

    # If any paths reached Mars, check their validity
    for path in paths:
        assert path[0].body == "EARTH"
        assert path[-1].body == "MARS"
        assert path[-1].dv_spent > 0.0
        # Check intermediate flyby bodies
        if len(path) > 2:
            for state in path[1:-1]:
                assert state.body == "VENUS"
