import numpy as np
import pytest
from pathlib import Path
from astra.neural.gnn import (
    SolarSystemGNN, build_node_features, build_edge_features,
    PLANET_NODES, N_NODES, NODE_FEATURE_DIM, EDGE_FEATURE_DIM
)

def test_node_features_shape_and_determinism():
    f1 = build_node_features()
    f2 = build_node_features()
    assert f1.shape == (N_NODES, NODE_FEATURE_DIM)
    assert f1.dtype == np.float32
    np.testing.assert_array_equal(f1, f2)  # deterministic

def test_node_features_in_range():
    f = build_node_features()
    assert np.all(f >= 0.0) and np.all(f <= 1.0)

def test_edge_features_shape():
    r1 = np.array([1.496e8, 0.0, 0.0])
    v1 = np.array([0.0, 29.78, 0.0])
    r2 = np.array([0.0, 2.279e8, 0.0])
    feat = build_edge_features("EARTH", "MARS", r1, r2, v1, 200*86400.0)
    assert feat.shape == (EDGE_FEATURE_DIM,)
    assert feat.dtype == np.float32
    assert np.all(np.isfinite(feat))

def test_gnn_forward_pass():
    gnn = SolarSystemGNN(hidden_dim=16)
    nf = build_node_features()
    r1 = np.array([1.496e8, 0.0, 0.0])
    v1 = np.array([0.0, 29.78, 0.0])
    r2 = np.array([0.0, 2.279e8, 0.0])
    edge_dict = {
        (2, 4): build_edge_features("EARTH", "MARS", r1, r2, v1, 200*86400.0)
    }
    scores = gnn.predict_scores("EARTH", ["MARS", "VENUS"], nf, edge_dict)
    assert scores.shape == (2,)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)

def test_mcts_gnn_none_unchanged():
    """When gnn=None MCTSPlanner must initialize without error."""
    from pathlib import Path
    if not (Path("data/spice_kernels") / "de440.bsp").exists():
        pytest.skip("SPICE kernels required")
    from astra.physics.kernel import PhysicsKernel
    from astra.dsl.parser import parse_mission_file
    from astra.dsl.compiler import compile_mission
    from astra.optimization.mcts import MCTSPlanner
    kernel = PhysicsKernel().load()
    dsl = parse_mission_file("data/benchmarks/earth_mars_2031.yaml")
    mission = compile_mission(dsl, kernel.ephemeris)
    planner = MCTSPlanner(mission, kernel, n_iterations=5, gnn=None)
    result = planner.run()
    assert hasattr(result, "converged")

def test_gnn_training_logic():
    """Verify that train_on_mcts_results updates only output head parameters."""
    from astra.optimization.mcts import PhaseState
    from astra.physics.kernel import PhysicsKernel
    
    gnn = SolarSystemGNN(hidden_dim=16, lr=1e-2)
    nf = build_node_features()
    
    # Check that initially it's not trained
    assert not gnn.is_trained()
    
    # Save initial weights
    W_out_init = gnn.W_out.copy()
    b_out_init = gnn.b_out.copy()
    W_msg1_init = gnn.W_msg1.copy()
    
    # Construct a dummy path
    path = [
        PhaseState(body="EARTH", epoch=0.0, v_helio=np.zeros(3), dv_spent=0.0),
        PhaseState(body="MARS", epoch=200 * 86400.0, v_helio=np.zeros(3), dv_spent=5.0)
    ]
    
    # We need a mock kernel or loaded kernel
    from pathlib import Path
    if not (Path("data/spice_kernels") / "de440.bsp").exists():
        pytest.skip("SPICE kernels required")
    kernel = PhysicsKernel().load()
    
    # Train
    loss = gnn.train_on_mcts_results(
        mcts_paths=[path],
        rewards=[0.8],
        node_features=nf,
        kernel=kernel,
        epochs=5,
        seed=42
    )
    
    assert gnn.is_trained()
    assert len(loss) == 5
    
    # Output weights should be updated
    assert not np.array_equal(gnn.W_out, W_out_init)
    assert not np.array_equal(gnn.b_out, b_out_init)
    
    # Message-passing weights must remain frozen
    np.testing.assert_array_equal(gnn.W_msg1, W_msg1_init)
