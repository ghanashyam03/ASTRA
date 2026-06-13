import time
import numpy as np
from pathlib import Path
from astra.physics.kernel import PhysicsKernel
from astra.dsl.parser import parse_mission_file
from astra.dsl.compiler import compile_mission
from astra.optimization.mcts import MCTSPlanner, PhaseState
from astra.neural.gnn import SolarSystemGNN, build_node_features
from astra.state.orbital_state import CelestialBody

def count_nodes(node):
    return 1 + sum(count_nodes(c) for c in node.children)

def calculate_explored_nodes(planner):
    return count_nodes(planner.root)

def run_mission_benchmark(name, mission, kernel, max_depth, n_iterations, dv_budget, flyby_candidates, seed=42):
    print(f"\n==========================================")
    print(f"BENCHMARK: {name}")
    print(f"==========================================")

    # 1. Standard MCTS (no GNN)
    planner_std = MCTSPlanner(
        mission=mission,
        kernel=kernel,
        max_depth=max_depth,
        n_iterations=n_iterations,
        dv_budget=dv_budget,
        seed=seed,
        flyby_candidates=flyby_candidates,
        gnn=None
    )
    
    start_time = time.perf_counter()
    res_std = planner_std.run()
    std_time = time.perf_counter() - start_time
    std_nodes = calculate_explored_nodes(planner_std)
    
    print(f"Standard MCTS:")
    print(f"  Converged: {res_std.converged}")
    print(f"  Best Delta-V: {res_std.best_dv_total:.3f} km/s" if res_std.converged else "  Best Delta-V: N/A")
    print(f"  Explored Nodes: {std_nodes}")
    print(f"  Runtime: {std_time:.3f} s")
    print(f"  Paths Found: {len(res_std.all_paths)}")

    # 2. Train Graph-Based Policy Model (GNN) on Standard MCTS results
    gnn = SolarSystemGNN(hidden_dim=32, lr=1e-2, seed=seed)
    node_features = build_node_features()
    
    # Calculate rewards for all paths found by standard MCTS
    paths = res_std.all_paths
    if paths:
        rewards = [max(0.0, 1.0 - p[-1].dv_spent / dv_budget) for p in paths]
        print(f"Training Graph-Based Policy Model on {len(paths)} paths...")
        gnn.train_on_mcts_results(
            mcts_paths=paths,
            rewards=rewards,
            node_features=node_features,
            kernel=kernel,
            epochs=20,
            seed=seed
        )
    else:
        # If no paths found, pre-train on a dummy path to simulate having a trained model
        print("No successful paths found. Pre-training on a dummy path to verify execution path...")
        dummy_path = [
            PhaseState(body="EARTH", epoch=mission.departure_epoch_start, v_helio=np.zeros(3), dv_spent=0.0),
            PhaseState(body=mission.destination_body.name, epoch=mission.departure_epoch_start + 200*86400, v_helio=np.zeros(3), dv_spent=5.0)
        ]
        gnn.train_on_mcts_results(
            mcts_paths=[dummy_path],
            rewards=[0.9],
            node_features=node_features,
            kernel=kernel,
            epochs=1,
            seed=seed
        )
    
    # 3. GNN-Guided MCTS
    planner_gnn = MCTSPlanner(
        mission=mission,
        kernel=kernel,
        max_depth=max_depth,
        n_iterations=n_iterations,
        dv_budget=dv_budget,
        seed=seed,
        flyby_candidates=flyby_candidates,
        gnn=gnn
    )
    
    start_time = time.perf_counter()
    res_gnn = planner_gnn.run()
    gnn_time = time.perf_counter() - start_time
    gnn_nodes = calculate_explored_nodes(planner_gnn)
    
    print(f"GNN-Guided MCTS:")
    print(f"  Converged: {res_gnn.converged}")
    print(f"  Best Delta-V: {res_gnn.best_dv_total:.3f} km/s" if res_gnn.converged else "  Best Delta-V: N/A")
    print(f"  Explored Nodes: {gnn_nodes}")
    print(f"  Runtime: {gnn_time:.3f} s")
    print(f"  Paths Found: {len(res_gnn.all_paths)}")
    
    # Compare
    node_reduction = (std_nodes - gnn_nodes) / std_nodes * 100 if std_nodes > 0 else 0
    time_diff = (gnn_time - std_time) / std_time * 100 if std_time > 0 else 0
    
    print(f"Comparison:")
    print(f"  Node Exploration Reduction: {node_reduction:.1f}%")
    print(f"  Runtime Overhead: {time_diff:+.1f}%")
    
    return {
        "mission": name,
        "std_converged": res_std.converged,
        "std_dv": res_std.best_dv_total if res_std.converged else float('inf'),
        "std_nodes": std_nodes,
        "std_time": std_time,
        "std_paths": len(res_std.all_paths),
        "gnn_converged": res_gnn.converged,
        "gnn_dv": res_gnn.best_dv_total if res_gnn.converged else float('inf'),
        "gnn_nodes": gnn_nodes,
        "gnn_time": gnn_time,
        "gnn_paths": len(res_gnn.all_paths),
        "node_reduction": node_reduction,
        "time_overhead": time_diff
    }

def main():
    kernel = PhysicsKernel().load()
    
    # 1. Earth -> Mars
    dsl_em = parse_mission_file("data/benchmarks/earth_mars_2031.yaml")
    mission_em = compile_mission(dsl_em, kernel.ephemeris)
    res_em = run_mission_benchmark(
        name="Earth -> Mars",
        mission=mission_em,
        kernel=kernel,
        max_depth=2,
        n_iterations=150,
        dv_budget=15.0,
        flyby_candidates=[],
        seed=42
    )

    # 2. Earth -> Venus -> Mars
    # Modify Earth-Mars mission to allow Venus flybys and increase max depth
    mission_evm = compile_mission(dsl_em, kernel.ephemeris)
    mission_evm.tof_max_seconds = 400.0 * 86400.0 # Allow longer duration for Venus detour
    res_evm = run_mission_benchmark(
        name="Earth -> Venus -> Mars",
        mission=mission_evm,
        kernel=kernel,
        max_depth=3,
        n_iterations=200,
        dv_budget=20.0,
        flyby_candidates=["VENUS"],
        seed=42
    )

    # 3. Earth -> Jupiter
    # Compile a custom mission to Jupiter
    mission_ej = compile_mission(dsl_em, kernel.ephemeris)
    mission_ej.destination_body = CelestialBody.JUPITER
    mission_ej.tof_min_seconds = 300.0 * 86400.0
    mission_ej.tof_max_seconds = 900.0 * 86400.0
    res_ej = run_mission_benchmark(
        name="Earth -> Jupiter",
        mission=mission_ej,
        kernel=kernel,
        max_depth=3,
        n_iterations=200,
        dv_budget=30.0,
        flyby_candidates=["VENUS", "EARTH", "MARS"],
        seed=42
    )
    
    # Print Summary Table
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"{'Mission':<25} | {'Std Nodes':<10} | {'GNN Nodes':<10} | {'Reduction':<10} | {'Std Time':<10} | {'GNN Time':<10} | {'Overhead':<10}")
    print("-"*80)
    for r in [res_em, res_evm, res_ej]:
        print(f"{r['mission']:<25} | {r['std_nodes']:<10} | {r['gnn_nodes']:<10} | {r['node_reduction']:>8.1f}% | {r['std_time']:>8.3f}s | {r['gnn_time']:>8.3f}s | {r['time_overhead']:>+8.1f}%")
    print("="*80)

if __name__ == "__main__":
    main()
