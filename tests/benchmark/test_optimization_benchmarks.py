import json
import time
from pathlib import Path

import numpy as np
import pytest

from astra.dsl.compiler import CompiledMission, compile_mission
from astra.dsl.parser import parse_mission_string
from astra.neural.pinn import LambertPINNEnsemble
from astra.optimization.engine import optimize_mission_mcts
from astra.physics.kernel import PhysicsKernel

SPICE_AVAILABLE = (Path("data/spice_kernels") / "de440.bsp").exists()

SCENARIO_EARTH_MARS = """
version: "1.0"
mission_id: "earth_mars_benchmark"
spacecraft:
  name: "Ares-1"
  dry_mass_kg: 1800.0
  fuel_mass_kg: 2400.0
  propulsion:
    type: chemical
    isp_seconds: 450.0
    thrust_newtons: 22000.0
trajectory:
  origin:
    body: Earth
    orbit:
      type: circular
      altitude_km: 200.0
  destination:
    body: Mars
    orbit:
      type: circular
      altitude_km: 300.0
launch_window:
  start: "2031-01-01T00:00:00Z"
  end: "2033-01-01T00:00:00Z"
  resolution_days: 10
  tof_min_days: 120
  tof_max_days: 280
constraints:
  - type: max_delta_v
    value_km_s: 20.0
    hard: true
objectives:
  - metric: delta_v_total
    direction: minimize
    weight: 1.0
physics:
  models: [patched_conics]
  ephemeris: DE440
optimization:
  strategy: bayesian
  budget:
    max_evaluations: 100
  seed: 42
"""

SCENARIO_EARTH_VENUS_MARS = """
version: "1.0"
mission_id: "earth_venus_mars_benchmark"
spacecraft:
  name: "Ares-1"
  dry_mass_kg: 1800.0
  fuel_mass_kg: 2400.0
  propulsion:
    type: chemical
    isp_seconds: 450.0
    thrust_newtons: 22000.0
trajectory:
  origin:
    body: Earth
    orbit:
      type: circular
      altitude_km: 200.0
  destination:
    body: Mars
    orbit:
      type: circular
      altitude_km: 300.0
launch_window:
  start: "2028-01-01T00:00:00Z"
  end: "2030-01-01T00:00:00Z"
  resolution_days: 10
  tof_min_days: 100
  tof_max_days: 400
constraints:
  - type: max_delta_v
    value_km_s: 25.0
    hard: true
objectives:
  - metric: delta_v_total
    direction: minimize
    weight: 1.0
physics:
  models: [patched_conics]
  ephemeris: DE440
optimization:
  strategy: bayesian
  budget:
    max_evaluations: 100
  seed: 42
"""

SCENARIO_EARTH_JUPITER = """
version: "1.0"
mission_id: "earth_jupiter_benchmark"
spacecraft:
  name: "Ares-1"
  dry_mass_kg: 1800.0
  fuel_mass_kg: 2400.0
  propulsion:
    type: chemical
    isp_seconds: 450.0
    thrust_newtons: 22000.0
trajectory:
  origin:
    body: Earth
    orbit:
      type: circular
      altitude_km: 200.0
  destination:
    body: Jupiter
    orbit:
      type: circular
      altitude_km: 3000.0
launch_window:
  start: "2030-01-01T00:00:00Z"
  end: "2032-01-01T00:00:00Z"
  resolution_days: 10
  tof_min_days: 500
  tof_max_days: 1000
constraints:
  - type: max_delta_v
    value_km_s: 30.0
    hard: true
objectives:
  - metric: delta_v_total
    direction: minimize
    weight: 1.0
physics:
  models: [patched_conics]
  ephemeris: DE440
optimization:
  strategy: bayesian
  budget:
    max_evaluations: 100
  seed: 42
"""


def generate_training_data(
    kernel: PhysicsKernel, mission: CompiledMission, n_samples: int
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rng = np.random.default_rng(42)
    dep_epochs = rng.uniform(mission.departure_epoch_start, mission.departure_epoch_end, n_samples)
    tof_seconds = rng.uniform(mission.tof_min_seconds, mission.tof_max_seconds, n_samples)

    X = []
    y_dv = []
    r1_ns = []
    r2_ns = []
    tof_list = []
    r1_vs = []
    r2_vs = []
    vp_deps = []
    vp_arrs = []

    from astra.explainability.window_rationale import compute_synodic_period
    from astra.neural.features import build_geometric_features
    from astra.physics.lambert import find_best_transfer

    syn_days = compute_synodic_period(mission.origin_body, mission.destination_body)
    synodic_period_s = syn_days * 86400.0 if syn_days != float("inf") else 0.0

    for i in range(n_samples):
        dep = dep_epochs[i]
        tof = tof_seconds[i]
        try:
            r1_state = kernel.get_body_state(mission.origin_body, dep)
            r2_state = kernel.get_body_state(mission.destination_body, dep + tof)

            feat = build_geometric_features(
                dep_epoch=dep,
                tof_seconds=tof,
                r1_km=r1_state.position,
                v1_km_s=r1_state.velocity,
                r2_km=r2_state.position,
                dep_epoch_min=mission.departure_epoch_start,
                dep_epoch_max=mission.departure_epoch_end,
                tof_min=mission.tof_min_seconds,
                tof_max=mission.tof_max_seconds,
                synodic_period_s=synodic_period_s,
            )

            sol = find_best_transfer(
                r1=r1_state.position,
                v1_body=r1_state.velocity,
                r2=r2_state.position,
                v2_body=r2_state.velocity,
                tof=tof,
                mu=1.32712440018e11,
                max_revs=0,
            )

            X.append(feat)
            y_dv.append(sol.delta_v)
            r1_ns.append(np.linalg.norm(r1_state.position))
            r2_ns.append(np.linalg.norm(r2_state.position))
            tof_list.append(tof)
            r1_vs.append(r1_state.position)
            r2_vs.append(r2_state.position)
            vp_deps.append(r1_state.velocity)
            vp_arrs.append(r2_state.velocity)
        except Exception:
            pass

    return (
        np.array(X, dtype=np.float32),
        np.array(y_dv, dtype=np.float32),
        np.array(r1_ns, dtype=np.float32),
        np.array(r2_ns, dtype=np.float32),
        np.array(tof_list, dtype=np.float32),
        np.array(r1_vs, dtype=np.float32),
        np.array(r2_vs, dtype=np.float32),
        np.array(vp_deps, dtype=np.float32),
        np.array(vp_arrs, dtype=np.float32),
    )


@pytest.mark.skipif(not SPICE_AVAILABLE, reason="SPICE kernels required")
def test_optimization_benchmarks() -> None:
    """Compare Standard MCTS vs Surrogate-Guided MCTS and output a comparison matrix."""
    kernel = PhysicsKernel().load()

    scenarios = {
        "Earth_to_Mars": SCENARIO_EARTH_MARS,
        "Earth_to_Venus_to_Mars": SCENARIO_EARTH_VENUS_MARS,
        "Earth_to_Jupiter": SCENARIO_EARTH_JUPITER,
    }

    mission_flyby_candidates = {
        "Earth_to_Mars": [],
        "Earth_to_Venus_to_Mars": ["VENUS"],
        "Earth_to_Jupiter": [],
    }

    from typing import Any

    results: dict[str, dict[str, Any]] = {}

    for name, spec in scenarios.items():
        print(f"\nRunning Benchmark Scenario: {name}...")
        dsl = parse_mission_string(spec)
        mission = compile_mission(dsl, kernel.ephemeris)

        # Generate dataset and train the surrogate ensemble
        ret = generate_training_data(kernel, mission, 100)
        X, y_dv, r1_ns, r2_ns, tofs, r1_vs, r2_vs, vp_deps, vp_arrs = ret

        surrogate = LambertPINNEnsemble(hidden_dims=[32, 16], ensemble_size=5)
        surrogate.train_on_dataset(
            x_data=X,
            v_targets=y_dv,
            r1_norms=r1_ns,
            r2_norms=r2_ns,
            tof_seconds=tofs,
            epochs=10,
            batch_size=32,
            r1_vecs=r1_vs,
            r2_vecs=r2_vs,
            v_planet_depart=vp_deps,
            v_planet_arrive=vp_arrs,
        )

        preds = surrogate.predict_batch(X, v_planet_depart=vp_deps, v_planet_arrive=vp_arrs)
        mean_error = float(np.mean(np.abs(preds - y_dv)))

        # Standard MCTS
        t0 = time.perf_counter()
        res_std = optimize_mission_mcts(
            mission=mission,
            kernel=kernel,
            flyby_candidates=mission_flyby_candidates[name],
            n_iterations=50,
            dv_budget=20.0,
            seed=42,
            surrogate=None,
            uncertainty_weight=0.0,
        )
        dt_std = time.perf_counter() - t0

        dv_std = res_std.best_trajectory.delta_v_total if res_std.best_trajectory else float("inf")
        val_success_std = 0.0
        if res_std.best_trajectory:
            val_res = kernel.validate_trajectory(res_std.best_trajectory)
            val_success_std = 1.0 if val_res.is_valid else 0.0

        # Surrogate-Guided MCTS
        t0 = time.perf_counter()
        res_surr = optimize_mission_mcts(
            mission=mission,
            kernel=kernel,
            flyby_candidates=mission_flyby_candidates[name],
            n_iterations=50,
            dv_budget=20.0,
            seed=42,
            surrogate=surrogate,
            uncertainty_weight=0.5,
        )
        dt_surr = time.perf_counter() - t0

        dv_surr = (
            res_surr.best_trajectory.delta_v_total if res_surr.best_trajectory else float("inf")
        )
        val_success_surr = 0.0
        if res_surr.best_trajectory:
            val_res = kernel.validate_trajectory(res_surr.best_trajectory)
            val_success_surr = 1.0 if val_res.is_valid else 0.0

        results[name] = {
            "standard_mcts": {
                "runtime_s": dt_std,
                "delta_v_km_s": dv_std,
                "nodes_explored": res_std.n_evaluations,
                "validation_success_rate": val_success_std,
            },
            "surrogate_guided_mcts": {
                "runtime_s": dt_surr,
                "delta_v_km_s": dv_surr,
                "nodes_explored": res_surr.n_evaluations,
                "validation_success_rate": val_success_surr,
            },
            "surrogate_error": mean_error,
        }

    # Store results to JSON file
    out_path = Path(__file__).parent / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    # Print markdown table
    print("\nBenchmark Results Summary:")
    print(
        "| Scenario | Mode | Runtime (s) | Best Delta-v (km/s) |"
        " Nodes | Validation Rate | Surrogate Error |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for name, r in results.items():
        std = r["standard_mcts"]
        surr = r["surrogate_guided_mcts"]
        print(
            f"| {name} | Standard | {std['runtime_s']:.3f} | {std['delta_v_km_s']:.3f} |"
            f" {std['nodes_explored']} | {std['validation_success_rate']:.1f} | N/A |"
        )
        print(
            f"| {name} | Surrogate | {surr['runtime_s']:.3f} | {surr['delta_v_km_s']:.3f} |"
            f" {surr['nodes_explored']} | {surr['validation_success_rate']:.1f} |"
            f" {r['surrogate_error']:.3f} |"
        )

    # Assert basic run execution
    assert len(results) == 3
