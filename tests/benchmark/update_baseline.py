"""Script to update regression baseline.

Run manually after verified improvement.
Usage: uv run python tests/benchmark/update_baseline.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def update_baseline() -> dict[str, Any] | None:
    from astra.dsl.compiler import compile_mission
    from astra.dsl.parser import parse_mission_file
    from astra.optimization.engine import optimize_mission_bayesian
    from astra.optimization.pareto import compute_pareto_quality
    from astra.physics.kernel import PhysicsKernel

    kernel = PhysicsKernel().load()
    dsl = parse_mission_file("data/benchmarks/earth_mars_2031.yaml")
    mission = compile_mission(dsl, kernel.ephemeris)
    result = optimize_mission_bayesian(mission, kernel, n_trials=2000, time_limit=120.0, seed=42)

    if not result.converged or result.best_trajectory is None:
        print("ERROR: Optimization did not converge. Baseline not updated.")
        return None

    best = result.best_trajectory
    quality = compute_pareto_quality(result.pareto_front) if result.pareto_front else None

    baseline = {
        "earth_mars_2031": {
            "best_dv_km_s": round(best.delta_v_total, 6),
            "best_tof_days": round(best.duration_days, 3),
            "pareto_size": len(result.pareto_front),
            "n_evaluations": result.n_evaluations,
            "n_feasible": result.n_feasible,
            "hypervolume_indicator": round(quality.hypervolume_indicator, 4)
            if quality
            else 0.0,
        }
    }

    path = Path("data/benchmarks/regression_baseline.json")
    path.write_text(json.dumps(baseline, indent=2) + "\n")
    print("Baseline updated:")
    print(json.dumps(baseline, indent=2))
    return baseline


if __name__ == "__main__":
    update_baseline()
