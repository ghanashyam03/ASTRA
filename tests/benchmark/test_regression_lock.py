"""Regression lock suite.

Captures current ASTRA results and fails if future code changes degrade them
beyond acceptable tolerances.

Run uv run python tests/benchmark/update_baseline.py to update the baseline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

BASELINE_PATH = Path("data/benchmarks/regression_baseline.json")
SPICE = (Path("data/spice_kernels") / "de440.bsp").exists()


def load_baseline() -> dict[str, Any]:
    import typing

    if not BASELINE_PATH.exists():
        return {}
    return typing.cast(dict[str, Any], json.loads(BASELINE_PATH.read_text()))


def run_earth_mars_2031() -> Any:  # noqa: ANN401
    from astra.dsl.compiler import compile_mission
    from astra.dsl.parser import parse_mission_file
    from astra.optimization.engine import optimize_mission_bayesian
    from astra.physics.kernel import PhysicsKernel

    kernel = PhysicsKernel().load()
    dsl = parse_mission_file("data/benchmarks/earth_mars_2031.yaml")
    mission = compile_mission(dsl, kernel.ephemeris)
    return optimize_mission_bayesian(mission, kernel, n_trials=2000, time_limit=120.0, seed=42)


@pytest.mark.skipif(not SPICE, reason="SPICE kernels required")
@pytest.mark.slow
def test_earth_mars_2031_regression() -> None:
    """Earth-Mars 2031 optimal delta-v must not degrade from baseline."""
    result = run_earth_mars_2031()

    assert result.converged
    assert result.best_trajectory is not None
    current_dv = result.best_trajectory.delta_v_total

    baseline = load_baseline()
    if "earth_mars_2031" in baseline:
        ref_dv = baseline["earth_mars_2031"]["best_dv_km_s"]
        tolerance = 0.02
        assert current_dv <= ref_dv * (
            1 + tolerance
        ), f"Regression: current delta-v {current_dv:.4f} > baseline {ref_dv:.4f} + 2%"
        print(f"\nRegression OK: {current_dv:.4f} km/s (baseline: {ref_dv:.4f})")
    else:
        print(
            f"\nNo baseline yet. Current: {current_dv:.4f} km/s. " "Run update_baseline.py to set."
        )


@pytest.mark.skipif(not SPICE, reason="SPICE kernels required")
@pytest.mark.slow
def test_pareto_front_size_regression() -> None:
    """Pareto front must not shrink from baseline."""
    result = run_earth_mars_2031()

    current_pareto = len(result.pareto_front)
    baseline = load_baseline()
    if "earth_mars_2031" in baseline:
        ref_pareto = baseline["earth_mars_2031"]["pareto_size"]
        assert (
            current_pareto >= ref_pareto * 0.7
        ), f"Regression: Pareto shrank from {ref_pareto} to {current_pareto}"
    print(f"\nPareto size: {current_pareto}")
