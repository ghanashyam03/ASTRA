# Prompt 26 Validation Audit

## 1. Curtis Example 5.2 Results

Computed v1:
```json
[-5.992495020058082, 1.9253667141903978, 3.245638050488974]
```

Published Curtis v1:
```json
[-5.9925, 1.9254, 3.2456]
```

Component-wise v1 error:
```json
[4.9799419175045045e-06, -3.32858096021571e-05, 3.805048897387309e-05]
```

Computed v2:
```json
[-3.312458502994096, -4.196619007811479, -0.38528905983617645]
```

Published Curtis v2:
```json
[-3.3125, -4.1966, -0.3853]
```

Component-wise v2 error:
```json
[4.149700590394545e-05, -1.900781147856634e-05, 1.094016382352514e-05]
```

Reconstruction position error after propagation:
```text
7.817802109581074e-07 km
```

Branch:
```text
short-way/prograde
transfer_angle_deg = 100.29252420729621
cross_z = 158500000.0
```

ASTRA's Lambert solution uses the same short-way/prograde branch as the Curtis reference.

## 2. Lambert Solver Validation Summary

Curtis benchmark pytest output:
```text
tests/benchmark/test_curtis_lambert.py::test_curtis_example_5_2_v1 PASSED
tests/benchmark/test_curtis_lambert.py::test_curtis_example_5_2_reconstruction PASSED
tests/benchmark/test_curtis_lambert.py::test_lambert_hohmann_earth_mars PASSED

3 passed in 0.85s
```

Total Curtis tests passed/failed:
```text
passed = 3
failed = 0
```

Maximum observed Curtis reference velocity error:
```text
4.149700590394545e-05 km/s
```

Final verdict:
```text
Lambert correctness passes the Curtis Example 5.2 analytical validation.
```

## 3. Regression Baseline Contents

Generated `data/benchmarks/regression_baseline.json`:
```json
{
  "earth_mars_2031": {
    "best_dv_km_s": 5.341704,
    "best_tof_days": 209.967,
    "pareto_size": 218,
    "n_evaluations": 2000,
    "n_feasible": 812,
    "hypervolume_indicator": 237.3008
  }
}
```

Explicit values:
```text
best_dv_km_s = 5.341704
best_tof_days = 209.967
pareto_size = 218
hypervolume_indicator = 237.3008
n_evaluations = 2000
n_feasible = 812
```

## 4. Regression Lock Verification

Final `test_regression_lock.py` output:
```text
============================= test session starts =============================
platform win32 -- Python 3.12.5, pytest-9.0.3, pluggy-1.6.0
rootdir: C:\Users\ghana\OneDrive\Desktop\ASTRA\astra
configfile: pyproject.toml
collecting ... collected 2 items

tests/benchmark/test_regression_lock.py::test_earth_mars_2031_regression
Regression OK: 5.3417 km/s (baseline: 5.3417)
PASSED
tests/benchmark/test_regression_lock.py::test_pareto_front_size_regression
Pareto size: 218
PASSED

============================== 2 passed in 7.42s ==============================
RUNTIME_SECONDS=9.456
```

Current values vs baseline:
```json
{
  "baseline": {
    "best_dv_km_s": 5.341704,
    "best_tof_days": 209.967,
    "pareto_size": 218,
    "n_evaluations": 2000,
    "n_feasible": 812,
    "hypervolume_indicator": 237.3008
  },
  "current": {
    "best_dv_km_s": 5.341704066052435,
    "best_tof_days": 209.96749054046555,
    "pareto_size": 218,
    "n_evaluations": 2000,
    "n_feasible": 812,
    "hypervolume_indicator": 237.30082217114722
  },
  "percentage_difference": {
    "best_dv_km_s": 1.2365423960777195e-06,
    "best_tof_days": 0.00023362741075247544,
    "pareto_size": 0.0,
    "n_evaluations": 0.0,
    "n_feasible": 0.0,
    "hypervolume_indicator": 9.343056242510681e-06
  }
}
```

Implementation note:
```text
The prompt's regression test budget used 1000 trials while update_baseline.py
used 2000 trials. That made Pareto size compare 89 current solutions against
218 baseline solutions, failing by construction. The regression test was aligned
to the baseline budget: n_trials=2000, time_limit=120.0. No benchmark tolerance
was changed.
```

## 5. Benchmark Runtime Summary

Curtis benchmark runtime:
```text
pytest runtime = 0.85s
wall runtime = 2.409s
```

Baseline generation runtime:
```text
RUNTIME_SECONDS = 7.139
```

Regression test runtime:
```text
pytest runtime = 7.42s
wall runtime = 9.456s
```

Additional verification:
```text
uv run pytest tests/ -q
122 passed in 30.98s

uv run ruff check tests/benchmark/
All checks passed!

uv run mypy tests/benchmark/test_curtis_lambert.py
Success: no issues found in 1 source file
```

Prerequisite checks:
```text
uv run pytest tests/ -q: 116 passed before changes, 122 passed after changes
from astra.optimization.pareto import compute_pareto_quality: OK
Earth-Mars optimize_mission_bayesian convergence:
  converged = True
  best_dv = 5.348694735895096
  best_tof = 213.3390024155461
  pareto_size = 89
  n_evaluations = 1000
  n_feasible = 380
from astra.physics.differentiable import JAX_AVAILABLE: skipped per user request
```

## 6. Merge Assessment

B) MERGE WITH CHANGES

Reason:
```text
The validation suite passes, the analytical Lambert result matches Curtis, and
the regression lock now compares like-for-like budgets. The only caveat is that
the Earth-Venus-Mars benchmark validates the currently implemented multi-body
DSL and continuous Venus flyby periapsis physics, not a full continuous
multi-leg optimizer. The current MCTS planner is a coarse fixed-TOF phase
planner and did not reliably satisfy the requested Venus-assisted <9 km/s
trajectory threshold.
```
