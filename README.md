# ASTRA: Autonomous Space Trajectory Reasoning Architecture

ASTRA is a premium, physics-constrained orbital trajectory optimization and mission analysis platform designed for advanced astrodynamics research.

---

## Key Core Features

### 1. Hardened Astrodynamics Core (`astra.physics`)
*   **Precision Orbital Primitives**: Complete double-precision (`np.float64`) Cartesian orbital states with built-in specific mechanical energy, angular momentum, eccentricity vector, and orbital period computation.
*   **Barycenter-Safe SPICE Ephemeris Engine**: Differentiates precisely between planetary system barycenters and body centers, preventing silent failures when using standard JPL SPK data (`de440.bsp`).
*   **Izzo Lambert Solver**: Universal variables algorithm for two-point boundary value problems. Hardened against collinear transfer geometries and time-of-flight bounds, raising deterministic exceptions (`LambertSingularityError` and `LambertConvergenceError`).
*   **Pluggable Integrator Interface**: Supports numerical propagation using the Runge-Kutta 4(5) solver (`RK45Integrator`) with modular support for symplectic or custom adaptive integration solvers.
*   **Physical Collision Detection**: Incorporates physical planetary equatorial radii boundaries (`PHYSICAL_RADIUS`) inside propagation loops to raise terminal collision exceptions synchronously.

### 2. ASTRA Mission DSL (`astra.dsl`)
*   **Strict Pydantic v2 Schema**: High-level validation models verifying dry mass, fuel mass, Isp boundaries, time-of-flight spans, launch windows, constraints, and objective weights.
*   **YAML/JSON Parsers**: Robust parser utility matching standard YAML/JSON specs directly to schemas.
*   **Mission Compiler**: Compiles high-level user specifications into domain-level strongly typed constructs (`Spacecraft`, `PropulsionSystem`, `CelestialBody`) and converts UTC epochs to J2000 barycentric dynamical time (TDB) seconds automatically using naive date string ISO-format parameters.

### 3. Trajectory Optimization Engine (`astra.optimization`)
*   **Search Space boundaries (`SearchSpace`)**: Tracks and structures parameters for J2000 departure epochs and time-of-flight seconds.
*   **Porkchop Grid Computation (`compute_porkchop`)**: Calculates precise grid points of departure dates and time of flight, querying SPICE targets and performing Lambert solutions to map orbital feasibility.
*   **Multi-Objective TPE Optimizer (`optimize_mission_bayesian`)**: Employs Optuna TPE and NSGA-II multi-objective samplers to evaluate optimal trade-offs between flight duration and total launch/capture $\Delta v$, returning best feasible solutions and complete Pareto-front structures.
*   **Maneuvers & Trajectories (`astra.state.trajectory`)**: Houses complete representations of impulsive orbital maneuvers (`Maneuver`) and multi-impulse orbital transfers (`Trajectory`).

### 4. Explainability Engine (`astra.explainability`)
*   **Delta-V Budget Decompositions (`decompose_delta_v`)**: Breaks down launch and arrival maneuver magnitudes, percentages, and epoch points, applying a standard 3% navigation margin.
*   **Launch Window Rationales (`build_window_rationale`)**: Analyzes planetary phase angles at departure, synodic orbits, and departure C3 energies to generate computed rationales explaining optimal departure dates.
*   **Constraint Compliance Analysis (`analyze_constraints`)**: Identifies margins and boundary-binding limits (within 5% of constraint margins) for total transfer $\Delta v$ and flight duration.
*   **Pareto Tradeoff Interpretation (`analyze_pareto`)**: Computes optimal slope parameters (e.g. extra $\Delta v$ required per day saved) to interpret duration vs propellant trade-offs quantitatively across Pareto fronts.

---

## Repository Structure

```text
astra/
├── data/
│   └── benchmarks/
│       └── earth_mars_2031.yaml      # Reference Mission Specification
├── docs/
│   └── physics_limitations.md        # Comprehensive Physics Core Limitations
├── src/
│   └── astra/
│       ├── dsl/                      # Mission DSL, Parser, Schema, Compiler
│       ├── explainability/           # Delta-V breakdowns, window rationales, constraints
│       ├── optimization/             # Porkchop computation, Bayesian optimizer, Search Space
│       ├── physics/                  # Lambert Solver, Ephemeris Engine, Propagator
│       └── state/                    # Spacecraft, Trajectory, and Orbital Primitives
└── tests/
    ├── integration/                  # End-to-End Optimization integration tests
    └── unit/
        ├── dsl/                      # Mission DSL Unit Tests
        ├── explainability/           # Explainability Unit Tests
        ├── physics/                  # Physics Core Unit Tests
        └── state/                    # Primitive Primitives Unit Tests
```

---

## Quickstart & Commands

### 1. Developer Setup
Standard Python environments are managed via the high-performance `uv` package manager:
```powershell
# Run the entire automated test suite (26 unit and integration tests)
uv run pytest -v

# Check code styling and lint compliance
uv run ruff check src tests

# Verify strict type annotations
uv run mypy src tests
```

### 2. Parse & Compile a Mission
```python
from astra.dsl import parse_mission_file, compile_mission
from astra.physics import EphemerisEngine
from pathlib import Path

# Load and parse standard YAML mission
dsl_mission = parse_mission_file("data/benchmarks/earth_mars_2031.yaml")

# Compile mission into strongly typed objects
compiled = compile_mission(dsl_mission)

print(f"Mission: {compiled.mission_id}")
print(f"Spacecraft Name: {compiled.spacecraft.name}")
print(f"Target Origin: {compiled.origin_body.value}")
```

### 3. Compute Porkchop Plot Grid
```python
import numpy as np
from astra.physics import PhysicsKernel
from astra.dsl import parse_mission_file, compile_mission
from astra.optimization import compute_porkchop

# Load SPICE and parse reference Earth-Mars 2031 mission
kernel = PhysicsKernel().load()
dsl = parse_mission_file("data/benchmarks/earth_mars_2031.yaml")
mission = compile_mission(dsl, kernel.ephemeris)

# Compute a 50x50 porkchop grid of launch opportunities
dep_epochs, tof_days, dv_grid = compute_porkchop(mission, kernel, n_dep=50, n_tof=50)
print(f"Computed grid shape: {dv_grid.shape}")
print(f"Minimum solved Δv: {np.nanmin(dv_grid):.3f} km/s")
```

### 4. Bayesian Multi-Objective Optimization
```python
from astra.physics import PhysicsKernel
from astra.dsl import parse_mission_file, compile_mission
from astra.optimization import optimize_mission_bayesian

# Load SPICE and compile mission
kernel = PhysicsKernel().load()
dsl = parse_mission_file("data/benchmarks/earth_mars_2031.yaml")
mission = compile_mission(dsl, kernel.ephemeris)

# Optimize trajectory trade-offs (TPE / NSGA-II)
result = optimize_mission_bayesian(mission, kernel, n_trials=500, time_limit=60.0)

if result.converged and result.best_trajectory:
    print(f"Optimizer found a feasible transfer!")
    print(f"Best Delta-V Total: {result.best_trajectory.delta_v_total:.3f} km/s")
    print(f"Transfer Duration: {result.best_trajectory.duration_days:.1f} days")
    print(f"Pareto Front Size: {len(result.pareto_front)}")
```

### 5. Trajectory Rationale & Explanations
```python
from astra.physics import PhysicsKernel
from astra.dsl import parse_mission_file, compile_mission
from astra.optimization import optimize_mission_bayesian
from astra.explainability import explain

kernel = PhysicsKernel().load()
dsl = parse_mission_file("data/benchmarks/earth_mars_2031.yaml")
mission = compile_mission(dsl, kernel.ephemeris)
result = optimize_mission_bayesian(mission, kernel, n_trials=500, time_limit=60.0)

if result.best_trajectory:
    # Generate full explainability trace from actual computations
    trace = explain(
        trajectory=result.best_trajectory,
        mission=mission,
        pareto_front=result.pareto_front,
        ephemeris=kernel.ephemeris
    )
    
    # Export explanations to dictionary
    explanation_data = trace.to_dict()
    
    print(f"Window Phase Angle: {explanation_data['window_rationale']['planet_phase_angle_deg']}°")
    print(f"Tradeoff Saved Day: {explanation_data['pareto_analysis']['avg_tradeoff_km_s_per_day']} km/s")
    for point in explanation_data['window_rationale']['rationale']:
        print(f"  - {point}")
```

---

## FastAPI API Layer & Trajectory Storage (`astra.api` and `astra.data`)

ASTRA exposes a standard, high-performance FastAPI web application layer to query celestial bodies, calculate porkchop opportunity grids, serialize 3D trajectory rendering-ready data structures, and persist optimization runs.

### Persistence & Databases
*   **DuckDB Trajectory Persistent Storage**:
    *   **Development / Production Default**: Optimization results and computed trajectory details automatically persist to `data/astra.duckdb` locally.
    *   **Testing Mode**: When running integration tests, ASTRA overrides the store to run strictly inside a transient, fast in-memory DuckDB environment (`:memory:`).
*   **In-Memory Job Tracking (Non-persistent)**:
    *   Long-running optimization requests (`POST /v1/missions/optimize`) are queued to FastAPI `BackgroundTasks` as an asynchronous job.
    *   **Crucial Limitation**: The active optimization job state is held strictly in an in-memory dictionary. Active/historical job records and status logs **do not persist** across server restarts.

### API Limitations
*   **Single-rev Transfers Only**: The under-the-hood Lambert solvers and patched-conics propagators are limited to single-revolution two-body orbital transfers.
*   **Single-Worker/Thread Handoff**: The global, in-memory SPICE physics kernel loads dynamically at startup and behaves as a singleton worker resource.
*   **Degraded Mode Support**: If local SPICE ephemeris files are missing during server startup, ASTRA boots gracefully into a degraded mode. Under degraded mode, standard health checks pass, but precise body state lookups and optimization requests will return 503 Service Unavailable or 400 Bad Request error codes.

### Current Supported Endpoints
*   `GET /v1/health` - Inspect ASTRA service status and see if SPICE ephemeris engine is loaded.
*   `POST /v1/missions/optimize` - Queue an asynchronous Bayesian trajectory optimization job using a YAML mission specification string.
*   `GET /v1/missions/{job_id}/status` - Poll optimization job status (`queued`, `running`, `complete`, `failed`).
*   `GET /v1/missions/{job_id}/result` - Retrieve the complete optimization result dictionary once completed.
*   `GET /v1/trajectories/{trajectory_id}` - Fetch a stored trajectory record from DuckDB.
*   `GET /v1/trajectories/{trajectory_id}/explanation` - Retrieve the explainability trace details associated with a trajectory.
*   `GET /v1/bodies/{body_name}/state` - Query double-precision state vector positions/velocities relative to the Sun.
*   `POST /v1/windows/porkchop` - Compute a detailed launch/arrival time-of-flight porkchop energy grid.

### Current Visualization Capabilities (`astra.visualization`)
*   **3D Trajectory Rendering Data**: Integrates astronomical-unit scaling to serialize spacecraft coordinates, maneuver epoch vectors, and planetary tracks into beautiful, browser-ready structures (`build_render_data()`).
*   **Plotly-Ready Porkchops**: Converts raw grid arrays into serialized Plotly-ready contour maps (`build_porkchop_plot()`), replacing NaN values with JSON-serializable `None` values and identifying the global energy minimum.

