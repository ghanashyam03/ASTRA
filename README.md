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


---

## Ephemeris Cache & Replay Manifest Architecture

To support high-performance computations and rigorous scientific reproducibility, ASTRA incorporates an epoch-quantized ephemeris cache and deterministic replay manifest system.

### 1. Ephemeris Cache (`astra.data.cache`)
ASTRA's Bayesian optimization and porkchop grid calculations involve tens of thousands of coordinate and velocity lookups. Querying SPICE directly for every state lookup is computationally expensive.
* **Epoch-Quantized LRU Cache**: Employs an in-memory `OrderedDict`-based Least Recently Used (LRU) cache (`EphemerisCache`) to store planet state vectors (`position` and `velocity`) for a given target, observer, and frame.
* **Quantization Guard**: Epochs (J2000 seconds) are quantized to a configurable grid (default: 60-second resolution via `DEFAULT_QUANTIZATION_SECONDS`) to collapse close, numerically equivalent epochs into single cache entries without affecting physics accuracy.
* **Disk Persistence**: Supports local state preservation. The cache can be loaded from or serialized to a JSON file (`persist_path`) for cross-run reuse:
  ```python
  from pathlib import Path
  from astra.data.cache import EphemerisCache
  from astra.physics import PhysicsKernel

  # Initialize PhysicsKernel with a persistent disk cache
  cache = EphemerisCache(max_entries=50_000, persist_path=Path("data/cache.json"))
  kernel = PhysicsKernel(cache=cache)
  ```
* **Performance Reporting**: The cache tracks hits, misses, and evictions dynamically. You can inspect hit-rate statistics at any time:
  ```python
  stats = cache.stats.to_dict()
  print(f"Cache Hit Rate: {stats['hit_rate_pct']}% (Hits: {stats['hits']}, Misses: {stats['misses']})")
  ```

### 2. Deterministic Replay & Reproducibility (`astra.data.replay`)
ASTRA provides a robust scientific reproducibility workflow using JSON-based replay manifests.
* **Metadata Capture**: A `ReplayManifest` captures the exact system configurations, software versions (`astra`, `python`), SPICE kernel SHA-256 checksums, DSL mission YAML specification text, random seed, and optimization search budget.
* **Determinism**: Allows exact reproduction of optimization traces and Pareto frontiers.

### 3. Replay CLI Workflow
You can save and replay optimization runs directly from the command line:

* **Save a Manifest after Optimization**:
  ```powershell
  uv run astra optimize data/benchmarks/earth_mars_2031.yaml \
      --trials 500 --time-limit 60 --save-manifest data/benchmarks/test_manifest.json
  ```

* **Deterministic Replay from a Manifest**:
  ```powershell
  uv run astra optimize --replay data/benchmarks/test_manifest.json
  ```

---

## Multi-Revolution Lambert Support (`astra.physics`)

To support long-duration interplanetary missions, ASTRA incorporates complete multi-revolution Lambert targeting using Dario Izzo's universal variable method. This enables the calculation of transfer conics where the spacecraft makes $n \ge 1$ complete revolutions around the central body before arriving at its destination.

### 1. The Physics & Branch Definition
For any target transfer duration longer than the minimum possible multi-revolution time ($T > T_{min}(n)$), there exist exactly **two solutions** per revolution count $n \ge 1$:
*   **Low $\Delta v$ Branch (Lowpath)**: Corresponds to $x > x_{min}$ (larger universal variable $x$, closer to $1.0$). This represents orbits with smaller eccentricities, smaller energy deviations, and generally lower fuel costs.
*   **High $\Delta v$ Branch (Highpath)**: Corresponds to $x < x_{min}$ (smaller universal variable $x$, closer to $-1.0$). This represents highly eccentric orbits with focus above the chord, usually resulting in significantly higher fuel consumption.

### 2. Turning Point & Singularity Safeguards
Multi-revolution orbits are physically bounded by a minimum time of flight, below which no mathematical solution exists.
*   **Halley Turning Point Finder**: ASTRA resolves the turning point $x_{min}$ where $dT/dx = 0$ using a high-precision third-order Halley iterative search starting from a robust initial guess of $x_0 = -0.5$.
*   **Boundary Enforcement**: If the non-dimensional target time of flight $T$ is less than $T_{min}(n)$, the solver explicitly rejects the invalid solution space by throwing a `LambertSingularityError`.
*   **Convergence Safeguard**: During Householder root-finding iterations, a dampening guard prevents steps from crossing the $x_{min}$ turning point boundary. If a step would cross $x_{min}$, the algorithm dampens it to move only 90% of the way to the boundary, ensuring robust convergence to the correct branch.

### 3. Validation Methodology
Every multi-revolution capability is validated against rigorous manual and automated test criteria:
*   **Backwards Compatibility ($n=0$)**: $n=0$ (single-rev) runs through the identical validated baseline universal variables route and matches `lambert_izzo` bit-for-bit.
*   **Conic Propagation Verification**: Transfer orbits solved by `lambert_izzo_multirev` are numerically propagated over the target time of flight using double-precision two-body integration. The spacecraft must arrive back at the destination position within a strictly enforced tolerance of **< 500 km**.
*   **Reference Benchmarks**: Demonstrated on Earth-Mars transfer windows where long-duration launch opportunities successfully locate cheaper $1$-rev and $2$-rev conics (e.g. dropping Earth-Mars $\Delta v$ from $44.4$ km/s down to $14.29$ km/s over an 800-day window).

---

## Heliocentric Δv vs Mission Δv

ASTRA differentiates between **Heliocentric $\Delta v$** (which represents the deep-space velocity changes relative to the Sun) and **Mission $\Delta v$** (which incorporates planetary gravity wells via sphere-of-influence patching for realistic launch vehicle injection and orbital insertion).

### 1. Mathematical Formulations

*   **Heliocentric $\Delta v$**:
    Assumes maneuvers are performed in deep space far from planetary gravity wells.
    $$\Delta v_{\text{helio}} = \| \mathbf{v}_{\text{sc,dep}} - \mathbf{v}_{\text{body,dep}} \| + \| \mathbf{v}_{\text{body,arr}} - \mathbf{v}_{\text{sc,arr}} \| = v_{\infty,\text{dep}} + v_{\infty,\text{arr}}$$
    where $v_{\infty}$ is the hyperbolic excess speed at departure/arrival.

*   **Patched-Conics (Sphere of Influence) $\Delta v$**:
    Computes precise orbital maneuvers within the planetary gravity wells (Laplace SOI) using circular parking/capture orbits:
    *   **Departure Burn (Trans-Mars Injection, TMI)**: Accelerates from a circular parking orbit at altitude $h_{\text{park}}$:
        $$v_{\text{park}} = \sqrt{\frac{\mu_{\text{origin}}}{R_{\text{origin}} + h_{\text{park}}}}$$
        $$v_{\text{hyp,dep}} = \sqrt{v_{\infty,\text{dep}}^2 + \frac{2\mu_{\text{origin}}}{R_{\text{origin}} + h_{\text{park}}}}$$
        $$\Delta v_{\text{TMI}} = v_{\text{hyp,dep}} - v_{\text{park}}$$
    *   **Arrival Burn (Mars Orbit Insertion, MOI)**: Decelerates from a hyperbolic approach to a circular capture orbit at altitude $h_{\text{capture}}$:
        $$v_{\text{cap}} = \sqrt{\frac{\mu_{\text{dest}}}{R_{\text{dest}} + h_{\text{capture}}}}$$
        $$v_{\text{hyp,arr}} = \sqrt{v_{\infty,\text{arr}}^2 + \frac{2\mu_{\text{dest}}}{R_{\text{dest}} + h_{\text{capture}}}}$$
        $$\Delta v_{\text{MOI}} = v_{\text{hyp,arr}} - v_{\text{cap}}$$
    *   **Total Mission $\Delta v$**:
        $$\Delta v_{\text{total}} = \Delta v_{\text{TMI}} + \Delta v_{\text{MOI}}$$

### 2. The Oberth Effect & Mission Design Implications

The planetary gravity well provides a significant speed boost near the body's periapsis. Because mechanical energy change is proportional to speed ($\Delta E_k \approx v \Delta v$), performing burns deep within a gravity well (at high speed) yields a vastly larger orbital energy change than in deep heliocentric space.

As a result:
*   **High-Energy Transits**: The SOI-patched $\Delta v$ can be numerically **smaller** than the heliocentric $v_{\infty}$ sum. The Oberth effect at Earth and Mars dramatically reduces the propellant required to escape and capture.
*   **Hohmann Geometry**: For very low-energy transfers, the escape/capture gravity well overhead is positive, typically adding between $0.1$ and $2.0$ km/s depending on the target parking orbit altitudes.

### 3. Empirical Reference: Earth-Mars 2031 Transfer

Below is a comparative breakdown of the optimal 280-day Earth-Mars transfer from our benchmark suite:

| Parameter | Heliocentric Formulation | Patched-Conics (SOI) Formulation |
| :--- | :--- | :--- |
| **Departure Burn (TMI)** | $3.2372$ km/s (as $v_{\infty}$) | **$3.6904$ km/s** (from $200$ km LEO) |
| **Arrival Burn (MOI)** | $3.5471$ km/s (as $v_{\infty}$) | **$2.5761$ km/s** (to $300$ km Mars orbit) |
| **Total $\Delta v$** | $6.7843$ km/s | **$6.2665$ km/s** |
| **Departure $C_3$** | $10.4794$ $\text{km}^2/\text{s}^2$ | $10.4794$ $\text{km}^2/\text{s}^2$ |

