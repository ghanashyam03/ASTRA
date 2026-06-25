# ASTRA: Autonomous Space Trajectory Reasoning Architecture

ASTRA is a physics-constrained orbital trajectory optimization and mission analysis platform designed for advanced astrodynamics research.

---

## Key Core Features

### 1. Hardened Astrodynamics Core (`astra.physics`)
*   **Precision Orbital Primitives**: Complete double-precision (`np.float64`) Cartesian orbital states with built-in specific mechanical energy, angular momentum, eccentricity vector, and orbital period computation.
*   **Barycenter-Safe SPICE Ephemeris Engine**: Differentiates precisely between planetary system barycenters and body centers, preventing silent failures when using standard JPL SPK data (`de440.bsp`).
*   **Izzo Lambert Solver**: Universal variables algorithm for two-point boundary value problems. Hardened against collinear transfer geometries and time-of-flight bounds, raising deterministic exceptions (`LambertSingularityError` and `LambertConvergenceError`).
*   **Pluggable Integrator Interface**: Supports numerical propagation using the Runge-Kutta 4(5) solver (`RK45Integrator`) with modular support for symplectic or custom adaptive integration solvers.
*   **Physical Collision Detection**: Incorporates physical planetary equatorial radii boundaries (`PHYSICAL_RADIUS`) inside propagation loops to raise terminal collision exceptions synchronously.
*   **Modular Perturbation Forces**: Provides a pluggable force model system (`ForceModel`) enabling composable ODE construction (combining point-mass gravity, J2 perturbations, solar radiation pressure, and atmospheric drag).

### 2. ASTRA Mission DSL (`astra.dsl`)
*   **Strict Pydantic v2 Schema**: High-level validation models verifying dry mass, fuel mass, Isp boundaries, time-of-flight spans, launch windows, constraints, objective weights, and multi-body flyby sequences with explicit DSM budgets.
*   **YAML/JSON Parsers**: Robust parser utility matching standard YAML/JSON specs directly to schemas.
*   **Mission Compiler**: Compiles high-level user specifications into domain-level strongly typed constructs (`Spacecraft`, `PropulsionSystem`, `CelestialBody`), converts UTC epochs to J2000 barycentric dynamical time (TDB) seconds, and extracts flyby sequences and DSM budgets, taking precedence over simple direct trajectories.

### 3. Trajectory Optimization Engine (`astra.optimization`)
*   **Search Space boundaries (`SearchSpace`)**: Tracks and structures parameters for J2000 departure epochs and time-of-flight seconds.
*   **Porkchop Grid Computation (`compute_porkchop`)**: Calculates precise grid points of departure dates and time of flight, querying SPICE targets and performing Lambert solutions to map orbital feasibility.
*   **Multi-Objective TPE Optimizer (`optimize_mission_bayesian`)**: Employs Optuna TPE and NSGA-II multi-objective samplers to evaluate optimal trade-offs between flight duration and total launch/capture $\Delta v$, returning best feasible solutions and complete Pareto-front structures.
*   **Maneuvers & Trajectories (`astra.state.trajectory`)**: Houses complete representations of impulsive orbital maneuvers (`Maneuver`) and multi-impulse orbital transfers (`Trajectory`).
*   **Multi-Leg Chain Resolver (`resolve_flyby_chain`)**: Solves a full multi-leg gravity assist sequence with explicit geometry and correction budget checking (unpowered vs powered/DSM) at every flyby node.

### 4. Explainability Engine (`astra.explainability`)
*   **Delta-V Budget Decompositions (`decompose_delta_v`)**: Breaks down launch and arrival maneuver magnitudes, percentages, and epoch points, applying a standard 3% navigation margin.
*   **Launch Window Rationales (`build_window_rationale`)**: Analyzes planetary phase angles at departure, synodic orbits, and departure C3 energies to generate computed rationales explaining optimal departure dates.
*   **Constraint Compliance Analysis (`analyze_constraints`)**: Identifies margins and boundary-binding limits (within 5% of constraint margins) for total transfer $\Delta v$ and flight duration.
*   **Pareto Tradeoff Interpretation (`analyze_pareto`)**: Computes optimal slope parameters (e.g. extra $\Delta v$ required per day saved) to interpret duration vs propellant trade-offs quantitatively across Pareto fronts.

### 5. Constraints Evaluation Engine (`astra.constraints`)
*   **Modular Constraint Checking**: Fully typed evaluation of physical (`check_min_periapsis`, `check_max_delta_v`), propellant (`check_propellant_budget` using the Tsiolkovsky equation), and temporal (`check_max_duration`, `check_launch_window`) limits.
*   **Unified Violation Reports**: Produces a structured `ConstraintReport` containing categorised lists of `ConstraintViolation` objects based on severity (`hard` vs `soft`).
*   **Decoupled Search Feasibility**: Integrated within global optimization solvers to enforce bounds while separating propellant budget limits to enable path-finding under under-fueled spacecraft designs.

---

## Modular Perturbation Forces Layer (`astra.physics.forces`)

ASTRA features a pluggable force model architecture to enable high-fidelity numerical propagation. The ODE solver in `propagator.py` builds the state derivative dynamically from a composable list of force components.

### 1. Pluggable Force Interface (`ForceModel`)
Every force model implements the abstract base class `ForceModel` and defines:
```python
def acceleration(self, state_vec: np.ndarray, t: float) -> np.ndarray:
```
where `state_vec` is `[x, y, z, vx, vy, vz]` (in `np.float64`) and returns `[ax, ay, az]` in km/s². All models are numerically protected to return a zero vector if `r_mag < 1e-6` km.

### 2. Implemented Force Models

#### PointMassGravity
Computes Keplerian central body gravity acceleration:

$$\mathbf{a}_{\text{grav}} = -\frac{\mu \mathbf{r}}{\|\mathbf{r}\|^3}$$

#### J2Perturbation
Models the oblateness perturbation of a planet:

$$a_x = \text{factor} \cdot x \left(\frac{5z^2}{\|\mathbf{r}\|^2} - 1\right)$$

$$a_y = \text{factor} \cdot y \left(\frac{5z^2}{\|\mathbf{r}\|^2} - 1\right)$$

$$a_z = \text{factor} \cdot z \left(\frac{5z^2}{\|\mathbf{r}\|^2} - 3\right)$$

where $\text{factor} = \frac{3}{2} J_2 \mu \frac{R_{\text{body}}^2}{\|\mathbf{r}\|^5}$. Equatorial radii ($R_{\text{body}}$) are retrieved from `PHYSICAL_RADIUS`. Constants are provided in `J2_CONSTANTS` (e.g., Earth: $1.08263 \times 10^{-3}$, Mars: $1.96045 \times 10^{-3}$).

#### SolarRadiationPressure
Calculates acceleration from solar photons, assuming the spacecraft is always in sunlight:

$$\mathbf{a}_{\text{srp}} = \frac{C_r A_{\text{m2}} P_{\text{solar}}}{\text{mass}_{\text{kg}}} \left(\frac{\text{AU}}{\|\mathbf{r}_{\text{sc}}\|}\right)^2 \mathbf{u}_{\text{sun}} \cdot 10^{-3}$$

where $A\_{\text{m2}}$ is cross-sectional area, $C\_r$ is reflectivity, $\mathbf{u}\_{\text{sun}} = -\mathbf{r}\_{\text{sc}} / \|\mathbf{r}\_{\text{sc}}\|$ is the unit vector pointing toward the Sun, and $P\_{\text{solar}} = 4.56 \times 10^{-6}$ N/m² at 1 AU ($1.496 \times 10^8$ km).

#### AtmosphericDrag
Computes drag using an exponential density model:

$$\mathbf{a}_{\text{drag}} = -\frac{1}{2} C_d \frac{A_{\text{m2}}}{\text{mass}_{\text{kg}}} \rho \|\mathbf{v}\| \mathbf{v} \cdot 10^{-3}$$

where $\rho = \rho\_0 e^{-\frac{h}{H}}$ (in kg/m³) is scaled to standard physical surface densities (Earth: $1.225$ kg/m³, Mars: $0.020$ kg/m³) from `ATMOSPHERE_CONSTANTS`. Altitudes exceeding the scale-height cutoff (Earth: $1000$ km, Mars: $200$ km) bypass the computation to return zero acceleration.

### 3. Known Limitations of Current Perturbation Models

#### No Shadow / Eclipse Model
Solar Radiation Pressure assumes the spacecraft has a line of sight to the Sun at all times, ignoring planetary shadows (umbra/penumbra).

#### Constant Scale Height
Atmospheric drag uses a static exponential atmosphere model with a single constant scale height ($H$). It does not account for diurnal/solar-cycle variation, temperature fluctuations, or atmospheric rotation.

#### Spherical Drag Coefficient
Spacecraft drag ($C_d$) is treated as isotropic and constant, ignoring orientation, attitude, and complex spacecraft geometry.

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
│       │   └── forces/               # Modular force models (Gravity, J2, SRP, Drag)
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

### 5. Hybrid Optimization (Global Bayesian + Local L-BFGS-B Refinement)
The hybrid optimization strategy combines the global search capabilities of Bayesian optimization (TPE / NSGA-II) with the rapid local convergence of SciPy's gradient-based `L-BFGS-B` method.

#### When to use which:
*   **Bayesian Optimizer (`optimize_mission_bayesian`)**: Best for mapping the complete global launch window search space and generating a diverse set of trade-offs along a multi-objective Pareto front.
*   **Hybrid Optimizer (`optimize_mission_hybrid`)**: Best when you need to refine the global solutions and converge to the true local minimum of the smooth, convex porkchop $\Delta v$ surface. It takes the top-K feasible solutions found by the Bayesian phase and executes L-BFGS-B local refinement on each.

#### Expected Improvement:
Local refinement typically improves the total trajectory $\Delta v$ by **0.01 to 0.50 km/s** over pure Bayesian global optimization (which operates on a coarser parameter resolution), converging closer to the true physical minimum.

#### Code Example:
```python
from astra.physics import PhysicsKernel
from astra.dsl import parse_mission_file, compile_mission
from astra.optimization import optimize_mission_hybrid

# Load SPICE and compile mission
kernel = PhysicsKernel().load()
dsl = parse_mission_file("data/benchmarks/earth_mars_2031.yaml")
mission = compile_mission(dsl, kernel.ephemeris)

# Run hybrid optimization: 300 Bayesian trials + local L-BFGS-B refinement on top-5 results
result = optimize_mission_hybrid(
    mission, kernel,
    n_trials_bayesian=300,
    n_refine_top_k=5,
    time_limit=60.0,
    seed=42,
)

if result.converged and result.best_trajectory:
    print(f"Hybrid optimizer refined trajectory!")
    print(f"Best Delta-V: {result.best_trajectory.delta_v_total:.3f} km/s")
    print(f"Phase 1 (Bayesian) Best Δv: {result.phase1_best_dv:.3f} km/s")
    print(f"Phase 2 (Refined) Best Δv: {result.phase2_best_dv:.3f} km/s")
    print(f"Refinement Improvement: {result.refinement_improvement_km_s:.4f} km/s")
```
### 6. Neural Pre-filtering Layer (Neural Acceleration)
ASTRA implements a neural-network pre-filtering layer to accelerate global trajectory optimization by skipping expensive physics computations for regions of the parameter space that are highly unlikely to be feasible.

#### The Neural Surrogate Interface
All neural components implement the abstract `NeuralSurrogate` interface, which enforces that `requires_physics_validation` strictly returns `True`. This ensures that every trajectory suggested by the neural layer is fully verified by the double-precision physics core (patched-conics and the Lambert solver) before acceptance, preventing unphysical or inaccurate results from entering the mission plan.

#### Physics-Grounded Geometric Features
Instead of temporal placeholders, ASTRA extracts 8 physically predictive geometric features computed in `< 0.1ms` directly from planetary position and velocity vectors:
1.  **Departure Epoch (`dep_epoch_normalized`)**: Departure date normalized across the search window to `[0, 1]`.
2.  **Time of Flight (`tof_normalized`)**: Flight duration normalized across the search span to `[0, 1]`.
3.  **Planet Phase Angle (`phase_angle_rad / π`)**: The orbital radial alignment angle between the origin and destination bodies at departure, normalized to `[0, 1]`.
4.  **Origin Distance (`r1_AU`)**: Distance of the origin planet from the Sun in Astronomical Units, normalized to `[0, 1]` by dividing by `5.0` and clipping.
5.  **Destination Distance (`r2_AU`)**: Distance of the destination planet from the Sun in Astronomical Units, normalized to `[0, 1]` by dividing by `5.0` and clipping.
6.  **Rough $v_{\infty}$ (`v_inf_rough / 10.0`)**: Approximate hyperbolic excess velocity estimated using the vis-viva equation on the transfer ellipse, scaled to `[0, 3]`.
7.  **Synodic Progress (`synodic_progress`)**: Progress of the departure date within the synodic period cycle of the origin and destination bodies `[0, 1]`.
8.  **Hohmann TOF Ratio (`tof_to_hohmann`)**: Ratio of the flight duration to the analytical Hohmann transfer time between circularized orbits, clipped to `[0, 4]`.

Dividing planetary distances `r1_AU` and `r2_AU` by `5.0` and clipping onto `[0, 1]` provides consistent input scaling with other features. This prevents backpropagation gradient instability and allows the classifier to generalize across different launch epochs and target planets (e.g. Venus vs Mars).

#### Pretraining and Search Filtering
*   **Feasibility Boundary**: The target feasibility label is defined as a total $\Delta v < 15.0$ km/s.
*   **Optuna Integration**: For the first 100 trials, global search explores the objective landscape freely. Starting at trial 100, the search engine queries planetary states at `dep` and `dep + tof`, computes the 8 geometric features, and queries the `FeasibilityClassifier`.
*   **Pruning**: If the predicted feasibility probability is below `0.3`, the physics call is skipped, pruning the branch immediately with a penalty cost. This avoids running the Lambert solver on infeasible transfers, saving 40-70% of evaluation time.
*   **Online Updates**: The classifier updates online based on the simulator feedback from every actual physics execution, adapting to local variations.

#### Evaluation Metrics
During performance evaluations, ASTRA computes:
*   **Metrics**: Accuracy, Precision, Recall, and ROC AUC.
*   **Confusion Matrix**: Complete counts of True Positives (`tp`), False Positives (`fp`), True Negatives (`tn`), and False Negatives (`fn`) at a decision threshold of `0.3` to explicitly monitor and minimize false negatives (avoiding pruning promising orbits).
*   **Scikit-Learn Fallback**: If `scikit-learn` is not present, ASTRA utilizes a mathematically exact Wilcoxon-Mann-Whitney U-statistic rank sum calculation in pure NumPy to compute the ROC AUC metric.

---

### 7. Uncertainty-Aware 6D PINN Surrogate & Guided MCTS
ASTRA features an advanced physics-informed neural network (PINN) surrogate ensemble
integrated directly with MCTS planning and active learning loops.

#### 6D Cartesian Velocity Predictor (`LambertPINN`)
Predicts the 6 components of departure and arrival heliocentric velocity vectors:
$$[v_{x,d}, v_{y,d}, v_{z,d}, v_{x,a}, v_{y,a}, v_{z,a}]$$
Linear output layers allow predicted components to correctly range into negative values.
Training minimizes a combined loss of target MSE and three physical residuals:
1.  **Vis-Viva constraint** matching departure velocity with the transfer semi-major axis.
2.  **Specific mechanical energy conservation** between the orbital energy and semi-major axis.
3.  **Specific angular momentum conservation** ($\mathbf{h}_{\text{dep}} = \mathbf{h}_{\text{arr}}$) along the transfer ellipse.

#### Epistemic Uncertainty & Active Learning (`LambertPINNEnsemble`)
*   **Deep Ensemble**: Instantiates $N=5$ models with unique initializations to quantify prediction
    variance and standard deviation (uncertainty) across search regions.
*   **Active Learning Manager**: Automatically monitors prediction uncertainty. If uncertainty
    exceeds `uncertainty_threshold`, it queries the physical Izzo Lambert solver, stores the exact
    solution in the buffer, and triggers network retraining when the buffer size reaches
    `retrain_every`.

#### Uncertainty-Aware MCTS Search
The UCT selection formula penalizes nodes with high prediction uncertainty:
$$UCT = \text{exploitation} + c \sqrt{\frac{\ln N}{n}} - w_{\text{unc}} \sigma$$

#### Multi-Stage Trajectory Validation
Candidate trajectories are verified against a 3-stage physics check in `validate_trajectory()`:
-   **Stage 1**: Check surrogate estimates against bounds.
-   **Stage 2**: Resolve the exact multi-revolution Lambert transfer trajectory.
-   **Stage 3**: Numerically propagate the exact departure state to verify arrival position error.

---

### 8. Trajectory Rationale & Explanations
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

### 9. Trajectory Analytics & Pareto Metrics

ASTRA computes advanced analytics on Pareto-optimal fronts and individual trajectories:

*   **Pareto Dominance**: Objective vectors $\mathbf{a}$ dominate $\mathbf{b}$ if $\mathbf{a}$ is at least as good as $\mathbf{b}$ in all objectives, and strictly better in at least one (minimizing total $\Delta v$ and time of flight).
*   **Hypervolume Indicator (HVI)**: Quantifies the area of objective space dominated by the Pareto front relative to a reference point (set at $1.1 \times \max(\text{objectives})$). Larger hypervolume indicates a higher quality front.
*   **Pareto Spread**: Measures the diversity and coverage of the frontier by computing the average pairwise Euclidean distance of normalized Pareto points.
*   **Sensitivity Analysis**: Approximates the local derivative of total transfer $\Delta v$ with respect to Time of Flight ($TOF$) and Departure Epoch ($dep$) via central finite differences at the optimal trajectory point:

$$\frac{\partial (\Delta v)}{\partial x} \approx \frac{f(x + \Delta x) - f(x - \Delta x)}{2 \Delta x}$$

    Robust bounds gracefully catch infeasible space exceptions, returning status metadata instead of failing.

---

## FastAPI API Layer & Trajectory Storage (`astra.api` and `astra.data`)

ASTRA exposes a standard, high-performance FastAPI web application layer to query celestial bodies, calculate porkchop opportunity grids, serialize 3D trajectory rendering-ready data structures, and persist optimization runs.

### Decoupled Router Architecture
*   **Modular Route Registries**: Endpoints are split into specialized submodule routers under `src/astra/api/routes/` (`health`, `missions`, `trajectories`, `physics`).
*   **Strict Typed Schemas**: Fully enforces Pydantic v2 schemas for all requests and simple response payloads (e.g. `HealthResponse`, `BodyStateResponse`, `JobStatusResponse`).
*   **Correlation & Latency Middleware**: Integrates a `RequestLoggingMiddleware` that stamps incoming requests with short correlation IDs, logs execution duration, and injects performance tracking response headers.

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
*   `GET /v1/missions/{id}/sensitivity` - Retrieve central finite-difference sensitivities (time of flight and departure epoch) for the best trajectory of a completed job.
*   `GET /v1/missions/{id}/pareto-metrics` - Retrieve Plotly-ready Pareto frontier coordinates, 2D hypervolume area, and Pareto spread diversity metrics for a completed job.
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

## Heliocentric $\Delta v$ vs Mission $\Delta v$

ASTRA differentiates between **Heliocentric $\Delta v$** (which represents the deep-space velocity changes relative to the Sun) and **Mission $\Delta v$** (which incorporates planetary gravity wells via sphere-of-influence patching for realistic launch vehicle injection and orbital insertion).

### 1. Mathematical Formulations

#### Heliocentric $\Delta v$
Assumes maneuvers are performed in deep space far from planetary gravity wells.

$$\Delta v_{\text{helio}} = \| \mathbf{v}_{\text{sc,dep}} - \mathbf{v}_{\text{body,dep}} \| + \| \mathbf{v}_{\text{body,arr}} - \mathbf{v}_{\text{sc,arr}} \| = v_{\infty,\text{dep}} + v_{\infty,\text{arr}}$$

where $v\_{\infty}$ is the hyperbolic excess speed at departure/arrival.

#### Patched-Conics (Sphere of Influence) $\Delta v$
Computes precise orbital maneuvers within the planetary gravity wells (Laplace SOI) using circular parking/capture orbits:

##### Departure Burn (Trans-Mars Injection, TMI)
Accelerates from a circular parking orbit at altitude $h\_{\text{park}}$:

$$v_{\text{park}} = \sqrt{\frac{\mu_{\text{origin}}}{R_{\text{origin}} + h_{\text{park}}}}$$

$$v_{\text{hyp,dep}} = \sqrt{v_{\infty,\text{dep}}^2 + \frac{2\mu_{\text{origin}}}{R_{\text{origin}} + h_{\text{park}}}}$$

$$\Delta v_{\text{TMI}} = v_{\text{hyp,dep}} - v_{\text{park}}$$

##### Arrival Burn (Mars Orbit Insertion, MOI)
Decelerates from a hyperbolic approach to a circular capture orbit at altitude $h\_{\text{capture}}$:

$$v_{\text{cap}} = \sqrt{\frac{\mu_{\text{dest}}}{R_{\text{dest}} + h_{\text{capture}}}}$$

$$v_{\text{hyp,arr}} = \sqrt{v_{\infty,\text{arr}}^2 + \frac{2\mu_{\text{dest}}}{R_{\text{dest}} + h_{\text{capture}}}}$$

$$\Delta v_{\text{MOI}} = v_{\text{hyp,arr}} - v_{\text{cap}}$$

##### Elliptical Capture Support
If an elliptical capture orbit is specified (using `apoapsis_km` as the radius from the body center), the MOI burn inserts the spacecraft into the capture ellipse at its periapsis. The arrival delta-v is the periapsis deceleration burn only:

$$v_{\text{peri,ellipse}} = \sqrt{\mu_{\text{dest}} \left(\frac{2}{r_{\text{peri}}} - \frac{1}{a_{\text{capture}}}\right)}$$

$$\Delta v_{\text{MOI}} = v_{\text{hyp,arr}} - v_{\text{peri,ellipse}}$$

where $r\_{\text{peri}} = R\_{\text{dest}} + h\_{\text{capture}}$ and $a\_{\text{capture}} = (r\_{\text{peri}} + r\_{\text{apo}}) / 2.0$.

##### Circularization Burn Modeling
If the mission requires circularization from this capture ellipse, the subsequent circularization burn at the capture orbit's apoapsis is calculated as:

$$\Delta v_{\text{circularization}} = v_{\text{circular}} - v_{\text{apo,ellipse}}$$

where $v\_{\text{circular}} = \sqrt{\frac{\mu\_{\text{dest}}}{r\_{\text{apo}}}}$ and $v\_{\text{apo,ellipse}} = \sqrt{\mu\_{\text{dest}} \left(\frac{2}{r\_{\text{apo}}} - \frac{1}{a\_{\text{capture}}}\right)}$.

##### Total Mission $\Delta v$
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


---

## Gravity Assist & Flyby Physics (`astra.physics.flyby`)

Gravity assists (flybys) enable interplanetary trajectories to gain or lose heliocentric energy without expending propellant by exploiting the gravitational field of a planetary body.

### Trajectory Formulations & Classifications

*   **Direct Transfers**: A simple two-impulse conic transfer between the origin planet and the destination planet (e.g. Earth to Mars direct).
*   **Flyby Transfers**: A multi-body transfer sequence that utilizes one or more intermediate planetary flybys (e.g. Earth -> Venus -> Mars) to shape the trajectory.
*   **Unpowered Flybys**: The spacecraft enters the planet's Sphere of Influence (SOI) and exits with the same hyperbolic excess speed magnitude ($|v_{\infty, out}| = |v_{\infty, in}|$), but its direction is deflected by the hyperbolic turn angle $\delta$ due to planetary gravity.
*   **Powered Flybys**: The gravity assist is augmented by an impulsive propulsion burn $\Delta v_{\text{powered}}$ performed at periapsis. This alters the outgoing hyperbolic excess speed ($|v_{\infty, out}| \ne |v_{\infty, in}|$) and changes the turn angle.

### Hyperbolic Deflection Physics
The hyperbolic turn angle $\delta$ is given by:

$$\sin(\delta/2) = \frac{1}{e}$$

where the eccentricity $e$ of the hyperbolic flyby passage is:

$$e = 1 + \frac{r_p v_{\infty}^2}{\mu}$$

For asymmetric powered flybys, the total turn angle is the sum of the incoming and outgoing hyperbolic asymptote angles:

$$\delta = \arcsin(1/e_{\text{in}}) + \arcsin(1/e_{\text{out}})$$

### Feasibility Checking & Powered Correction Resolver
In multi-leg sequences where incoming and outgoing excess speed magnitudes are fixed by independent Lambert transfers, the turn angle and burn requirements are solved via a deterministic search over periapsis $r_p$ within safe altitude bounds:
1. **Unpowered Feasibility**: Verified via `check_flyby_feasibility` to see if gravity alone satisfies the required turn with numerical speed conservation.
2. **Powered Correction Search**: If required deflection exceeds the unpowered ceiling, a bounded grid search finds the $r_p$ that satisfies $\delta$ while minimizing powered $\Delta v$. If no valid $r_p$ achieves the turn, or the required $\Delta v$ exceeds the local and DSM budgets, the leg is hard-rejected.

---

## Mission Phase Planning with MCTS (`astra.optimization.mcts`)

ASTRA implements a discrete Monte Carlo Tree Search (MCTS) planner to solve the combinatorial problem of finding optimal flyby body sequences and launch/flight schedules.

### MCTS Search Logic
-   **Nodes**: Represent a `PhaseState` consisting of the current celestial body, arrival epoch, incoming heliocentric velocity, and cumulative $\Delta v$ spent.
-   **Actions**: Selecting the next candidate flyby body (e.g., `VENUS`, `EARTH`, `MOON`) and a discrete time of flight (TOF).
-   **Rollouts**: Randomly selects valid transitions to search for paths reaching the destination body (e.g., `MARS`).
-   **UCT Selection**: Balances exploration and exploitation using the Upper Confidence Bound for Trees:

$$UCT = \frac{Q(s, a)}{N(s, a)} + C \times \sqrt{\frac{\ln N(s)}{N(s, a)}}$$

-   **Reward Function**: Evaluates feasibility and total $\Delta v$ cost. Feasible sequences that reach the destination within the budget receive a reward of:

$$\text{Reward} = 1.0 - \frac{\Delta v_{\text{total}}}{\Delta v_{\text{budget}}}$$

    Infeasible or incomplete sequences receive a reward of $0.0$, ensuring the planner prioritizes successful arrivals.

---

## Current Project State & Scientific Approximations

### What is Implemented
*   **Modular Forces Layer**: Composable force system (`ForceModel`) enabling integration of J2 oblateness, solar radiation pressure, and exponential atmospheric drag.
*   **Elliptical Capture & Circularization**: Multi-impulse capture orbit targeting, supporting elliptical periapsis insertion and subsequent apoapsis circularization burns.
*   **Powered and Unpowered Hyperbolic Flyby Models**: Vector rotations about orbital plane normals, periapsis speed calculations, and minimum safe altitude constraints.
*   **Safe Flyby Altitudes**: Physical radius boundaries combined with atmospheric clearance margins for `MERCURY`, `VENUS`, `EARTH`, `MOON`, `MARS`, `JUPITER`, and `SATURN`.
*   **MCTS Sequence Planner**: Traversal with uncertainty penalization, UCT selection, and dynamic path validation featuring explicit flyby feasibility checking and DSM budget tracking.
*   **Multi-Leg Chain Resolver**: `resolve_flyby_chain` algorithm for deterministic geometric and budget verification of gravity assist paths.

### Remaining Scientific Approximations
*   **Instantaneous Flyby (Patched-Conics)**: Flybys are treated as occurring instantaneously at the planet's heliocentric position. The heliocentric delta-v gained `dv_helio_km_s` is a patched-conics approximation ($||v_{\infty, out} - v_{\infty, in}||$) rather than a true numerical heliocentric integration of the trajectory under 3-body or heliocentric gravity.
*   **Planar Orientation**: Choice of flyby orbit plane normal defaults to an arbitrary perpendicular axis if not explicitly provided.

### Intentionally Out of Scope
*   **N-Body Numerical Integration**: High-fidelity gravitational propagation under multiple bodies simultaneously.
*   **Complex Aerodynamics**: Aerothermal modeling, lift forces, and active guidance during aerocapture/aerobraking passes.

---

## Validation & Verification Registry

ASTRA enforces a multi-tiered trajectory validation strategy to ensure numerical precision and scientific credibility. This methodology is split into three main categories:

1. **Analytical Validation**: Direct mathematical checks of core solvers (such as the Universal Variables Izzo Lambert solver and two-body Keplerian propagators) against closed-form analytical solutions and standard textbook problems (e.g. [Curtis Example 5.2](docs/benchmarks.md#1-curtis-earth-satellite-lambert-bvp)).
2. **Benchmark Validation**: Optimization regression tests, search efficiency benchmarks, and Pareto frontier generation checks over synthetic trajectory search spaces (e.g., the [Earth-Mars 2031](docs/benchmarks.md#7-earth-mars-2031-standard--long-tof) reference mission).
3. **Historical Mission-Inspired Validation**: Validation of launch windows, C3 departure energies, transfer durations (TOF), and flyby deflection angles inspired by historical planetary missions (e.g., [Mars Odyssey 2001](docs/benchmarks.md#4-mars-odyssey-2001), [MRO 2005](docs/benchmarks.md#5-mars-reconnaissance-orbiter-2005), [Cassini Venus gravity assists](docs/benchmarks.md#3-cassini-venus-flyby-1998), and [Apollo-style Lunar free-return transfers](docs/benchmarks.md#6-lunar-free-return-trajectory)). These are handled as approximations within the limits of patched-conics and two-body dynamics.

For a detailed breakdown of every benchmark currently supported, physical assumptions, target values, expected outputs, and runtime characteristics, see the canonical [Benchmarks Registry](docs/benchmarks.md).

---

## Optimization Strategies

ASTRA supports 6 trajectory optimization strategies tailored for different planetary geometries, search space dimensions, and runtime constraints:

1. **Bayesian Optimization (`bayesian`)**: Trajectory search powered by Tree-structured Parzen Estimators (TPE) via Optuna to navigate multi-dimensional launch windows.
2. **Hybrid Optimization (`hybrid`)**: Combines global Bayesian exploration with local gradient-based optimization (L-BFGS-B) to refine the top candidate trajectory windows.
3. **Neural Pre-filtering (`neural`)**: Active classifier-guided search that uses geometric features and a `FeasibilityClassifier` surrogate to prune infeasible transfer regions, reducing Lambert solver queries.
4. **Discrete sequence search (`mcts`)**: Trajectory sequence planner utilizing Monte Carlo Tree Search to explore planetary flyby paths and optimal synodic schedules.
5. **Gravity Deflection (`flyby`)**: Deflection optimization for powered and unpowered flybys, satisfying minimum periapsis clearances.
6. **PINN Acceleration (`pinn`)**: Deep ensemble Cartesian velocity prediction utilizing Physics-Informed Neural Networks with conservation loss residuals and epistemic uncertainty-aware active learning.
7. **Gated Chain Optimization (`chain_gated`)**: Multi-leg trajectory chain optimization over departure epoch and leg TOFs, enforcing strict physical deflection feasibility via `optimize_mission_chain` and the gated chain resolver.

## Running the Acceptance Test

To execute the full end-to-end scientific acceptance test verifying the Physics, DSL, Constraints, Optimization, Explainability, Pareto Quality, Sensitivity Analysis, and Data Persistence layers:

```bash
uv run pytest tests/benchmark/test_full_system_acceptance.py -v -m slow
```

## Canonical Benchmark Registry

| ID | Benchmark Name | Category | Expected $\Delta v$ (km/s) / Verification Target |
|---|---|---|---|
| 1 | Curtis Earth Satellite Lambert BVP | Analytical Validation | Published $\mathbf{v}_1 = [-5.9925, 1.9254, 3.2456]$ km/s (Error $< 1\times 10^{-4}$ km/s) |
| 2 | Earth-Moon Lambert Transfer | Analytical & Physics Validation | TLI excess velocity $v_{\infty} \approx 6.2$ km/s |
| 3 | Cassini Venus Flyby (1998) | Historical Mission-Inspired | Deflection angle $\approx 39.95^\circ$ |
| 4 | Mars Odyssey (2001) | Historical Mission-Inspired | $\approx 5.75$ km/s |
| 5 | Mars Reconnaissance Orbiter (2005) | Historical Mission-Inspired | C3 departure energy $\approx 16.84\text{ km}^2/\text{s}^2$ |
| 6 | Lunar Free-Return Trajectory | Historical Mission-Inspired | Feasible outbound Apollo-class transfer ($\approx 3$ days TOF) |
| 7 | Earth-Mars 2031 (Standard) | Benchmark Validation | $3.0 < \Delta v < 8.0$ km/s (Optimal: $\approx 6.27$ km/s) |
| 8 | Earth-Mars 2031 (Long TOF) | Benchmark Validation | Multi-rev optimal $\Delta v \approx 14.29$ km/s |
| 9 | Earth-Venus-Mars 2032 Flyby | Multi-body Gravity-Assist | Total $\Delta v < 9.0$ km/s, Venus periapsis $> 300$ km |
| 10 | Regression Lock | Regression Prevention | Standard Earth-Mars 2031 $\Delta v$ within $\pm 2\%$ of baseline |
| 11 | Galileo VEEGA (1989) | Historical Mission-Inspired | Multi-leg gated chain optimizer test (Correctly Non-convergent due to hardcoded bounds) |
| 12 | Cassini VVE (1997) | Historical Mission-Inspired | Repeated-body query verification & non-convergence under patched-conics limits |





