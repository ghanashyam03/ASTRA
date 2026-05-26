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
*   **Mission Compiler**: Compiles high-level user specifications into domain-level strongly typed constructs (`Spacecraft`, `PropulsionSystem`, `CelestialBody`) and converts UTC epochs to J2000 barycentric dynamical time (TDB) seconds automatically.

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
│       ├── physics/                  # Lambert Solver, Ephemeris Engine, Propagator
│       └── state/                    # Spacecraft and Orbital State Primitives
└── tests/
    └── unit/
        ├── dsl/                      # Mission DSL Unit Tests
        ├── physics/                  # Physics Core Unit Tests
        └── state/                    # Primitive Primitives Unit Tests
```

---

## Quickstart & Commands

### 1. Developer Setup
Standard Python environments are managed via the high-performance `uv` package manager:
```powershell
# Run the entire automated test suite (all 18 unit tests)
uv run pytest

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

### 3. Basic Orbital Propagation
```python
import numpy as np
from astra.state import OrbitalState, CelestialBody, ReferenceFrame
from astra.physics import propagate_two_body

# Initialize a low-Earth orbit state (float64)
state = OrbitalState(
    epoch=0.0,
    position=np.array([6778.137, 0.0, 0.0], dtype=np.float64),
    velocity=np.array([0.0, 7.668, 0.0], dtype=np.float64),
    frame=ReferenceFrame.J2000,
    central_body=CelestialBody.EARTH
)

# Propagate state by one orbit duration
final_state = propagate_two_body(state, dt_seconds=5400.0)
print(f"Final Position: {final_state.position} km")
print(f"Integration Steps: {final_state.metadata['nsteps']}")
```
