# ASTRA Trajectory Benchmark Registry & Validation Strategy

This document serves as the canonical benchmark registry for the Autonomous Space Trajectory Reasoning Architecture (ASTRA). It defines the validation and regression benchmarking strategy, outlining every supported benchmark case, its purpose, physical assumptions, target metrics, runtime characteristics, and known limitations.

---

## Validation Typology

To maintain the scientific credibility of ASTRA, validations are classified into three distinct categories:

1. **Analytical Validation**: Mathematical verification of core solvers (e.g., Lambert solvers, propagators) against exact closed-form analytical solutions or standardized textbook problems.
2. **Benchmark Validation**: Regression and performance tests designed to verify optimizer convergence, search speed, and feasibility pre-filtering correctness over synthetic reference trajectories.
3. **Historical Mission-Inspired Validation**: Verifications based on historical space missions (e.g., Mars Odyssey, MRO, Apollo, Cassini) to confirm that ASTRA reproduces historical parameters within acceptable tolerances, recognizing model simplifications.

---

## Supported Benchmark Inventory

### 1. Curtis Earth Satellite Lambert BVP
*   **Category**: Analytical Validation
*   **Purpose**: Verify the exact mathematical correctness of the Universal Variables Izzo Lambert solver.
*   **Source**: *Orbital Mechanics for Engineering Students* (Curtis), Example 5.2.
*   **Physical Assumptions**: Geocentric two-body Keplerian mechanics, zero planetary perturbations, spherical Earth gravitational field.
*   **Validation Targets**:
    *   Initial position: $\mathbf{r}_1 = [5000.0, 10000.0, 2100.0]$ km
    *   Final position: $\mathbf{r}_2 = [-14600.0, 2500.0, 7000.0]$ km
    *   Time of flight: $1.0$ hour ($3600.0$ seconds)
    *   Earth gravitational parameter: $\mu_\oplus = 398600.4418\text{ km}^3/\text{s}^2$
*   **Expected Outputs**:
    *   Departure velocity: $\mathbf{v}_1 = [-5.992495, 1.925367, 3.245638]$ km/s
    *   Arrival velocity: $\mathbf{v}_2 = [-3.312459, -4.196619, -0.385289]$ km/s
    *   Tolerance: Absolute error $< 1\times 10^{-5}$ km/s.
*   **Runtime Characteristics**: $< 1$ ms (Deterministic mathematical convergence in $\sim 6$ iterations).
*   **Limitations & Simplifications**: Ignores Earth oblateness ($J_2$), solar radiation pressure, lunar third-body gravity, and atmospheric drag.

### 2. Earth-Moon Lambert Transfer
*   **Category**: Analytical & Physics Validation
*   **Purpose**: Test Lambert solver convergence at close-proximity planetary-satellite scales with short flight durations.
*   **Physical Assumptions**: Geocentric framework, Moon treated as a point mass moving according to DE440 ephemerides, patched-conics approximation at departure.
*   **Validation Targets**:
    *   Departure orbit: $200$ km altitude circular LEO ($r_1 = 6571.0$ km).
    *   Arrival position: Center of the Moon ($r_2 = \mathbf{r}_\text{Moon}$) at epoch `2025-06-01T00:00:00`.
    *   Time of flight: $3.5$ days.
    *   Trans-Lunar Injection (TLI) excess velocity $v_\infty$: $2.0 < v_{\infty} < 7.0$ km/s.
*   **Expected Outputs**: Convergence is guaranteed, with TLI excess speed approximately $\sim 6.2$ km/s.
*   **Runtime Characteristics**: $< 5$ ms (Single SPICE lookup + single Lambert solve).
*   **Limitations & Simplifications**: Ignores three-body gravitational interactions (e.g. Earth-Moon-spacecraft CR3BP dynamics) and lunar gravity well capture.

### 3. Cassini Venus Flyby (1998)
*   **Category**: Historical Mission-Inspired Validation (Physics Consistency)
*   **Purpose**: Validate unpowered hyperbolic flyby physics (gravity assist) including deflection angles and heliocentric energy changes.
*   **Physical Assumptions**: Patched-conics approximation within the planet's Sphere of Influence (SOI), instantaneous flyby at the planet's position relative to the Sun.
*   **Validation Targets**:
    *   Approach excess speed $v_{\infty,\text{in}}$: $9.7$ km/s at Venus.
    *   Periapsis altitude $h_p$: $600$ km.
    *   Hyperbolic deflection angle (turn angle): $\approx 39.95^\circ$.
*   **Expected Outputs**:
    *   Deflection angle: $39.95^\circ$ (within $\pm 10^\circ$ tolerance).
    *   Heliocentric energy gain: Derived from planet-relative deflection.
*   **Runtime Characteristics**: $< 1$ ms.
*   **Limitations & Simplifications**: Uses simplified two-body hyperbolic passage. The historical Cassini flyby had a periapsis altitude of $284$ km for the first flyby (April 1998) and $600$ km for the second flyby (June 1999). Additionally, historical deflection is measured in the heliocentric frame (incorporating Venus's orbital velocity vector), whereas this validation focuses on planet-relative two-body hyperbolic deflection consistency.

### 4. Mars Odyssey (2001)
*   **Category**: Historical Mission-Inspired Validation
*   **Purpose**: Validate Earth-Mars direct transfers using SPICE ephemerides, launch window parsing, and the hybrid optimization engine.
*   **Specification File**: `data/benchmarks/mars_odyssey_2001.yaml`
*   **Physical Assumptions**: Two-body patched-conics, double-precision ephemerides (DE440), circular Earth staging orbit at 185 km, elliptical Mars insertion orbit.
*   **Validation Targets**:
    *   Launch window: `2001-04-01` to `2001-04-14`.
    *   Historical launch: `2001-04-07`.
    *   Historical reference C3: $16.4\text{ km}^2/\text{s}^2$.
    *   Historical reference TOF: $200.0$ days.
*   **Expected Outputs**:
    *   Optimized TOF: $199.75$ days (within $10\%$ of historical 200 days).
    *   Optimized C3: $10.38\text{ km}^2/\text{s}^2$ (within $40\%$ of historical 16.4 $\text{km}^2/\text{s}^2$).
    *   Total $\Delta v$: $\sim 5.75$ km/s.
*   **Runtime Characteristics**: $\approx 1.5 - 2.5$ seconds using the hybrid optimizer (global TPE + L-BFGS-B local refinement).
*   **Limitations & Simplifications**: The optimizer minimizes total mission $\Delta v$ (departure + insertion). Because historical missions incorporate complex non-delta-v operational constraints (such as power/thermal orientation, communication windows, and landing site arrival requirements), the optimal trajectory found by ASTRA has a lower C3 departure energy ($10.38$ vs $16.4\text{ km}^2/\text{s}^2$) while matching the transfer duration almost exactly.

### 5. Mars Reconnaissance Orbiter (2005)
*   **Category**: Historical Mission-Inspired Validation
*   **Purpose**: Secondary independent validation of Earth-Mars direct transfers and hybrid optimizer convergence.
*   **Specification File**: `data/benchmarks/mro_2005.yaml`
*   **Physical Assumptions**: Patched-conics, staging altitude 185 km, elliptical capture orbit at Mars.
*   **Validation Targets**:
    *   Launch window: `2005-08-10` to `2005-08-14`.
    *   Historical launch: `2005-08-12`.
    *   Historical reference C3: $14.2\text{ km}^2/\text{s}^2$.
    *   Historical reference TOF: $211.0$ days.
*   **Expected Outputs**:
    *   Optimized TOF: $215.4$ days (within $10\%$ of reference).
    *   Optimized C3: $16.84\text{ km}^2/\text{s}^2$ (within $20\%$ of reference).
*   **Runtime Characteristics**: $\approx 1.5 - 2.5$ seconds.
*   **Limitations & Simplifications**: Similar to Mars Odyssey, the lack of operational, spacecraft, and scheduling constraints in the optimizer leads to a slightly different trade-off in TMI/MOI magnitudes, shifting the C3 energy.

### 6. Lunar Free-Return Trajectory
*   **Category**: Historical Mission-Inspired Validation
*   **Purpose**: Test Earth-Moon-Earth free-return geometry where the spacecraft loops around the Moon and returns to Earth without firing its engine.
*   **Specification File**: `data/benchmarks/lunar_free_return.yaml`
*   **Physical Assumptions**: Patched-conics Moon flyby, circular staging orbit at Earth (200 km), close lunar periapsis (2500 km).
*   **Validation Targets**:
    *   Launch window: `2025-06-01` to `2025-06-03`.
    *   Reference outbound TOF: $\approx 3$ days (Apollo class).
*   **Expected Outputs**: Feasible outbound transfer conics converging around $3 - 4$ days.
*   **Runtime Characteristics**: $< 1$ second.
*   **Limitations & Simplifications**: ASTRA's two-impulse transfer model approximates the outbound transfer leg to the Moon's sphere of influence. A full free-return loop requires three-body numerical integration (Earth + Moon gravity acting simultaneously), which is outside the scope of patched-conics solvers.

### 7. Earth-Mars 2031 (Standard & Long TOF)
*   **Category**: Benchmark Validation (Regression & Performance)
*   **Purpose**: Acceptance testing for the entire optimization pipeline (parsing, compilation, neural pre-filtering, Pareto frontier generation, and explainability).
*   **Specification Files**: `data/benchmarks/earth_mars_2031.yaml`, `data/benchmarks/earth_mars_long_tof_2031.yaml`
*   **Physical Assumptions**: Patched-conics, double-precision ephemerides, circular departure (200 km LEO), elliptical capture at Mars.
*   **Validation Targets**:
    *   Feasible $\Delta v$ range: $3.0 < \Delta v_\text{total} < 8.0$ km/s.
    *   Duration: $< 250$ days.
    *   Maneuvers: Exactly 2 (TMI and MOI).
    *   Multi-revolution: Verify cheaper 1-rev and 2-rev conics on long TOF windows.
*   **Expected Outputs**:
    *   Standard optimal $\Delta v$: $\approx 6.27$ km/s (departure: $3.69$ km/s, arrival: $2.58$ km/s).
    *   Long TOF optimal multi-rev $\Delta v$: $\approx 14.29$ km/s (compared to single-rev $44.4$ km/s).
*   **Runtime Characteristics**: $\approx 10 - 20$ seconds (or $< 5$ seconds with neural pre-filtering enabled).
*   **Limitations & Simplifications**: Standard 2-impulse interplanetary transfer model.

### 8. Curtis Example 5.2 (Geocentric Lambert BVP)
*   **Category**: Analytical validation
*   **Source**: Curtis, *Orbital Mechanics for Engineering Students*, 4th ed., Ex. 5.2.
*   **Validation Targets**:
    *   $\mathbf{r}_1 = [5000, 10000, 2100]$ km
    *   $\mathbf{r}_2 = [-14600, 2500, 7000]$ km
    *   Time of flight: $3600$ s
    *   $\mu = 398600.4418\text{ km}^3/\text{s}^2$ (Earth)
*   **Expected Outputs**:
    *   Published $\mathbf{v}_1 = [-5.9925, 1.9254, 3.2456]$ km/s
    *   ASTRA tolerance: $|\mathbf{v}_{1,\text{computed}} - \mathbf{v}_{1,\text{ref}}| < 1\times10^{-4}$ km/s per component
*   **Runtime Characteristics**: $< 1$ ms (pure math, no SPICE).
*   **Significance**: Gold standard for Lambert solver correctness.

### 9. Earth-Venus-Mars 2032 Flyby
*   **Category**: Multi-body gravity-assist validation
*   **Specification File**: `data/benchmarks/earth_venus_mars_2032.yaml`
*   **Expected Outputs**:
    *   Total $\Delta v < 9.0$ km/s
    *   Venus periapsis $> 300$ km altitude
*   **Tests**: Continuous periapsis optimization and multi-body DSL schema coverage.
*   **Runtime Characteristics**: $\approx 2$ minutes.

### 10. Regression Lock
*   **Category**: Regression prevention
*   **Captures**: Best $\Delta v$, Pareto size, and hypervolume indicator (HVI) for Earth-Mars 2031.
*   **Tolerance**: $\Delta v \pm 2\%$, Pareto size $\pm 30\%$.
*   **Update Process**: Run `uv run python tests/benchmark/update_baseline.py` after verified improvement.

### 11. Galileo VEEGA (1989)
*   **Category**: Historical Mission-Inspired Validation
*   **Purpose**: Test the multi-leg optimizer on a multi-year, 3-flyby gravity-assist sequence (Venus-Earth-Earth-Jupiter).
*   **Specification File**: `data/benchmarks/galileo_veega_1989.yaml`
*   **Physical Assumptions**: Patched-conics, double-precision ephemerides, circular departure (300 km LEO), circular capture at Jupiter (200000 km).
*   **Validation Targets**:
    *   Honest documentation of model fidelity limits (the patched-conics model cannot resolve Galileo's 731-day Earth-Earth resonance loop and 1094-day Earth-Jupiter transfer due to the optimizer's hardcoded 400-day leg TOF limit and single-rev restriction).
*   **Expected Outputs**: Convergence outcome is `False` (0 feasible out of 1500 evaluations) under these constraints, with strict physical self-consistency checks.
*   **Runtime Characteristics**: $\approx 5 - 15$ seconds.
*   **Limitations & Simplifications**: Neglects continuous third-body perturbations and resonant multi-rev legs.

### 12. Cassini VVE (1997)
*   **Category**: Historical Mission-Inspired Validation
*   **Purpose**: Test the multi-leg optimizer on a sequence with consecutive flybys of the same body (Venus-Venus-Earth-Saturn).
*   **Specification File**: `data/benchmarks/cassini_vve_1998.yaml`
*   **Physical Assumptions**: Patched-conics, double-precision ephemerides, circular departure (300 km LEO), circular capture placeholder at Saturn (100000 km).
*   **Validation Targets**:
    *   Verify repeated-body state lookups query distinct epochs.
    *   Characterize model fidelity limits (lack of 1:1 resonance for the Venus-Venus transfer and 400-day leg TOF limit).
*   **Expected Outputs**: Convergence outcome is `False` (0 feasible out of 1500 evaluations), with repeated-body lookups successfully verified to query distinct planetary positions.
*   **Runtime Characteristics**: $\approx 5 - 15$ seconds.
*   **Limitations & Simplifications**: Excludes Cassini's distant Jupiter pass; ignores 1:1 resonance physics and continuous third-body gravity.

### 13. MESSENGER Chain (2004)
*   **Category**: Historical Mission-Inspired Validation
*   **Purpose**: Test the multi-leg optimizer on a chain with the launch origin body (Earth) reappearing later in the sequence as a flyby (Earth-Earth-Venus-Venus-Mercury-Mercury-Mercury).
*   **Specification File**: `data/benchmarks/messenger_chain_2004.yaml`
*   **Physical Assumptions**: Patched-conics, double-precision ephemerides, circular departure (200 km LEO), circular capture at Mercury (200 km).
*   **Validation Targets**:
    *   Verify the origin body reused as a flyby queries distinct epochs (launch epoch vs ~1 year later flyby epoch).
    *   Characterize model fidelity limits (lack of resonant/multi-rev transfers and leg TOF bounds).
*   **Expected Outputs**: Convergence outcome is `False` (0 feasible out of 1500 evaluations), with the origin-vs-flyby epoch separation check successfully verified.
*   **Runtime Characteristics**: $\approx 5 - 15$ seconds.
*   **Limitations & Simplifications**: Ignores resonant orbits (1:1 Earth-Earth and Venus-Venus) and continuous solar gravity perturbations.

### 14. Voyager 2 Grand Tour (1977)
*   **Category**: Historical Mission-Inspired Validation
*   **Purpose**: Test the gated chain solver on a 4-flyby sequence visiting all four giant outer planets (Earth-Jupiter-Saturn-Uranus-Neptune) under generous leg TOF bounds.
*   **Specification File**: `data/benchmarks/voyager2_grand_tour_1977.yaml`
*   **Physical Assumptions**: Patched-conics, double-precision ephemerides, circular departure (200 km LEO), circular capture placeholder at Neptune (10000 km).
*   **Validation Targets**:
    *   Test optimizer search capability over a 12-year cruise with high sensitivity to epoch timings.
    *   Demonstrate that the chain solver enforces physical deflection limits for outer planets without silently accepting infeasible trajectories.
*   **Expected Outputs**: Convergence outcome is `False` (0 feasible out of 2000 evaluations) due to high sensitivity of the search space, unpowered flybys, and absence of intermediate correction maneuvers beyond a small budget.
*   **Runtime Characteristics**: $\approx 10 - 20$ seconds.
*   **Limitations & Simplifications**: Ignores continuous third-body perturbations of the giant planets and solar gravity during multi-year legs.

