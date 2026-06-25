# Voyager 2 Grand Tour — Divergence Characterization

## Outcome
The optimization run did not converge. Out of 2000 evaluations, the solver found 0 feasible chains and 2000 rejected chains (`converged=False`, `n_evaluations=2000`, `n_feasible=0`).

## Candidate explanation 1: Δv-only optimization vs. NASA's multi-constraint choice
This explanation points to the divergence between the objective of a simple mathematical optimizer (minimizing total delta-V only) and the actual multi-objective constraint profile of the historical Voyager 2 mission. 

NASA's trajectory selection was constrained by:
1. **Launch Vehicle Limits:** The Titan IIIE-Centaur launch vehicle dictated strict limits on launch energy ($C_3$), which bounded the departure velocities.
2. **Scientific Objectives and Instruments:** Flight geometries were selected to enable specific imaging angles, close approaches to moons (e.g., Triton at Neptune, Titan at Saturn), and to measure magnetic fields and planetary atmospheres.
3. **Radiation and Ring Safety:** Voyager 2 had to pass outside Jupiter's lethal radiation belts and avoid Saturn's rings, which restricted the range of safe periapsis altitudes.
4. **Comm and Tracking Constraints:** Continuous line-of-sight and antenna pointing margins back to Earth had to be maintained.

Because ASTRA's optimizer minimizes delta-V only, it does not target these scientific/operational constraints, which would lead to a different trajectory design even if the search space was successfully resolved. However, since the solver produced 0 feasible trajectories, this explanation alone does not account for the absolute lack of convergence.

## Candidate explanation 2: optimizer found a different but locally-good chain
This explanation addresses whether the optimizer searched the historical window but converged to a different locally optimal solution. 

Because the optimizer returned 0 feasible chains out of 2000 trials, it did not find any alternative locally-good trajectory. The reasons for this search failure include:
1. **Chaotic Sensitivity (Dimensionality and Time):** The search space has 5 dimensions (departure epoch + 4 leg TOFs) over a 12-year mission duration. A minor shift in the arrival epoch at Jupiter changes the required outgoing velocity vector to Saturn. A tiny deviation at Earth or Jupiter cascades into massive errors at Uranus and Neptune. This chaotic sensitivity makes finding a valid chain like a needle in a multidimensional haystack.
2. **Deflection Limits:** With a modest, realistic DSM budget (0.2 km/s) and unpowered flybys, the planet alignments must be mathematically precise. Any slight epoch misalignment requires a turn angle that exceeds the gravity-assist deflection ceiling of the planets, triggering an immediate `impossible_geometry` rejection.
3. **No Resonant/Multi-Rev Transfers:** The search was limited to single-revolution conics (`max_revs_per_leg: 0`), preventing the optimizer from utilizing resonant or multi-revolution legs that could help bridge timing mismatches.

## Candidate explanation 3: genuine physics-fidelity limitation
This candidate explanation concerns whether the instantaneous flyby/patched-conics approximation introduces physical errors at this scale (e.g. ignoring continuous third-body perturbations of the giant planets and the Sun during long cruises).

This explanation is NOT assessed by this prompt and is the specific subject of Prompt 38. We do not speculate on physics-fidelity limits here.

## Conclusion
The evidence from this run supports **Candidate Explanation 2** as the primary driver for the lack of convergence: the extreme sensitivity of the 5-leg chain over a 12-year cruise makes the feasible region in the search space extremely small (virtually zero under a random/Bayesian search of 2000 trials). Candidate Explanation 1 represents a secondary design divergence (minimizing delta-V vs. satisfying scientific and safety constraints), but the search complexity (Explanation 2) is what structurally blocked the optimizer from finding any solution. Explanation 3 is deferred to Prompt 38's direct measurement of patched-conics fidelity limits.
