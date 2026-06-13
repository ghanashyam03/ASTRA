# Post-Mortem Analysis: Graph-Based Policy Model (SolarSystemGNN)

**Status:** Rejected / Abandoned (Kept in feature branch, not merged to main)  
**Date:** June 13, 2026  
**Author:** Antigravity Coding Assistant & ASTRA Team  

---

## Executive Summary
This document outlines the findings, benchmark data, and scientific analysis regarding the integration of `SolarSystemGNN` (a graph-based policy model) into the `MCTSPlanner` sequence-search algorithm. 

The goal of this feature was to replace standard uniform random action selection during MCTS rollouts with a learned policy that prioritizes promising planetary transfer legs. However, benchmarking showed **0.0% reduction in explored nodes** alongside a **105.0% increase in runtime overhead** for outer solar system missions. Consequently, this approach has been rejected for production and will remain in the feature branch for reference only.

---

## 1. What We Learnt

### I. The "Rollout Policy vs. Tree Expansion" Misalignment
In MCTS, tree nodes are expanded and visited during the UCB1 selection and expansion phases (`_select` and `_expand`). The rollout simulation phase (`_simulate`) only runs a single sample path down to `max_depth` to evaluate the terminal reward. 
- Guiding the rollout path changes the *reward* estimate propagating back, but it does **not** prune or restrict the nodes created in the tree structure.
- Consequently, node exploration counts remain identical because UCB1 still requires exploring the tree branches based on node visits and exploitation scores, rendering the GNN policy ineffective at search-space reduction.

### II. The "All-Positive Label" Training Collapse
The GNN was trained on successful paths extracted from MCTS results (`result.all_paths`). 
- By definition, every path in `all_paths` was validated and had a cumulative $\Delta v$ under the mission budget.
- This meant that nearly all training samples had rewards above the threshold ($r > 0.3$), generating a training label of `1.0` for almost every single action transition.
- Mathematically, when trained via REINFORCE with binary cross-entropy on only positive labels, the model's logits collapsed. The GNN simply learned to output a score of `~1.0` (high probability) for *every* candidate action, causing the ranking to degenerate back into the default sequence order.

### III. The Limits of Reservoir GNN Architectures
Because the GNN's 2-layer message-passing and node updating weights were frozen (randomly initialized via He initialization), no graph representation learning occurred. The model acted as a linear classifier (updating only the output weights $W_{out}$ and $b_{out}$) on static random projections. 
- Without updating the message-passing layers, the network was unable to learn to aggregate complex multi-body orbital mechanics representations dynamically.

### IV. The SPICE Ephemeris Lookup Bottleneck
Planetary positions and velocities are retrieved dynamically from the SPICE kernel (`PhysicsKernel.get_body_state`).
- Even after caching the origin body state `r1, v1` once per rollout step, the edge feature construction still required querying the destination state `r2` for *every candidate action* to calculate delta positions, phase angles, and time-of-flight ratios.
- For a deep search (e.g., Earth-Jupiter flybys) with multiple actions, the number of ephemeris requests scaled by orders of magnitude, doubling the search runtime.

---

## 2. Benchmark Evidence

Comparing **Standard MCTS** (uniform shuffles) vs. **GNN-Guided MCTS** (graph-based policy) with 150–200 search iterations:

| Mission | Std Nodes Explored | GNN Nodes Explored | Node Reduction (%) | Std Runtime (s) | GNN Runtime (s) | Runtime Overhead (%) |
|---|---|---|---|---|---|---|
| **Earth → Mars** | 7 | 7 | 0.0% | 0.002s | 0.001s | -30.8% |
| **Earth → Venus → Mars** | 49 | 49 | 0.0% | 0.016s | 0.019s | +17.8% |
| **Earth → Jupiter** | 141 | 141 | 0.0% | 0.046s | 0.095s | +105.0% |

### Key Observations:
- **Zero Pruning**: Node exploration count was completely unchanged.
- **Significant Slowdown**: For interplanetary missions (e.g. Jupiter) where multiple flyby combinations are evaluated, the GNN integration increased runtime by **+105.0%** due to feature construction and ephemeris querying.

---

## 3. Why the Approach Failed (Physics & Mathematical Breakdown)

1. **Patched-Conics Filtering is already an efficient heuristic**: In ASTRA, infeasible trajectories (e.g. negative periapsis altitudes or exceeding maximum flight durations) are immediately discarded by physics-based filters (`_apply_action`). These quick, physics-based checks act as a robust pruning mechanism. Adding a neural classifier to rank them adds overhead without improving on these hard boundaries.
2. **Missing Negative Samples**: A policy model cannot learn a gradient boundary without negative samples (failed trajectories). Since MCTS rollouts only return terminal rewards, transitions inside bad paths were not systematically penalized with `0.0` labels unless the entire sequence failed, which is a very noisy reinforcement learning signal.
3. **No Temporal Graph Representation**: The solar system is a dynamic graph where edges (transfer windows) exist only at specific epochs. The static node features (semimajor axis, period) and the single-step edge features computed at a specific epoch were insufficient to capture the resonant alignment of planetary transfer windows over long time scales.

---

## 4. Alternative Directions
For future optimization:
1. **Value Surrogates**: Rather than a policy network to order actions, use neural surrogates (such as PINNs or FNOs) to directly predict/estimate the $\Delta v$ cost of a transfer leg to bypass the Lambert solver entirely. (This is already supported in MCTS via the `surrogate` parameter).
2. **Offline Pre-Pruning**: Use synodic window rationales to compute forbidden search states offline, modifying the MCTS action generator (`_get_valid_actions`) directly rather than checking them during tree search.
