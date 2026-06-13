"""Graph Neural Network policy model for ASTRA flyby sequence planning.

Models the solar system as a directed graph where nodes are celestial bodies
and edges are candidate transfer legs. Implements 2-layer message passing
with mean aggregation. Pure NumPy — no deep learning frameworks.

This is a POLICY model, not a trajectory surrogate. It scores candidate next
bodies for MCTS rollouts, replacing uniform random selection with learned
heuristics. It does NOT implement the NeuralSurrogate interface.

Every trajectory suggested by the GNN policy is validated by the Lambert
solver and SOI patching before acceptance. The GNN only changes the ORDER
in which MCTS explores candidates — it cannot bypass physics validation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from astra.physics.flyby import SAFE_FLYBY_ALTITUDE_KM
from astra.state.orbital_state import GM, PHYSICAL_RADIUS, CelestialBody

# --- CONSTANTS ---

PLANET_NODES = ["MERCURY", "VENUS", "EARTH", "MOON", "MARS", "JUPITER", "SATURN"]
NODE_INDEX: dict[str, int] = {name: i for i, name in enumerate(PLANET_NODES)}
N_NODES = 7
NODE_FEATURE_DIM = 8
EDGE_FEATURE_DIM = 6

ORBITAL_PERIODS_DAYS: dict[str, float] = {
    "MERCURY": 87.97, "VENUS": 224.70, "EARTH": 365.25,
    "MOON": 27.32, "MARS": 686.97, "JUPITER": 4332.6, "SATURN": 10759.2,
}

SEMI_MAJOR_AXES_AU: dict[str, float] = {
    "MERCURY": 0.387, "VENUS": 0.723, "EARTH": 1.000,
    "MOON": 0.00257, "MARS": 1.524, "JUPITER": 5.203, "SATURN": 9.537,
}

# --- NODE FEATURE CONSTRUCTION ---

def build_node_features() -> np.ndarray:
    """Build (N_NODES, NODE_FEATURE_DIM) float32 feature matrix.
    Features per node:
      [0] orbital_period_normalized: period_days / max_period_days, clipped [0,1]
      [1] semi_major_axis_normalized: a_AU / 10.0, clipped [0,1]
      [2] log_mu_normalized: log10(GM[body]) / 11.5 (Sun GM log10 ~11.1), clipped [0,1]
      [3] physical_radius_normalized: R_km / 71492.0 (Jupiter radius), clipped [0,1]
      [4] is_inner_solar_system: 1.0 if a_AU < 2.0 (inside asteroid belt) else 0.0
      [5] has_atmosphere: 1.0 for VENUS, EARTH, MARS, JUPITER, SATURN else 0.0
      [6] is_moon: 1.0 for MOON else 0.0
      [7] safe_flyby_alt_normalized: SAFE_FLYBY_ALTITUDE_KM / 500.0, clipped [0,1]
    Deterministic — same result every call, seed-independent."""
    max_period = max(ORBITAL_PERIODS_DAYS.values())
    HAS_ATMOSPHERE = {"VENUS", "EARTH", "MARS", "JUPITER", "SATURN"}
    features = np.zeros((N_NODES, NODE_FEATURE_DIM), dtype=np.float32)
    for i, body in enumerate(PLANET_NODES):
        cb = CelestialBody[body]
        period = ORBITAL_PERIODS_DAYS[body]
        a_au = SEMI_MAJOR_AXES_AU[body]
        mu = GM.get(body, GM["EARTH"])
        R = PHYSICAL_RADIUS.get(cb, 6371.0)
        safe_alt = SAFE_FLYBY_ALTITUDE_KM.get(body, 300.0)
        features[i, 0] = min(period / max_period, 1.0)
        features[i, 1] = min(a_au / 10.0, 1.0)
        features[i, 2] = min(math.log10(max(mu, 1.0)) / 11.5, 1.0)
        features[i, 3] = min(R / 71492.0, 1.0)
        features[i, 4] = 1.0 if a_au < 2.0 else 0.0
        features[i, 5] = 1.0 if body in HAS_ATMOSPHERE else 0.0
        features[i, 6] = 1.0 if body == "MOON" else 0.0
        features[i, 7] = min(safe_alt / 500.0, 1.0)
    return features

# --- EDGE FEATURE CONSTRUCTION ---

def build_edge_features(
    origin: str,
    destination: str,
    r_origin_km: np.ndarray,
    r_dest_km: np.ndarray,
    v_origin_km_s: np.ndarray,
    tof_seconds: float,
) -> np.ndarray:
    """Build (EDGE_FEATURE_DIM,) float32 feature vector for a transfer leg.
    Features:
      [0] synodic_period_normalized: synodic_days / 800.0, clipped [0,1]
      [1] delta_r_au_normalized: |r_dest-r_origin| / (10*AU), clipped [0,1]
      [2] phase_angle_normalized: arccos(dot(r1_hat, r2_hat)) / pi
      [3] vinf_estimate_normalized: rough_vinf_km_s / 10.0, clipped [0,3]
      [4] tof_to_hohmann_ratio: tof / T_hohmann, clipped [0,4]
      [5] mu_ratio_normalized: log10(GM_dest/GM_origin) / 4.0 + 0.5, clipped [0,1]
    """
    from astra.explainability.window_rationale import compute_synodic_period
    from astra.neural.features import compute_hohmann_tof, compute_vis_viva_speed

    AU = 1.496e8
    cb_origin = CelestialBody[origin]
    cb_dest = CelestialBody[destination]
    syn_days = compute_synodic_period(cb_origin, cb_dest)
    synodic_norm = min(syn_days / 800.0, 1.0) if syn_days != float("inf") else 1.0
    
    r1n = float(np.linalg.norm(r_origin_km))
    r2n = float(np.linalg.norm(r_dest_km))
    delta_r = float(np.linalg.norm(r_dest_km - r_origin_km))
    delta_r_norm = min(delta_r / (10.0 * AU), 1.0)

    cos_phi = float(np.dot(r_origin_km, r_dest_km) / (max(r1n, 1e-10) * max(r2n, 1e-10)))
    cos_phi = max(-1.0, min(1.0, cos_phi))
    phase_norm = math.acos(cos_phi) / math.pi

    vinf_est = 0.0
    if r1n > 0 and r2n > 0:
        try:
            a_t = (r1n + r2n) / 2.0
            v_t = compute_vis_viva_speed(r1n, a_t)
            v_b = float(np.linalg.norm(v_origin_km_s))
            vinf_est = abs(v_t - v_b)
        except Exception:
            pass
    vinf_norm = min(vinf_est / 10.0, 3.0)

    tof_hoh = compute_hohmann_tof(max(r1n, 1.0), max(r2n, 1.0))
    tof_ratio = min(tof_seconds / max(tof_hoh, 1.0), 4.0)

    mu_orig = max(GM.get(origin, 3.986e5), 1.0)
    mu_dest = max(GM.get(destination, 3.986e5), 1.0)
    mu_ratio = min(max(math.log10(mu_dest / mu_orig) / 4.0 + 0.5, 0.0), 1.0)

    return np.array([synodic_norm, delta_r_norm, phase_norm,
                     vinf_norm, tof_ratio, mu_ratio], dtype=np.float32)

# --- GRAPH-BASED POLICY MODEL ARCHITECTURE ---

class SolarSystemGNN:
    """2-layer message passing graph-based policy model with mean aggregation.
    Input: node features + edge features.
    Output: scalar score per node (policy logit for MCTS rollout).

    Note: Only the output head (W_out, b_out) is trained; message-passing
    layers use frozen random projections (reservoir style).
    """

    def __init__(
        self,
        node_dim: int = NODE_FEATURE_DIM,    # 8
        edge_dim: int = EDGE_FEATURE_DIM,    # 6
        hidden_dim: int = 32,
        lr: float = 1e-3,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)

        def he_init(in_d: int, out_d: int) -> np.ndarray:
            return (rng.standard_normal((in_d, out_d))
                    * math.sqrt(2.0 / in_d)).astype(np.float32)

        # Layer 1: message network (node_j + edge) → hidden (frozen)
        self.W_msg1 = he_init(node_dim + edge_dim, hidden_dim)  # (14, 32)
        self.b_msg1 = np.zeros((1, hidden_dim), dtype=np.float32)
        # Layer 1: update network (node_i + aggregated_msg) → hidden (frozen)
        self.W_upd1 = he_init(node_dim + hidden_dim, hidden_dim)  # (40, 32)
        self.b_upd1 = np.zeros((1, hidden_dim), dtype=np.float32)
        # Layer 2: message network (frozen)
        self.W_msg2 = he_init(hidden_dim + edge_dim, hidden_dim)  # (38, 32)
        self.b_msg2 = np.zeros((1, hidden_dim), dtype=np.float32)
        # Layer 2: update network (frozen)
        self.W_upd2 = he_init(hidden_dim + hidden_dim, hidden_dim)  # (64, 32)
        self.b_upd2 = np.zeros((1, hidden_dim), dtype=np.float32)
        # Output policy head (trainable)
        self.W_out = he_init(hidden_dim, 1)  # (32, 1)
        self.b_out = np.zeros((1, 1), dtype=np.float32)

        self.hidden_dim = hidden_dim
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.lr = lr
        self._trained = False

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)  # type: ignore[no-any-return]

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))  # type: ignore[no-any-return]

    def message_passing(
        self,
        node_features: np.ndarray,
        edge_dict: dict[tuple[int, int], np.ndarray],
    ) -> np.ndarray:
        """Run 2-layer message passing.
        node_features: (N_NODES, node_dim)
        edge_dict: maps (src_idx, dst_idx) → (edge_dim,) feature vector
        Returns: (N_NODES, hidden_dim) updated node embeddings.
        """
        h = node_features.astype(np.float32)

        for layer in range(2):
            W_msg = self.W_msg1 if layer == 0 else self.W_msg2
            b_msg = self.b_msg1 if layer == 0 else self.b_msg2
            W_upd = self.W_upd1 if layer == 0 else self.W_upd2
            b_upd = self.b_upd1 if layer == 0 else self.b_upd2
            h_new = np.zeros((N_NODES, self.hidden_dim), dtype=np.float32)

            for dst in range(N_NODES):
                incoming_msgs = []
                for src in range(N_NODES):
                    if src == dst:
                        continue
                    edge_key = (src, dst)
                    if edge_key not in edge_dict:
                        continue
                    e_feat = edge_dict[edge_key]
                    msg_input = np.concatenate([h[src], e_feat]).reshape(1, -1)
                    msg = self._relu(msg_input @ W_msg + b_msg)
                    incoming_msgs.append(msg[0])
                if not incoming_msgs:
                    agg = np.zeros(self.hidden_dim, dtype=np.float32)
                else:
                    agg = np.mean(incoming_msgs, axis=0)
                upd_input = np.concatenate([h[dst], agg]).reshape(1, -1)
                h_new[dst] = self._relu(upd_input @ W_upd + b_upd)[0]
            h = h_new

        return h

    def predict_scores(
        self,
        current_body: str,
        candidate_bodies: list[str],
        node_features: np.ndarray,
        edge_dict: dict[tuple[int, int], np.ndarray],
    ) -> np.ndarray:
        """Score each candidate body as next flyby target.
        Returns (len(candidate_bodies),) float32 scores in [0, 1].
        Higher score = more promising next body to visit."""
        h = self.message_passing(node_features, edge_dict)
        src_idx = NODE_INDEX.get(current_body)
        scores = np.zeros(len(candidate_bodies), dtype=np.float32)
        for k, body in enumerate(candidate_bodies):
            dst_idx = NODE_INDEX.get(body)
            if src_idx is None or dst_idx is None:
                scores[k] = 0.5
                continue
            embedding = h[src_idx] + h[dst_idx]
            # Avoid warnings or dimension mismatch: reshape embedding to 2D
            embedding_2d = embedding.reshape(1, -1)
            raw_score = float((embedding_2d @ self.W_out + self.b_out)[0, 0])
            scores[k] = float(self._sigmoid(np.array([[raw_score]]))[0, 0])
        return scores

    def train_on_mcts_results(
        self,
        mcts_paths: list[list[Any]],    # list of PhaseState sequences
        rewards: list[float],       # reward per sequence
        node_features: np.ndarray,
        kernel: Any,             # PhysicsKernel for body state queries
        epochs: int = 20,
        seed: int = 42,
    ) -> list[float]:
        """Train output head weights of graph-based policy model on MCTS results using REINFORCE.
        For each transition in each sequence, compute edge features from live
        planet positions, then update weights to increase/decrease scores
        based on the sequence reward.
        Returns per-epoch mean loss."""
        rng = np.random.default_rng(seed)
        loss_history: list[float] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_updates = 0
            indices = rng.permutation(len(mcts_paths))
            for idx in indices:
                path = mcts_paths[idx]
                reward = float(rewards[idx])
                label = 1.0 if reward > 0.3 else 0.0
                if len(path) < 2:
                    continue
                for step in range(len(path) - 1):
                    phase_curr = path[step]
                    phase_next = path[step + 1]
                    curr_body = phase_curr.body
                    next_body = phase_next.body
                    if curr_body not in NODE_INDEX or next_body not in NODE_INDEX:
                        continue
                    # Build edge dict from live planet states at this epoch
                    edge_dict: dict[tuple[int, int], np.ndarray] = {}
                    try:
                        from astra.state.orbital_state import CelestialBody
                        cb_curr = CelestialBody[curr_body]
                        cb_next = CelestialBody[next_body]
                        r1 = kernel.get_body_state(cb_curr, phase_curr.epoch).position
                        v1 = kernel.get_body_state(cb_curr, phase_curr.epoch).velocity
                        r2 = kernel.get_body_state(cb_next, phase_next.epoch).position
                        tof = phase_next.epoch - phase_curr.epoch
                        src_i = NODE_INDEX[curr_body]
                        dst_i = NODE_INDEX[next_body]
                        edge_dict[(src_i, dst_i)] = build_edge_features(
                            curr_body, next_body, r1, r2, v1, tof
                        )
                    except Exception:
                        continue
                    # Forward
                    h = self.message_passing(node_features, edge_dict)
                    src_idx_n = NODE_INDEX[curr_body]
                    dst_idx_n = NODE_INDEX[next_body]
                    emb = h[src_idx_n] + h[dst_idx_n]
                    emb_2d = emb.reshape(1, -1)
                    raw = float((emb_2d @ self.W_out + self.b_out)[0, 0])
                    pred = float(self._sigmoid(np.array([[raw]]))[0, 0])
                    # Binary cross-entropy loss
                    eps = 1e-7
                    loss = -(label * math.log(pred + eps)
                             + (1 - label) * math.log(1 - pred + eps))
                    epoch_loss += loss
                    n_updates += 1
                    # Gradient of output layer only (simplified BPTT)
                    d_pred = pred - label
                    self.W_out -= self.lr * np.outer(emb, [d_pred]).astype(np.float32)
                    self.b_out -= self.lr * np.array([[d_pred]], dtype=np.float32)
            loss_history.append(epoch_loss / max(n_updates, 1))
        self._trained = True
        return loss_history

    def is_trained(self) -> bool:
        return self._trained
