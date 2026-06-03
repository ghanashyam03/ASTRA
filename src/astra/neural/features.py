"""Physics-grounded geometric features for neural surrogate models.
All features must be computable WITHOUT calling Lambert (< 0.1ms each).
"""
from __future__ import annotations

import math

import numpy as np

from astra.state.orbital_state import GM

AU = 1.496e8  # km
MU_SUN = GM["SUN"]

def compute_hohmann_tof(r1_km: float, r2_km: float) -> float:
    """Approximate Hohmann transfer TOF [seconds] between circular orbits.
    T_Hohmann = π * sqrt((r1 + r2)³ / (8μ))
    """
    a_transfer = (r1_km + r2_km) / 2.0
    return math.pi * math.sqrt(a_transfer**3 / MU_SUN)

def compute_vis_viva_speed(r_km: float, a_km: float) -> float:
    """Speed on a Keplerian orbit [km/s]: v = sqrt(μ(2/r - 1/a))."""
    return math.sqrt(MU_SUN * (2.0 / r_km - 1.0 / a_km))

def build_geometric_features(
    dep_epoch: float,
    tof_seconds: float,
    r1_km: np.ndarray,           # origin position at departure
    v1_km_s: np.ndarray,         # origin velocity at departure
    r2_km: np.ndarray,           # destination position at arrival
    dep_epoch_min: float,
    dep_epoch_max: float,
    tof_min: float,
    tof_max: float,
    synodic_period_s: float,
) -> np.ndarray:
    """Compute 8-element geometric feature vector.

    Features:
      0: dep_epoch_normalized ∈ [0, 1]
      1: tof_normalized ∈ [0, 1]
      2: phase_angle_rad / π ∈ [0, 1]  (angle between r1 and r2 vectors)
      3: r1_AU ∈ [0, 1]               (origin distance in AU, normalized)
      4: r2_AU ∈ [0, 1]               (destination distance in AU, normalized)
      5: v_inf_rough / 10.0            (rough v_inf estimate, scaled)
      6: synodic_progress ∈ [0, 1]    (position in synodic cycle)
      7: tof_to_hohmann ∈ [0, 3]      (TOF / Hohmann_TOF, clipped)

    All features are physically meaningful and computable without Lambert.
    """
    dep_range = max(dep_epoch_max - dep_epoch_min, 1.0)
    tof_range = max(tof_max - tof_min, 1.0)
    feat_0 = (dep_epoch - dep_epoch_min) / dep_range

    feat_1 = (tof_seconds - tof_min) / tof_range

    # Phase angle between departure and arrival positions
    r1_norm = float(np.linalg.norm(r1_km))
    r2_norm = float(np.linalg.norm(r2_km))
    if r1_norm > 0 and r2_norm > 0:
        cos_phi = float(np.dot(r1_km, r2_km) / (r1_norm * r2_norm))
        cos_phi = max(-1.0, min(1.0, cos_phi))
        phase_angle = math.acos(cos_phi)
    else:
        phase_angle = 0.0
    feat_2 = phase_angle / math.pi  # normalized to [0, 1]

    # Orbital radii in AU, normalized onto approximately [0, 1] via clipping at 5.0 AU
    feat_3 = min(r1_norm / AU, 5.0) / 5.0
    feat_4 = min(r2_norm / AU, 5.0) / 5.0

    # Rough v_inf estimate: |v_transfer_at_r1 - v_body_at_r1|
    # Using vis-viva on the transfer ellipse: a_transfer ≈ (r1+r2)/2
    if r1_norm > 0 and r2_norm > 0:
        a_transfer = (r1_norm + r2_norm) / 2.0
        try:
            v_transfer = compute_vis_viva_speed(r1_norm, a_transfer)
            v_body = float(np.linalg.norm(v1_km_s))
            v_inf_rough = abs(v_transfer - v_body)
        except Exception:
            v_inf_rough = 0.0
    else:
        v_inf_rough = 0.0
    feat_5 = min(v_inf_rough / 10.0, 3.0)  # scale to ~[0, 3]

    # Synodic cycle progress: (dep_epoch % synodic_period) / synodic_period
    if synodic_period_s > 0:
        feat_6 = (dep_epoch % synodic_period_s) / synodic_period_s
    else:
        feat_6 = 0.0

    # TOF / Hohmann TOF ratio
    tof_hohmann = compute_hohmann_tof(r1_norm, r2_norm)
    if tof_hohmann > 0:
        feat_7 = min(tof_seconds / tof_hohmann, 4.0)
    else:
        feat_7 = 1.0

    return np.array([feat_0, feat_1, feat_2, feat_3, feat_4,
                     feat_5, feat_6, feat_7], dtype=np.float32)
