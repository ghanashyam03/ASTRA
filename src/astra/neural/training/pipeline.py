"""Pipeline for generating training datasets from the physics kernel."""
from __future__ import annotations

import numpy as np

from astra.optimization.engine import evaluate_transfer
from astra.physics.kernel import PhysicsKernel
from astra.state.orbital_state import GM, CelestialBody


def generate_transfer_dataset(
    kernel: PhysicsKernel,
    origin_body: CelestialBody,
    destination_body: CelestialBody,
    dep_start: float,
    dep_end: float,
    tof_min: float,
    tof_max: float,
    n_samples: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate transfer samples using physical parameters to pretrain the neural network."""
    rng = np.random.default_rng(seed)
    
    # Generate uniform samples within bounds
    dep_epochs = rng.uniform(dep_start, dep_end, n_samples)
    tof_seconds = rng.uniform(tof_min, tof_max, n_samples)
    
    X = np.zeros((n_samples, 8), dtype=np.float32)
    dv_y = np.zeros(n_samples, dtype=np.float32)
    feas_y = np.zeros(n_samples, dtype=np.float32)
    
    mu_sun = GM["SUN"]
    
    for i in range(n_samples):
        dep = dep_epochs[i]
        tof = tof_seconds[i]
        
        # Build features
        X[i, 0] = float((dep - dep_start) / max(dep_end - dep_start, 1.0))
        X[i, 1] = float((tof - tof_min) / max(tof_max - tof_min, 1.0))
        # Placeholder planet positions (zeroed)
        X[i, 2:] = 0.0
        
        try:
            r1 = kernel.get_body_state(origin_body, dep).position
            v1 = kernel.get_body_state(origin_body, dep).velocity
            arr = dep + tof
            r2 = kernel.get_body_state(destination_body, arr).position
            v2 = kernel.get_body_state(destination_body, arr).velocity
            
            traj = evaluate_transfer(r1, v1, r2, v2, dep, tof, mu_sun)
            if traj is not None:
                dv_y[i] = float(traj.delta_v_total)
                feas_y[i] = 1.0 if traj.delta_v_total < 30.0 else 0.0
            else:
                dv_y[i] = 99.0
                feas_y[i] = 0.0
        except Exception:
            dv_y[i] = 99.0
            feas_y[i] = 0.0
            
    return X, dv_y, feas_y
