"""Pipeline for generating training datasets from the physics kernel."""
from __future__ import annotations

import numpy as np

from astra.explainability.window_rationale import compute_synodic_period
from astra.neural.features import build_geometric_features
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
    
    # Precompute synodic period in seconds
    syn_days = compute_synodic_period(origin_body, destination_body)
    synodic_period_s = syn_days * 86400.0 if syn_days != float("inf") else 0.0
    
    for i in range(n_samples):
        dep = dep_epochs[i]
        tof = tof_seconds[i]
        
        try:
            r1_state = kernel.get_body_state(origin_body, dep)
            r1 = r1_state.position
            v1 = r1_state.velocity
            arr = dep + tof
            r2_state = kernel.get_body_state(destination_body, arr)
            r2 = r2_state.position
            v2 = r2_state.velocity
            
            # Build geometric features instead of empty placeholders
            X[i] = build_geometric_features(
                dep_epoch=dep,
                tof_seconds=tof,
                r1_km=r1,
                v1_km_s=v1,
                r2_km=r2,
                dep_epoch_min=dep_start,
                dep_epoch_max=dep_end,
                tof_min=tof_min,
                tof_max=tof_max,
                synodic_period_s=synodic_period_s,
            )
            
            traj = evaluate_transfer(
                r1, v1, r2, v2, dep, tof, mu_sun,
                origin_body=origin_body.value,
                destination_body=destination_body.value,
            )
            if traj is not None:
                dv_y[i] = float(traj.delta_v_total)
                feas_y[i] = 1.0 if traj.delta_v_total < 15.0 else 0.0
            else:
                dv_y[i] = 99.0
                feas_y[i] = 0.0
        except Exception:
            dv_y[i] = 99.0
            feas_y[i] = 0.0
            
    return X, dv_y, feas_y


def generate_pinn_dataset(
    kernel: PhysicsKernel,
    origin_body: CelestialBody,
    destination_body: CelestialBody,
    dep_start: float,
    dep_end: float,
    tof_min: float,
    tof_max: float,
    n_samples: int = 1000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate training data specifically for LambertPINN regression.
    
    Returns:
      X: (N, 8) geometric features float32
      dv_y: (N,) total Δv in km/s float32 (99.0 for infeasible)
      r1_norms: (N,) departure body radius from Sun [km] float32
      r2_norms: (N,) arrival body radius from Sun [km] float32
      tof_seconds: (N,) time of flight [seconds] float32
    
    This is different from generate_transfer_dataset() in that it also
    returns the orbital geometry needed for PINN physics residual computation.
    """
    from astra.explainability.window_rationale import compute_synodic_period
    from astra.neural.features import build_geometric_features
    
    rng = np.random.default_rng(seed)
    mu_sun = GM["SUN"]
    dep_epochs = rng.uniform(dep_start, dep_end, n_samples)
    tofs = rng.uniform(tof_min, tof_max, n_samples)
    
    syn_days = compute_synodic_period(origin_body, destination_body)
    synodic_s = syn_days * 86400.0 if syn_days != float("inf") else 0.0
    
    X = np.zeros((n_samples, 8), dtype=np.float32)
    dv_y = np.full(n_samples, 99.0, dtype=np.float32)
    r1_norms = np.zeros(n_samples, dtype=np.float32)
    r2_norms = np.zeros(n_samples, dtype=np.float32)
    tof_out = tofs.astype(np.float32)
    
    for i in range(n_samples):
        dep = dep_epochs[i]
        tof = tofs[i]
        try:
            r1_state = kernel.get_body_state(origin_body, dep)
            r2_state = kernel.get_body_state(destination_body, dep + tof)
            r1 = r1_state.position
            v1 = r1_state.velocity
            r2 = r2_state.position
            v2 = r2_state.velocity
            
            X[i] = build_geometric_features(
                dep, tof, r1, v1, r2,
                dep_start, dep_end, tof_min, tof_max, synodic_s
            )
            r1_norms[i] = float(np.linalg.norm(r1))
            r2_norms[i] = float(np.linalg.norm(r2))
            
            from astra.optimization.engine import evaluate_transfer
            traj = evaluate_transfer(
                r1, v1, r2, v2, dep, tof, mu_sun,
                origin_body=origin_body.name,
                destination_body=destination_body.name,
            )
            if traj is not None:
                dv_y[i] = float(traj.delta_v_total)
        except Exception:
            pass
    
    return X, dv_y, r1_norms, r2_norms, tof_out

