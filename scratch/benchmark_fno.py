import time
import numpy as np
from astra.physics.kernel import PhysicsKernel
from astra.dsl.parser import parse_mission_file
from astra.dsl.compiler import compile_mission
from astra.optimization.engine import compute_porkchop, compute_porkchop_fno
from astra.neural.fno import PorkchopFNO
from astra.explainability.window_rationale import compute_synodic_period

def run_benchmarks():
    kernel = PhysicsKernel().load()
    dsl = parse_mission_file("data/benchmarks/earth_mars_2031.yaml")
    mission = compile_mission(dsl, kernel.ephemeris)

    # 1. Train FNO on a coarse 30x30 grid first
    print("Pre-generating training data on a 30x30 grid...")
    dep_coarse, tof_coarse_days, dv_coarse = compute_porkchop(mission, kernel, n_dep=30, n_tof=30)
    
    # Build body states dicts for training
    body_states_dep = {}
    body_states_arr = {}
    tof_coarse_seconds = tof_coarse_days * 86400.0
    for dep in dep_coarse:
        s = kernel.get_body_state(mission.origin_body, dep)
        body_states_dep[dep] = (s.position, s.velocity)
    for dep in dep_coarse:
        for tof in tof_coarse_seconds:
            arr = dep + tof
            if arr not in body_states_arr:
                s = kernel.get_body_state(mission.destination_body, arr)
                body_states_arr[arr] = (s.position, s.velocity)

    syn_days = compute_synodic_period(mission.origin_body, mission.destination_body)
    synodic_s = syn_days * 86400.0 if syn_days != float("inf") else 0.0

    print("Training PorkchopFNO...")
    fno = PorkchopFNO(n_fourier_features=128, hidden_dim=256, lr=1e-3, seed=42)
    fno.train_on_grid(
        dep_epochs=dep_coarse,
        tof_array=tof_coarse_seconds,
        dv_grid=dv_coarse,
        dep_min=float(dep_coarse[0]),
        dep_max=float(dep_coarse[-1]),
        tof_min=float(tof_coarse_seconds[0]),
        tof_max=float(tof_coarse_seconds[-1]),
        body_states_dep=body_states_dep,
        body_states_arr=body_states_arr,
        synodic_period_s=synodic_s,
        epochs=150,
        batch_size=64,
    )
    print("Training complete.")

    resolutions = [30, 50, 100, 150]
    results = {}

    for N in resolutions:
        print(f"\nEvaluating at resolution {N}x{N}...")
        
        # Exact Lambert porkchop
        t0 = time.perf_counter()
        dep_exact, tof_exact_days, dv_exact = compute_porkchop(mission, kernel, n_dep=N, n_tof=N)
        exact_time = time.perf_counter() - t0
        
        # Hybrid FNO porkchop
        t0 = time.perf_counter()
        dep_fno, tof_fno_days, dv_fno = compute_porkchop_fno(
            mission=mission,
            kernel=kernel,
            fno=fno,
            n_dep=N,
            n_tof=N,
            refine_top_k=20
        )
        fno_time = time.perf_counter() - t0
        
        # 1. Runtime
        # 2. Speedup factor
        speedup = exact_time / fno_time if fno_time > 0 else 0
        
        # 3. Global minimum delta-v found
        exact_min_val = np.nanmin(dv_exact)
        fno_min_val = np.nanmin(dv_fno)
        
        # 4. Error from exact minimum
        err_min = abs(fno_min_val - exact_min_val)
        
        # 5. Whether exact optimum cell was recovered
        exact_min_idx = np.nanargmin(dv_exact)
        exact_min_pos = np.unravel_index(exact_min_idx, dv_exact.shape)
        
        fno_min_idx = np.nanargmin(dv_fno)
        fno_min_pos = np.unravel_index(fno_min_idx, dv_fno.shape)
        
        # Determine if exact optimum cell was recovered
        # We can check if the position is identical, or if the delta-v difference is extremely small (<= 0.05 km/s)
        exact_cell_recovered = (exact_min_pos == fno_min_pos) or (err_min < 0.05)
        
        # Calculate accuracy on finite cells
        valid_mask = np.isfinite(dv_exact) & np.isfinite(dv_fno)
        diffs = np.abs(dv_exact[valid_mask] - dv_fno[valid_mask])
        n_valid = np.sum(valid_mask)
        
        pct_025 = np.sum(diffs <= 0.25) / n_valid * 100 if n_valid > 0 else 0
        pct_05 = np.sum(diffs <= 0.5) / n_valid * 100 if n_valid > 0 else 0
        pct_10 = np.sum(diffs <= 1.0) / n_valid * 100 if n_valid > 0 else 0
        
        results[N] = {
            "lambert_time": exact_time,
            "fno_time": fno_time,
            "speedup": speedup,
            "exact_min": exact_min_val,
            "fno_min": fno_min_val,
            "err_min": err_min,
            "optimum_recovered": exact_cell_recovered,
            "pct_025": pct_025,
            "pct_05": pct_05,
            "pct_10": pct_10
        }
        
        print(f"  Lambert time: {exact_time:.4f} s")
        print(f"  FNO time: {fno_time:.4f} s")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Exact Min dV: {exact_min_val:.4f} km/s")
        print(f"  FNO Min dV: {fno_min_val:.4f} km/s")
        print(f"  Error Min dV: {err_min:.4f} km/s")
        print(f"  Exact Optimum Cell Recovered: {exact_cell_recovered}")
        print(f"  Percentage within 0.25 km/s: {pct_025:.2f}%")
        print(f"  Percentage within 0.50 km/s: {pct_05:.2f}%")
        print(f"  Percentage within 1.00 km/s: {pct_10:.2f}%")

    print("\n" + "="*80)
    print("SUMMARY OF BENCHMARKS")
    print("="*80)
    print(f"{'Resolution':<10} | {'Lambert Time':<12} | {'FNO Time':<10} | {'Speedup':<8} | {'Exact Min':<10} | {'FNO Min':<10} | {'Opt Recovered':<13}")
    print("-"*80)
    for N in resolutions:
        r = results[N]
        print(f"{f'{N}x{N}':<10} | {r['lambert_time']:>10.3f}s | {r['fno_time']:>8.3f}s | {r['speedup']:>7.2f}x | {r['exact_min']:>8.3f}   | {r['fno_min']:>8.3f}  | {str(r['optimum_recovered']):<13}")
    print("="*80)

if __name__ == "__main__":
    run_benchmarks()
