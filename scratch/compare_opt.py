import numpy as np
from astra.physics.kernel import PhysicsKernel
from astra.dsl import parse_mission_file, compile_mission
from astra.optimization.engine import evaluate_transfer
from astra.state.orbital_state import GM

kernel = PhysicsKernel().load()
dsl = parse_mission_file('data/benchmarks/earth_mars_2031.yaml')
m = compile_mission(dsl, kernel.ephemeris)

n_dep = 100
n_tof = 100
dep_epochs = np.linspace(m.departure_epoch_start, m.departure_epoch_end, n_dep)
tof_seconds = np.linspace(m.tof_min_seconds, m.tof_max_seconds, n_tof)

best_helio_dv = float('inf')
best_helio_traj = None
best_helio_dep = None
best_helio_tof = None

best_patched_dv = float('inf')
best_patched_traj = None
best_patched_dep = None
best_patched_tof = None

sun_mu = GM["SUN"]

for dep in dep_epochs:
    try:
        r1 = kernel.get_body_state(m.origin_body, dep).position
        v1 = kernel.get_body_state(m.origin_body, dep).velocity
    except Exception:
        continue
    for tof in tof_seconds:
        arr = dep + tof
        try:
            r2 = kernel.get_body_state(m.destination_body, arr).position
            v2 = kernel.get_body_state(m.destination_body, arr).velocity
        except Exception:
            continue
        
        # 1. Evaluate Heliocentric
        t_helio = evaluate_transfer(
            r1, v1, r2, v2, dep, tof, sun_mu,
            origin_body=m.origin_body.name,
            destination_body=m.destination_body.name,
            parking_altitude_km=m.parking_altitude_km,
            capture_altitude_km=m.capture_altitude_km,
            use_soi_patching=False
        )
        if t_helio is not None:
            dv_helio = t_helio.delta_v_total
            if dv_helio < best_helio_dv:
                best_helio_dv = dv_helio
                best_helio_traj = t_helio
                best_helio_dep = dep
                best_helio_tof = tof

        # 2. Evaluate Patched
        t_patched = evaluate_transfer(
            r1, v1, r2, v2, dep, tof, sun_mu,
            origin_body=m.origin_body.name,
            destination_body=m.destination_body.name,
            parking_altitude_km=m.parking_altitude_km,
            capture_altitude_km=m.capture_altitude_km,
            use_soi_patching=True
        )
        if t_patched is not None:
            dv_patched = t_patched.delta_v_total
            if dv_patched < best_patched_dv:
                best_patched_dv = dv_patched
                best_patched_traj = t_patched
                best_patched_dep = dep
                best_patched_tof = tof

print("=== BEST HELIOCENTRIC TRANSFER ===")
print(f"Departure Epoch: {best_helio_dep}")
print(f"TOF: {best_helio_tof / 86400.0:.2f} days")
print(f"Heliocentric Delta-v: {best_helio_dv:.4f} km/s")

# Evaluate this best heliocentric point with SOI patching to compare directly
r1_h = kernel.get_body_state(m.origin_body, best_helio_dep).position
v1_h = kernel.get_body_state(m.origin_body, best_helio_dep).velocity
r2_h = kernel.get_body_state(m.destination_body, best_helio_dep + best_helio_tof).position
v2_h = kernel.get_body_state(m.destination_body, best_helio_dep + best_helio_tof).velocity
t_helio_patched = evaluate_transfer(
    r1_h, v1_h, r2_h, v2_h, best_helio_dep, best_helio_tof, sun_mu,
    origin_body=m.origin_body.name,
    destination_body=m.destination_body.name,
    parking_altitude_km=m.parking_altitude_km,
    capture_altitude_km=m.capture_altitude_km,
    use_soi_patching=True
)
print("Same transfer evaluated with SOI patching:")
print(f"  TMI Delta-v (km/s): {t_helio_patched.metadata.get('dv1_km_s'):.4f}")
print(f"  MOI Delta-v (km/s): {t_helio_patched.metadata.get('dv2_km_s'):.4f}")
print(f"  Total Patched Delta-v: {t_helio_patched.delta_v_total:.4f} km/s")
print(f"  Departure C3 (km2/s2): {t_helio_patched.metadata.get('c3_km2_s2'):.4f}")

print("\n=== BEST PATCHED-CONICS (SOI) TRANSFER ===")
print(f"Departure Epoch: {best_patched_dep}")
print(f"TOF: {best_patched_tof / 86400.0:.2f} days")
print(f"Total Patched Delta-v: {best_patched_dv:.4f} km/s")
print(f"  TMI Delta-v (km/s): {best_patched_traj.metadata.get('dv1_km_s'):.4f}")
print(f"  MOI Delta-v (km/s): {best_patched_traj.metadata.get('dv2_km_s'):.4f}")
print(f"  Departure C3 (km2/s2): {best_patched_traj.metadata.get('c3_km2_s2'):.4f}")
print(f"  Departure v_inf (km/s): {best_patched_traj.metadata.get('v_inf_dep_km_s'):.4f}")
print(f"  Arrival v_inf (km/s): {best_patched_traj.metadata.get('v_inf_arr_km_s'):.4f}")
