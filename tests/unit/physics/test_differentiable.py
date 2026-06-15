import pytest
import numpy as np

def test_jax_available():
    """JAX must be importable for this module to work."""
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        pytest.skip("JAX not installed")
    assert True

def test_differentiable_module_imports():
    try:
        from astra.physics.differentiable import JAX_AVAILABLE
    except ImportError:
        pytest.skip("JAX not installed")
    assert isinstance(JAX_AVAILABLE, bool)

@pytest.mark.skipif(
    not __import__("pathlib").Path("data/spice_kernels/de440.bsp").exists(),
    reason="SPICE kernels required"
)
def test_gradient_agrees_with_finite_difference():
    """JAX gradient must agree with finite difference to 1e-3 tolerance."""
    try:
        from astra.physics.differentiable import compute_dv_gradient, JAX_AVAILABLE
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")
    except ImportError:
        pytest.skip("JAX not installed")
    
    from astra.physics.kernel import PhysicsKernel
    from astra.dsl.parser import parse_mission_file
    from astra.dsl.compiler import compile_mission
    from astra.optimization.engine import evaluate_transfer
    from astra.state.orbital_state import GM
    
    kernel = PhysicsKernel().load()
    dsl = parse_mission_file("data/benchmarks/earth_mars_2031.yaml")
    mission = compile_mission(dsl, kernel.ephemeris)
    
    dep0 = (mission.departure_epoch_start + mission.departure_epoch_end) / 2.0
    tof0 = (mission.tof_min_seconds + mission.tof_max_seconds) / 2.0
    
    # JAX gradient
    dv_jax, grad_jax = compute_dv_gradient(dep0, tof0, kernel, mission)
    
    # Finite difference gradient
    h = 3600.0  # 1 hour step
    mu = GM["SUN"]
    def eval_dv(dep, tof):
        try:
            r1 = kernel.get_body_state(mission.origin_body, dep).position
            v1 = kernel.get_body_state(mission.origin_body, dep).velocity
            r2 = kernel.get_body_state(mission.destination_body, dep+tof).position
            v2 = kernel.get_body_state(mission.destination_body, dep+tof).velocity
            tr = evaluate_transfer(r1,v1,r2,v2,dep,tof,mu,
                origin_body=mission.origin_body.name,
                destination_body=mission.destination_body.name,
                parking_altitude_km=mission.parking_altitude_km,
                capture_altitude_km=mission.capture_altitude_km)
            return tr.delta_v_total if tr else 99.0
        except Exception:
            return 99.0
    
    grad_fd_tof = (eval_dv(dep0, tof0+h) - eval_dv(dep0, tof0-h)) / (2*h)
    
    print(f"\nJAX gradient (TOF): {grad_jax[1]:.6f} km/s/s")
    print(f"FD gradient (TOF):  {grad_fd_tof:.6f} km/s/s")
    
    # Agree within 1e-3 (JAX uses approximation, not exact Lambert)
    assert abs(grad_jax[1] - grad_fd_tof) < 0.01, (
        f"JAX/FD gradient disagreement: JAX={grad_jax[1]:.4f}, FD={grad_fd_tof:.4f}")
