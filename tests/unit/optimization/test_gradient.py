import numpy as np
from astra.optimization.gradient import refine_trajectory_lbfgsb

def test_refinement_improves_simple_quadratic() -> None:
    """L-BFGS-B must find minimum of x² + y² starting from (1, 1)."""
    def quadratic(x: np.ndarray) -> float:
        return float(x[0]**2 + x[1]**2)
    result = refine_trajectory_lbfgsb(
        quadratic,
        x0=np.array([1.0, 1.0]),
        bounds=[(-10.0, 10.0), (-10.0, 10.0)],
    )
    assert result.converged
    assert result.f_refined < 1e-6
    assert result.improvement_km_s > 0.99

def test_refinement_respects_bounds() -> None:
    """Refined solution must stay within provided bounds."""
    def valley(x: np.ndarray) -> float:
        return float((x[0] - 5.0)**2 + x[1]**2)
    result = refine_trajectory_lbfgsb(
        valley,
        x0=np.array([0.0, 0.0]),
        bounds=[(0.0, 3.0), (-1.0, 1.0)],  # True minimum at x=5 is outside bounds
    )
    assert 0.0 <= result.x_refined[0] <= 3.0
    assert -1.0 <= result.x_refined[1] <= 1.0

def test_refinement_handles_infeasible_start() -> None:
    """Starting from an infeasible region (f=99) must not crash."""
    def always_infeasible(x: np.ndarray) -> float:
        return 99.0
    result = refine_trajectory_lbfgsb(
        always_infeasible,
        x0=np.array([0.0, 0.0]),
        bounds=[(-1.0, 1.0), (-1.0, 1.0)],
    )
    assert result.improvement_km_s == 0.0
    # Should not raise, result.converged may be False

def test_refinement_result_has_all_fields() -> None:
    def obj(x: np.ndarray) -> float:
        return float(x[0]**2)
    result = refine_trajectory_lbfgsb(obj, np.array([2.0]), [(-10.0, 10.0)])
    assert hasattr(result, "x_refined")
    assert hasattr(result, "f_refined")
    assert hasattr(result, "improvement_km_s")
    assert hasattr(result, "n_evaluations")
    assert hasattr(result, "wall_time_s")
    assert result.n_evaluations > 0
    assert result.wall_time_s >= 0.0
