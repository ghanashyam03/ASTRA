from __future__ import annotations

from astra.visualization.sensitivity import SensitivityPoint, central_difference


def test_central_difference_quadratic() -> None:
    def f(x: float) -> float:
        return x**2

    f0, fp, fm = central_difference(f, 3.0, 0.1)
    grad = (fp - fm) / (2 * 0.1)
    assert abs(f0 - 9.0) < 1e-10
    assert abs(grad - 6.0) < 1e-4


def test_sensitivity_point_to_dict() -> None:
    sp = SensitivityPoint(
        parameter_name="tof",
        baseline_value=200.0 * 86400,
        perturbation_step=86400.0,
        units="km/s per day",
        baseline_dv=6.0,
        dv_plus=6.1,
        dv_minus=5.9,
        gradient=0.1,
    )
    d = sp.to_dict()
    assert d["parameter"] == "tof"
    assert d["gradient_km_s_per_unit"] == 0.1
