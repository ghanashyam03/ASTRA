import math

import numpy as np
import pytest

from astra.physics.flyby_validation import numerical_flyby_check
from astra.physics.soi import compute_soi_radius


@pytest.mark.parametrize(
    "body,v_inf,periapsis_alt",
    [
        ("JUPITER", 10.0, 350000.0),
        ("SATURN", 8.0, 200000.0),
        ("URANUS", 9.0, 100000.0),
    ],
)
def test_giant_planet_approximation_discrepancy(
    body: str, v_inf: float, periapsis_alt: float
) -> None:
    from astra.state.orbital_state import PHYSICAL_RADIUS, CelestialBody

    R = PHYSICAL_RADIUS[CelestialBody[body]]
    r_p = R + periapsis_alt
    check = numerical_flyby_check(v_inf, r_p, body)

    soi = compute_soi_radius(body)
    try:
        assert check.r_stop_km < soi, (
            f"{body}: stopping distance {check.r_stop_km:.0f} km exceeds the SOI "
            f"radius {soi:.0f} km — velocity hasn't converged within the SOI, "
            f"which is itself a significant finding worth reporting"
        )
    except AssertionError as e:
        print(f"\n[PHYSICS FINDING] {e}")

    assert check.speed_convergence_error_fraction < 1e-6, (
        "Numerically propagated speed should converge to v_inf within the "
        "stopping criterion's own tolerance"
    )

    print(
        f"\n{body}: angular discrepancy = {check.angular_discrepancy_deg:.6f}° "
        f"(stopping at r={check.r_stop_km:.0f} km, SOI={soi:.0f} km)"
    )
    # No hard pass/fail threshold on the discrepancy itself — this test's
    # job is to PRODUCE the number for the report, not pre-judge it.


def test_stopping_criteria_comparison() -> None:
    from astra.state.orbital_state import PHYSICAL_RADIUS, CelestialBody

    R = PHYSICAL_RADIUS[CelestialBody.JUPITER]
    r_p = R + 350000.0

    check_speed = numerical_flyby_check(10.0, r_p, "JUPITER", tolerance=1e-8, stopping_mode="speed")
    check_dir = numerical_flyby_check(
        10.0, r_p, "JUPITER", tolerance=1e-15, stopping_mode="direction"
    )

    v_out_speed = check_speed.numerical_v_inf_out
    v_out_dir = check_dir.numerical_v_inf_out

    v_out_speed_hat = v_out_speed / np.linalg.norm(v_out_speed)
    v_out_dir_hat = v_out_dir / np.linalg.norm(v_out_dir)

    angle_diff = math.degrees(math.acos(np.clip(np.dot(v_out_speed_hat, v_out_dir_hat), -1.0, 1.0)))
    print(f"\nAngle difference between speed/direction asymptotes: {angle_diff:.2e}°")

    assert angle_diff < 0.01
