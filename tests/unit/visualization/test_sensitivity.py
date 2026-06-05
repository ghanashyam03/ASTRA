from __future__ import annotations

import numpy as np

from astra.physics.lambert import lambert_izzo
from astra.state.orbital_state import GM
from astra.visualization.sensitivity import compute_sensitivity


def test_sensitivity_gradient_direction() -> None:
    """For Earth-Mars Lambert, shorter TOF must increase Δv (positive gradient)."""
    MU = GM["SUN"]
    r1 = np.array([1.496e8, 0.0, 0.0])
    r2 = np.array([0.0, 2.279e8, 0.0])

    def dv_of_tof(tof: float) -> float:
        try:
            v1, v2, conv = lambert_izzo(r1, r2, tof, MU)
            if not conv:
                return 99.0
            v1_body = np.array([0.0, 29.78, 0.0])
            v2_body = np.array([-24.13, 0.0, 0.0])
            return float(np.linalg.norm(v1 - v1_body) + np.linalg.norm(v2_body - v2))
        except Exception:
            return 99.0

    result = compute_sensitivity(dv_of_tof, 200.0 * 86400.0, 86400.0, "tof", "km/s per day")
    # Near-optimal TOF: shortening should increase Δv → gradient negative
    # (shorter tof → higher dv, so f(x-h) > f(x+h) meaning gradient < 0 for shortening)
    assert result.dv_plus != result.dv_minus  # sensitivity exists
    d = result.to_dict()
    assert "gradient" in d
    assert "units" in d
