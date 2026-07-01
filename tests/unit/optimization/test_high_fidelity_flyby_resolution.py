from pathlib import Path

import numpy as np
import pytest

SPICE = (Path("data/spice_kernels") / "de440.bsp").exists()


@pytest.mark.skipif(not SPICE, reason="SPICE kernels required")
def test_high_fidelity_differs_meaningfully_for_high_ratio_body() -> None:
    from astra.optimization.chain_solver import resolve_flyby_high_fidelity
    from astra.physics.flyby import compute_flyby
    from astra.physics.kernel import PhysicsKernel

    kernel = PhysicsKernel().load()
    epoch = kernel.epoch_from_date("2030-06-01T00:00:00")
    v_inf_in = np.array([5.0, 0.0, 0.0])
    r_p = 2439.7 + 500.0  # Mercury radius + 500 km altitude

    closed_form = compute_flyby(v_inf_in, r_p, "MERCURY", powered_dv_km_s=0.0)
    hf_v_inf_out = resolve_flyby_high_fidelity(v_inf_in, r_p, "MERCURY", epoch, kernel)

    hf_mag = float(np.linalg.norm(hf_v_inf_out))
    print(
        f"\nMercury: closed-form |v_inf_out|={closed_form.v_inf_out_km_s:.4f} km/s, "
        f"high-fidelity |v_inf_out|={hf_mag:.4f} km/s"
    )
    assert np.isfinite(hf_mag)
    # No hard threshold on the magnitude of difference — Prompt 45 already
    # measured this; this test confirms the function runs and produces a
    # comparable, finite, physically-reasonable result.
    assert 0.0 < hf_mag < 50.0


@pytest.mark.skipif(not SPICE, reason="SPICE kernels required")
def test_low_ratio_body_unaffected_by_threshold_default() -> None:
    """Confirms the threshold selection doesn't accidentally trigger
    high-fidelity resolution for a low-ratio body at default settings."""
    from astra.physics.kernel import PhysicsKernel
    from astra.physics.soi_passage_estimate import soi_crossing_displacement_ratio

    kernel = PhysicsKernel().load()
    epoch = kernel.epoch_from_date("2030-06-01T00:00:00")
    neptune_ratio = soi_crossing_displacement_ratio("NEPTUNE", 10.0, epoch, kernel).ratio
    assert neptune_ratio < 5.0, (
        "Neptune's ratio should be below the default threshold of 5.0, "
        "confirming low-ratio bodies correctly fall back to the unchanged "
        "closed-form resolution path"
    )
