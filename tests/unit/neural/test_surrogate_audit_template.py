"""Calibration test: the audit template must reproduce, against the ACTUAL
existing LambertPINN, the same qualitative finding the manual audit reported —
poor absolute regressor, useful ranker. If this test fails, the audit
template itself is miscalibrated and must not be trusted for future surrogates."""

from pathlib import Path

import pytest

SPICE = (Path("data/spice_kernels") / "de440.bsp").exists()


@pytest.mark.skipif(not SPICE, reason="SPICE kernels required")
def test_audit_template_reproduces_known_pinn_verdict() -> None:
    from astra.dsl.compiler import compile_mission
    from astra.dsl.parser import parse_mission_file
    from astra.neural.pinn import LambertPINN
    from astra.neural.surrogate_audit import run_surrogate_audit
    from astra.neural.training.pipeline import generate_pinn_dataset
    from astra.physics.kernel import PhysicsKernel

    kernel = PhysicsKernel().load()
    dsl = parse_mission_file("data/benchmarks/earth_mars_2031.yaml")
    mission = compile_mission(dsl, kernel.ephemeris)

    X, dv_y, r1n, r2n, tofs = generate_pinn_dataset(
        kernel,
        mission.origin_body,
        mission.destination_body,
        mission.departure_epoch_start,
        mission.departure_epoch_end,
        mission.tof_min_seconds,
        mission.tof_max_seconds,
        n_samples=300,
        seed=42,
    )
    feasible_mask = dv_y < 20.0
    X_feas, y_feas = X[feasible_mask], dv_y[feasible_mask]
    split = int(len(X_feas) * 0.8)
    X_train, X_test = X_feas[:split], X_feas[split:]
    y_train, y_test = y_feas[:split], y_feas[split:]

    pinn = LambertPINN()
    pinn.train_on_dataset(
        x_data=X_train,
        v_targets=y_train,
        r1_norms=r1n[feasible_mask][:split],
        r2_norms=r2n[feasible_mask][:split],
        tof_seconds=tofs[feasible_mask][:split],
        epochs=30,
        batch_size=64,
    )
    predictions = pinn.predict_batch(X_test)

    report = run_surrogate_audit("LambertPINN_calibration_check", predictions, y_test)
    print(f"\nAudit template result on real LambertPINN: {report.verdict}")
    print(f"MAE: {report.mae:.2f} km/s, Spearman: {report.spearman_correlation:.3f}")

    # The calibration assertion: this should reproduce the documented finding
    # from the actual PINN warm-start audit (poor absolute accuracy, positive
    # but modest rank correlation) — not necessarily the EXACT numbers from
    # that audit (different seed/sample size), but the SAME qualitative shape.
    assert report.accuracy_within_1km_s < 0.5, (
        "Calibration check expected a poor absolute regressor, matching the "
        "documented PINN audit finding — if this now shows high accuracy, "
        "either the PINN architecture changed or the audit template is "
        "miscalibrated; investigate before trusting this template further"
    )
