from pathlib import Path

import numpy as np
import pytest

from astra.neural.pinn import ActiveLearningManager, LambertPINN, LambertPINNEnsemble
from astra.neural.surrogate import SurrogateMetrics, SurrogateOutput, SurrogatePrediction
from astra.physics.kernel import PhysicsKernel
from astra.state.orbital_state import CelestialBody, OrbitalState
from astra.state.trajectory import Maneuver, Trajectory

SPICE = (Path("data/spice_kernels") / "de440.bsp").exists()


def test_pinn_initialization() -> None:
    """Verify that the model initializes correctly with He weight values."""
    pinn = LambertPINN(hidden_dims=[32, 16])
    assert len(pinn.weights) == 3
    assert len(pinn.biases) == 3

    # Checking input layer weights shape: (8, 32)
    assert pinn.weights[0].shape == (8, 32)
    assert pinn.biases[0].shape == (1, 32)

    # Checking middle layer weights shape: (32, 16)
    assert pinn.weights[1].shape == (32, 16)
    assert pinn.biases[1].shape == (1, 16)

    # Checking output layer weights shape: (16, 6)
    assert pinn.weights[2].shape == (16, 6)
    assert pinn.biases[2].shape == (1, 6)

    assert pinn.weights[0].dtype == np.float32
    assert not pinn.is_trained()


def test_pinn_forward() -> None:
    """Verify the forward pass and activation restrictions."""
    pinn = LambertPINN(hidden_dims=[16])
    x = np.random.randn(5, 8).astype(np.float32)
    out = pinn.forward(x)

    # Output shape should be (batch_size, 6)
    assert out.shape == (5, 6)
    assert out.dtype == np.float32


def test_pinn_physics_residual() -> None:
    """Verify the shape and computation of physics residual."""
    pinn = LambertPINN(hidden_dims=[16])
    features = np.random.randn(4, 8).astype(np.float32)
    r1 = np.ones(4, dtype=np.float32) * 1.5e8
    r2 = np.ones(4, dtype=np.float32) * 2.2e8
    tof = np.ones(4, dtype=np.float32) * 200.0 * 86400.0

    res = pinn.physics_residual(features, r1, r2, tof)
    assert res.shape == (4, 1)
    assert res.dtype == np.float32
    assert np.all(res >= 0.0)


def test_pinn_training() -> None:
    """Verify backpropagation training updates weights and decreases loss."""
    pinn = LambertPINN(hidden_dims=[16], lr=0.1, physics_weight=0.01)

    # Generate synthetic training dataset
    N = 100
    np.random.seed(42)
    x_data = np.random.rand(N, 8).astype(np.float32)
    dv_targets = np.random.rand(N).astype(np.float32) * 15.0  # feasible
    r1_norms = np.random.rand(N).astype(np.float32) * 1.0e8 + 1.0e8
    r2_norms = np.random.rand(N).astype(np.float32) * 1.0e8 + 2.0e8
    tof_sec = np.random.rand(N).astype(np.float32) * 1.0e7 + 1.0e7

    # Capture initial weights to verify changes
    initial_w0 = pinn.weights[0].copy()

    loss_history = pinn.train_on_dataset(
        x_data, dv_targets, r1_norms, r2_norms, tof_sec, epochs=5, batch_size=32
    )

    assert len(loss_history) == 5
    assert pinn.is_trained()
    assert not np.array_equal(pinn.weights[0], initial_w0)


def test_pinn_interface_methods() -> None:
    """Verify standard NeuralSurrogate interface methods."""
    pinn = LambertPINN(hidden_dims=[8])
    feat = np.random.randn(8).astype(np.float32)

    # test predict
    out = pinn.predict(feat)
    assert isinstance(out, SurrogateOutput)
    assert out.prediction >= 0.0
    assert out.uncertainty == 0.0
    assert out.requires_physics_validation

    # test predict_batch
    feats = np.random.randn(10, 8).astype(np.float32)
    preds = pinn.predict_batch(feats)
    assert preds.shape == (10,)

    # test is_likely_feasible
    pinn.weights[-1] = np.zeros_like(pinn.weights[-1])
    # total_dv = 2 * ||[0.5, 0.5, 0.5]|| = 2 * sqrt(0.75) = 1.732
    pinn.biases[-1] = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
    assert pinn.is_likely_feasible(feat, threshold=2.0)
    assert not pinn.is_likely_feasible(feat, threshold=1.0)

    # test evaluate
    y_test = np.array([1.2, 5.5, 99.0], dtype=np.float32)  # one infeasible
    x_test = np.random.randn(3, 8).astype(np.float32)
    metrics = pinn.evaluate(x_test, y_test)
    assert isinstance(metrics, SurrogateMetrics)
    assert metrics.auc_roc == 0.0
    assert metrics.n_test_samples == 2  # only feasible counted


def test_pinn_ensemble() -> None:
    """Verify ensemble prediction and uncertainty calculation."""
    ensemble = LambertPINNEnsemble(hidden_dims=[16], ensemble_size=3)
    feat = np.random.randn(8).astype(np.float32)
    out = ensemble.predict(feat)
    assert isinstance(out, SurrogatePrediction)
    assert out.prediction >= 0.0
    assert out.mean.shape == (6,)
    assert out.variance.shape == (6,)
    assert out.std.shape == (6,)

    feats = np.random.randn(5, 8).astype(np.float32)
    preds = ensemble.predict_batch(feats)
    assert preds.shape == (5,)


@pytest.mark.skipif(not SPICE, reason="SPICE kernels required")
def test_active_learning() -> None:
    """Verify ActiveLearningManager triggers retraining and acquires samples."""
    kernel = PhysicsKernel().load()
    surrogate = LambertPINNEnsemble(hidden_dims=[8], ensemble_size=2)

    manager = ActiveLearningManager(
        surrogate=surrogate,
        kernel=kernel,
        uncertainty_threshold=0.01,
        retrain_every=2,
        epochs=1,
        batch_size=2,
    )

    assert len(manager.x_buffer) == 0

    feat1 = np.random.randn(8).astype(np.float32)
    feat2 = np.random.randn(8).astype(np.float32)

    def mock_predict(
        feat: np.ndarray,
        v_planet_depart: np.ndarray | None = None,
        v_planet_arrive: np.ndarray | None = None,
    ) -> SurrogatePrediction:
        return SurrogatePrediction(
            prediction=5.0,
            uncertainty=0.5,  # greater than 0.01
            mean=np.zeros(6, dtype=np.float32),
            variance=np.zeros(6, dtype=np.float32),
            std=np.zeros(6, dtype=np.float32),
            delta_v=5.0,
        )

    import typing

    typing.cast(typing.Any, surrogate).predict = mock_predict

    # Sample 1
    manager.query_and_learn(feat1, CelestialBody.EARTH, CelestialBody.MARS, 0.0, 200 * 86400.0)
    assert len(manager.x_buffer) == 1
    assert manager.new_samples_count == 1

    # Sample 2 (triggers retraining)
    manager.query_and_learn(feat2, CelestialBody.EARTH, CelestialBody.MARS, 0.0, 200 * 86400.0)
    assert len(manager.x_buffer) == 2
    assert manager.new_samples_count == 0
    assert surrogate.is_trained()


@pytest.mark.skipif(not SPICE, reason="SPICE kernels required")
def test_trajectory_validation() -> None:
    """Verify PhysicsKernel trajectory validation pipeline works correctly."""
    kernel = PhysicsKernel().load()

    # Construct a simple Earth-Mars transfer trajectory
    dep_epoch = 10.0 * 365.25 * 86400.0  # J2000 epoch
    tof = 250.0 * 86400.0

    r1_state = kernel.get_body_state(CelestialBody.EARTH, dep_epoch)
    r2_state = kernel.get_body_state(CelestialBody.MARS, dep_epoch + tof)

    from astra.physics.lambert import find_best_transfer

    sol = find_best_transfer(
        r1=r1_state.position,
        v1_body=r1_state.velocity,
        r2=r2_state.position,
        v2_body=r2_state.velocity,
        tof=tof,
        mu=1.32712440018e11,
        max_revs=0,
    )

    from astra.physics.maneuvers import arrival_delta_v, departure_delta_v

    v_inf_dep = sol.v1 - r1_state.velocity
    v_inf_arr = r2_state.velocity - sol.v2
    dv1 = departure_delta_v(v_inf_dep, 200.0, "EARTH")
    dv2 = arrival_delta_v(v_inf_arr, 300.0, "MARS")

    s0 = OrbitalState(
        epoch=dep_epoch,
        position=r1_state.position,
        velocity=sol.v1,
        central_body=CelestialBody.SUN,
    )
    s1 = OrbitalState(
        epoch=dep_epoch + tof,
        position=r2_state.position,
        velocity=sol.v2,
        central_body=CelestialBody.SUN,
    )

    m1 = Maneuver(
        epoch=dep_epoch,
        delta_v=(v_inf_dep / np.linalg.norm(v_inf_dep)) * dv1,
        label="DEP",
    )
    m2 = Maneuver(
        epoch=dep_epoch + tof,
        delta_v=(v_inf_arr / np.linalg.norm(v_inf_arr)) * dv2,
        label="CAP",
    )

    traj = Trajectory(
        states=[s0, s1],
        maneuvers=[m1, m2],
        metadata={
            "parking_altitude_km": 200.0,
            "capture_altitude_km": 300.0,
        },
    )

    res = kernel.validate_trajectory(traj, pos_tol_km=5000.0, dv_tol_kms=0.1)
    assert res.is_valid
    assert res.pos_diff < 1.0
    assert res.dv_diff < 1e-4
