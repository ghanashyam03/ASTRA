import builtins
import math
from typing import Any

import numpy as np
import pytest

from astra.neural.feasibility import FeasibilityClassifier
from astra.neural.features import (
    AU,
    build_geometric_features,
    compute_hohmann_tof,
    compute_vis_viva_speed,
)
from astra.neural.surrogate import SurrogateMetrics, SurrogateOutput


def test_hohmann_tof_and_vis_viva() -> None:
    """Verify analytical transfer functions match basic astrodynamic expectations."""
    # Test vis-viva speed for circular orbit at 1 AU: v = sqrt(mu/r)
    # MU_SUN = GM["SUN"] = 1.32712440018e11 km^3/s^2
    mu = 1.32712440018e11
    r = 1.496e8  # 1 AU
    v_expected = math.sqrt(mu / r)
    v_actual = compute_vis_viva_speed(r, r)
    assert pytest.approx(v_actual, rel=1e-5) == v_expected

    # Test Hohmann TOF between identical orbits: should be semi-period of circular orbit
    # T = 2 * pi * sqrt(r^3/mu), semi-period is pi * sqrt(r^3/mu)
    tof_expected = math.pi * math.sqrt(r**3 / mu)
    tof_actual = compute_hohmann_tof(r, r)
    assert pytest.approx(tof_actual, rel=1e-5) == tof_expected


def test_build_geometric_features() -> None:
    """Verify that build_geometric_features outputs expected bounds, shapes and physical bounds."""
    r1 = np.array([AU, 0.0, 0.0])
    v1 = np.array([0.0, 30.0, 0.0])
    # 90 degrees phase angle
    r2 = np.array([0.0, AU * 2.0, 0.0])

    feats = build_geometric_features(
        dep_epoch=1000.0,
        tof_seconds=5000.0,
        r1_km=r1,
        v1_km_s=v1,
        r2_km=r2,
        dep_epoch_min=0.0,
        dep_epoch_max=2000.0,
        tof_min=1000.0,
        tof_max=9000.0,
        synodic_period_s=4000.0,
    )

    assert feats.shape == (8,)
    assert feats.dtype == np.float32

    # Check normalized dep epoch: (1000 - 0) / 2000 = 0.5
    assert feats[0] == 0.5
    # Check normalized TOF: (5000 - 1000) / 8000 = 0.5
    assert feats[1] == 0.5
    # Check phase angle / pi: 90 deg / 180 deg = 0.5
    assert pytest.approx(feats[2], abs=1e-5) == 0.5
    # Check r1_AU / 5.0: 1 AU / 5.0 = 0.2
    assert pytest.approx(feats[3], abs=1e-3) == 0.2
    # Check r2_AU / 5.0: 2 AU / 5.0 = 0.4
    assert pytest.approx(feats[4], abs=1e-3) == 0.4
    # Check synodic cycle progress: (1000 % 4000) / 4000 = 0.25
    assert feats[6] == 0.25


def test_build_geometric_features_limits() -> None:
    """Verify distance clipping and divide-by-zero safety."""
    r1 = np.array([0.0, 0.0, 0.0])
    v1 = np.array([0.0, 0.0, 0.0])
    r2 = np.array([0.0, 0.0, 0.0])

    feats = build_geometric_features(
        dep_epoch=0.0,
        tof_seconds=0.0,
        r1_km=r1,
        v1_km_s=v1,
        r2_km=r2,
        dep_epoch_min=0.0,
        dep_epoch_max=0.0,
        tof_min=0.0,
        tof_max=0.0,
        synodic_period_s=0.0,
    )
    assert feats.shape == (8,)
    assert np.all(feats >= 0.0)

    # Large values should be clipped
    r1_large = np.array([AU * 10.0, 0.0, 0.0])
    r2_large = np.array([0.0, AU * 20.0, 0.0])
    feats_large = build_geometric_features(
        dep_epoch=100.0,
        tof_seconds=100.0,
        r1_km=r1_large,
        v1_km_s=v1,
        r2_km=r2_large,
        dep_epoch_min=0.0,
        dep_epoch_max=10.0,
        tof_min=0.0,
        tof_max=10.0,
        synodic_period_s=10.0,
    )
    # Normalized distances are clipped at 5.0 AU, so min(10.0, 5.0)/5.0 = 1.0
    assert feats_large[3] == 1.0
    assert feats_large[4] == 1.0


def test_surrogate_interface_properties() -> None:
    """Ensure interface enforcement of requires_physics_validation and basic registry."""
    out = SurrogateOutput(prediction=0.5, uncertainty=0.1)
    assert out.requires_physics_validation is True

    clf = FeasibilityClassifier()
    assert clf.requires_physics_validation is True
    assert clf.is_trained() is True


def test_surrogate_metrics_to_dict() -> None:
    """Verify to_dict serialization including confusion matrix counts."""
    metrics = SurrogateMetrics(
        auc_roc=0.85234,
        accuracy=0.75,
        precision=0.8,
        recall=0.7,
        n_test_samples=100,
        tp=35,
        fp=10,
        tn=40,
        fn=15,
    )
    d = metrics.to_dict()
    assert d["auc_roc"] == 0.8523
    assert d["accuracy"] == 0.75
    assert d["precision"] == 0.8
    assert d["recall"] == 0.7
    assert d["n_test_samples"] == 100
    assert d["tp"] == 35
    assert d["fp"] == 10
    assert d["tn"] == 40
    assert d["fn"] == 15


def test_feasibility_classifier_evaluate_with_u_statistic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify classifier evaluation correctness, including custom rank-sum fallback."""
    clf = FeasibilityClassifier()

    # Create simple dataset:
    # 4 samples, first 2 are negative, next 2 are positive
    X = np.random.randn(4, 8).astype(np.float32)
    y = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)

    # Mock forward to return known predictions:
    # y_flat:  0.0, 0.0, 1.0, 1.0
    # preds:   0.1, 0.8, 0.4, 0.2
    # With threshold 0.3:
    # y_pred:  0.0, 1.0, 1.0, 0.0
    # This gives:
    # tp = 1 (preds >= 0.3 for y_flat=1.0: 0.4)
    # fp = 1 (preds >= 0.3 for y_flat=0.0: 0.8)
    # tn = 1 (preds < 0.3 for y_flat=0.0: 0.1)
    # fn = 1 (preds < 0.3 for y_flat=1.0: 0.2)

    def mock_forward(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # returns (a1, a2, a3)
        return (
            np.zeros((x.shape[0], 32)),
            np.zeros((x.shape[0], 16)),
            np.array([[0.1], [0.8], [0.4], [0.2]], dtype=np.float32),
        )

    monkeypatch.setattr(clf, "forward", mock_forward)

    # 1. Test with scikit-learn metrics if installed
    metrics = clf.evaluate(X, y)
    assert metrics.tp == 1
    assert metrics.fp == 1
    assert metrics.tn == 1
    assert metrics.fn == 1
    assert metrics.accuracy == 0.5
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5

    # 2. Force U-statistic fallback code path by hiding scikit-learn
    # We mock sklearn.metrics to raise ImportError on import
    original_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        if name == "sklearn.metrics" or name.startswith("sklearn"):
            raise ImportError("Mocked import error")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    metrics_fallback = clf.evaluate(X, y)
    assert metrics_fallback.tp == 1
    assert metrics_fallback.fp == 1

    # Calculate expected U-statistic AUC:
    # pos_preds = [0.4, 0.2]
    # neg_preds = [0.1, 0.8]
    # Comparing each pos_pred > neg_pred:
    # 0.4 > 0.1 (True)
    # 0.4 > 0.8 (False)
    # 0.2 > 0.1 (True)
    # 0.2 > 0.8 (False)
    # Total sum of pos > neg is 2.
    # U-statistic AUC = 2.0 / (2 * 2) = 0.5
    assert metrics_fallback.auc_roc == 0.5
