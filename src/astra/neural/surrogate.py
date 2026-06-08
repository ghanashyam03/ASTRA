"""Abstract base and registry for all ASTRA neural surrogate models.
Physics validation is MANDATORY — this is enforced at the interface level.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SurrogateOutput:
    prediction: float
    uncertainty: float
    requires_physics_validation: bool = True  # ALWAYS True in ASTRA

@dataclass
class SurrogatePrediction(SurrogateOutput):
    mean: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )  # shape (6,)
    variance: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )  # shape (6,)
    std: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )  # shape (6,)
    delta_v: float = 0.0

@dataclass
class SurrogateMetrics:
    """Performance metrics from evaluation on a labeled test set."""
    auc_roc: float         # for classifiers
    accuracy: float        # for classifiers
    precision: float
    recall: float
    n_test_samples: int
    tp: int | None = None
    fp: int | None = None
    tn: int | None = None
    fn: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "auc_roc": round(self.auc_roc, 4),
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "n_test_samples": self.n_test_samples,
        }
        if self.tp is not None:
            d["tp"] = self.tp
            d["fp"] = self.fp
            d["tn"] = self.tn
            d["fn"] = self.fn
        return d

class NeuralSurrogate(ABC):
    """Abstract base for all ASTRA neural surrogates.
    The requires_physics_validation property MUST always return True.
    Subclasses may not override it to False."""

    @property
    def requires_physics_validation(self) -> bool:
        return True

    @abstractmethod
    def predict(self, features: np.ndarray, **kwargs: Any) -> SurrogateOutput:  # noqa: ANN401
        """Predict output for a single feature vector."""
        ...

    @abstractmethod
    def is_trained(self) -> bool:
        ...

    @abstractmethod
    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> SurrogateMetrics:
        """Compute performance metrics on a labeled test set."""
        ...
