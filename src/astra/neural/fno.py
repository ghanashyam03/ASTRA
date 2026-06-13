"""Fourier Feature Network surrogate for porkchop grid acceleration.

IMPORTANT TERMINOLOGY NOTE: This is a Fourier Feature Network (FFN), not a
true Fourier Neural Operator (FNO). A true FNO uses trainable complex Fourier
integral layers. This implementation uses random Fourier features (Bochner's
theorem, Rahimi & Recht 2007) as fixed input encodings, followed by a standard
MLP. This approach efficiently captures the periodic structure of porkchop
grids (which are quasi-periodic at the synodic period) without requiring
complex-valued neural networks or Fourier integral operators.

The surrogate always requires physics validation: trained FNO predictions
identify candidate optima, but every candidate is verified by the Lambert
solver before being reported in the output grid.

Reference: Rahimi & Recht (2007). Random features for large-scale kernel
machines. NeurIPS 20.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from astra.neural.features import build_geometric_features
from astra.neural.surrogate import NeuralSurrogate, SurrogateMetrics, SurrogateOutput

INPUT_DIM = 10   # 2 normalized coords + 8 geometric features

class PorkchopFNO(NeuralSurrogate):
    """Fourier Feature Network surrogate for 2-D porkchop grid prediction.
    
    Architecture:
      Random Fourier features: 10-dim input → 2*n_fourier_features encoding
      MLP: [2*n_ff, hidden_dim, hidden_dim, 1] with ReLU + Softplus output
    
    Training loss: MSE in log(Δv+1) space to prevent large infeasible values
    from dominating the gradient.
    
    Output: predicted total Δv in km/s (always positive via Softplus).
    Predictions > 30 km/s are treated as infeasible (NaN in grid output).
    """

    def __init__(
        self,
        n_fourier_features: int = 128,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        fourier_scale: float = 1.0,
        seed: int = 42,
    ) -> None:
        # Random Fourier Feature matrix — fixed, never trained
        rng = np.random.default_rng(seed)
        self.B = rng.normal(0.0, fourier_scale, (INPUT_DIM, n_fourier_features)).astype(np.float32)
        
        ff_dim = 2 * n_fourier_features  # cos + sin features

        def he(d_in: int, d_out: int) -> np.ndarray:
            return (rng.standard_normal((d_in, d_out))
                    * math.sqrt(2.0 / d_in)).astype(np.float32)

        self.W1 = he(ff_dim, hidden_dim)
        self.b1 = np.zeros((1, hidden_dim), dtype=np.float32)
        self.W2 = he(hidden_dim, hidden_dim)
        self.b2 = np.zeros((1, hidden_dim), dtype=np.float32)
        self.W3 = he(hidden_dim, 1)
        self.b3 = np.zeros((1, 1), dtype=np.float32)

        self.n_ff = n_fourier_features
        self.ff_dim = ff_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self._trained = False

        # Cache for backprop
        self._z: list[np.ndarray] = []
        self._a: list[np.ndarray] = []

    def _fourier_encode(self, x: np.ndarray) -> np.ndarray:
        """Encode (batch, 10) → (batch, 2*n_ff) using random Fourier features."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        z = x @ self.B  # (batch, n_ff)
        return np.concatenate([
            np.cos(2.0 * math.pi * z),
            np.sin(2.0 * math.pi * z),
        ], axis=1).astype(np.float32)

    def _softplus(self, x: np.ndarray) -> np.ndarray:
        return np.log1p(np.exp(np.clip(x, -20.0, 20.0)))  # type: ignore[no-any-return]

    def _softplus_grad(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))  # type: ignore[no-any-return]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (batch, INPUT_DIM) → (batch, 1) predicted Δv km/s."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        phi = self._fourier_encode(x)
        self._z = []
        self._a = [phi]
        z1 = phi @ self.W1 + self.b1
        a1 = np.maximum(0.0, z1)
        self._z.append(z1)
        self._a.append(a1)
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(0.0, z2)
        self._z.append(z2)
        self._a.append(a2)
        z3 = a2 @ self.W3 + self.b3
        out = self._softplus(z3)
        self._z.append(z3)
        return np.clip(out, 0.1, 50.0)  # type: ignore[no-any-return]

    def _build_input(
        self,
        dep: float,
        tof: float,
        dep_min: float,
        dep_max: float,
        tof_min: float,
        tof_max: float,
        r1: np.ndarray,
        v1: np.ndarray,
        r2: np.ndarray,
        synodic_period_s: float,
    ) -> np.ndarray:
        """Build 10-dim input vector for one (dep, tof) grid point."""
        dep_norm = (dep - dep_min) / max(dep_max - dep_min, 1.0)
        tof_norm = (tof - tof_min) / max(tof_max - tof_min, 1.0)
        geo = build_geometric_features(
            dep, tof, r1, v1, r2,
            dep_min, dep_max, tof_min, tof_max, synodic_period_s
        )
        return np.array([dep_norm, tof_norm, *geo], dtype=np.float32)

    def train_on_grid(
        self,
        dep_epochs: np.ndarray,
        tof_array: np.ndarray,
        dv_grid: np.ndarray,
        dep_min: float,
        dep_max: float,
        tof_min: float,
        tof_max: float,
        body_states_dep: dict[float, tuple[np.ndarray, np.ndarray]],
        body_states_arr: dict[float, tuple[np.ndarray, np.ndarray]],
        synodic_period_s: float,
        epochs: int = 100,
        batch_size: int = 256,
        val_fraction: float = 0.1,
    ) -> dict[str, list[float]]:
        """Train on a computed porkchop grid.
        body_states_dep: {dep_epoch: (r1_km, v1_km_s)}
        body_states_arr: {arr_epoch: (r2_km, v2_km_s)}
        Loss: MSE in log(dv+1) space.
        Returns {"train_loss": [...], "val_loss": [...]} per epoch."""
        
        X_list, y_list = [], []
        for i, dep in enumerate(dep_epochs):
            if dep not in body_states_dep:
                continue
            r1, v1 = body_states_dep[dep]
            for j, tof in enumerate(tof_array):
                dv = dv_grid[i, j]
                if not np.isfinite(dv) or dv >= 25.0:
                    continue
                arr = dep + tof
                if arr not in body_states_arr:
                    continue
                r2, _ = body_states_arr[arr]
                x = self._build_input(dep, tof, dep_min, dep_max,
                                      tof_min, tof_max, r1, v1, r2, synodic_period_s)
                X_list.append(x)
                y_list.append(float(dv))

        if not X_list:
            return {"train_loss": [], "val_loss": []}

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        # Log-space targets
        y_log = np.log1p(y).reshape(-1, 1)

        n = len(X)
        n_val = max(1, int(n * val_fraction))
        X_val, y_val_log = X[:n_val], y_log[:n_val]
        X_tr, y_tr_log = X[n_val:], y_log[n_val:]

        rng = np.random.default_rng(0)
        train_losses, val_losses = [], []

        for ep in range(epochs):
            idx = rng.permutation(len(X_tr))
            ep_loss = 0.0
            n_batches = 0
            for start in range(0, len(X_tr), batch_size):
                b = idx[start:start + batch_size]
                xb, yb = X_tr[b], y_tr_log[b]
                pred_raw = self.forward(xb)
                pred_log = np.log1p(pred_raw)
                err = pred_log - yb
                loss = float(np.mean(err**2))
                ep_loss += loss
                n_batches += 1
                # Backprop
                d_pred_log = 2.0 * err / len(xb)
                # Gradient through log1p(pred): d(log1p(p))/dp = 1/(1+p)
                d_pred = d_pred_log / (1.0 + pred_raw + 1e-8)
                # Softplus gradient at output
                d_z3 = d_pred * self._softplus_grad(self._z[2])
                dW3 = self._a[2].T @ d_z3
                db3 = d_z3.mean(axis=0, keepdims=True)
                np.clip(dW3, -1.0, 1.0, out=dW3)
                np.clip(db3, -1.0, 1.0, out=db3)
                self.W3 -= self.lr * dW3
                self.b3 -= self.lr * db3
                d_a2 = d_z3 @ self.W3.T
                d_z2 = d_a2 * (self._z[1] > 0.0)
                dW2 = self._a[1].T @ d_z2
                db2 = d_z2.mean(axis=0, keepdims=True)
                np.clip(dW2, -1.0, 1.0, out=dW2)
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2
                d_a1 = d_z2 @ self.W2.T
                d_z1 = d_a1 * (self._z[0] > 0.0)
                dW1 = self._a[0].T @ d_z1
                db1 = d_z1.mean(axis=0, keepdims=True)
                np.clip(dW1, -1.0, 1.0, out=dW1)
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1
            # Validation loss
            val_pred_log = np.log1p(self.forward(X_val))
            val_loss = float(np.mean((val_pred_log - y_val_log)**2))
            train_losses.append(ep_loss / max(n_batches, 1))
            val_losses.append(val_loss)

        self._trained = True
        return {"train_loss": train_losses, "val_loss": val_losses}

    def predict_grid(
        self,
        dep_epochs: np.ndarray,
        tof_seconds: np.ndarray,
        dep_min: float,
        dep_max: float,
        tof_min: float,
        tof_max: float,
        body_states_dep: dict[float, tuple[np.ndarray, np.ndarray]],
        body_states_arr: dict[float, tuple[np.ndarray, np.ndarray]],
        synodic_period_s: float,
    ) -> np.ndarray:
        """Predict full (n_dep, n_tof) Δv grid. Returns NaN for infeasible cells."""
        n_dep, n_tof = len(dep_epochs), len(tof_seconds)
        grid = np.full((n_dep, n_tof), np.nan, dtype=np.float32)
        inputs = []
        positions = []
        for i, dep in enumerate(dep_epochs):
            if dep not in body_states_dep:
                continue
            r1, v1 = body_states_dep[dep]
            for j, tof in enumerate(tof_seconds):
                arr = dep + tof
                if arr not in body_states_arr:
                    continue
                r2, _ = body_states_arr[arr]
                x = self._build_input(dep, tof, dep_min, dep_max,
                                      tof_min, tof_max, r1, v1, r2, synodic_period_s)
                inputs.append(x)
                positions.append((i, j))
        if not inputs:
            return grid
        X = np.array(inputs, dtype=np.float32)
        preds = self.forward(X).flatten()
        for (i, j), pred in zip(positions, preds):
            grid[i, j] = pred if pred < 30.0 else np.nan
        return grid

    def find_top_k_candidates(
        self,
        predicted_grid: np.ndarray,
        dep_epochs: np.ndarray,
        tof_seconds: np.ndarray,
        top_k: int = 20,
    ) -> list[tuple[float, float]]:
        """Return top_k (dep_epoch, tof_seconds) with lowest predicted Δv."""
        finite_mask = np.isfinite(predicted_grid)
        if not np.any(finite_mask):
            return []
        flat = predicted_grid.copy()
        flat[~finite_mask] = 999.0
        flat_indices = np.argsort(flat.ravel())[:top_k]
        candidates = []
        for idx in flat_indices:
            i, j = np.unravel_index(idx, flat.shape)
            if np.isfinite(predicted_grid[i, j]):
                candidates.append((float(dep_epochs[i]), float(tof_seconds[j])))
        return candidates

    # ─── NeuralSurrogate interface ────────────────────────────────────────
    def predict(self, features: np.ndarray, **kwargs: Any) -> SurrogateOutput:
        """features: 10-dim vector [dep_norm, tof_norm, 8 geometric features]."""
        pred = float(self.forward(features.reshape(1, -1))[0, 0])
        return SurrogateOutput(prediction=pred, uncertainty=0.0,
                               requires_physics_validation=True)

    def is_trained(self) -> bool:
        return self._trained

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> SurrogateMetrics:
        """Evaluate on test set. y_test are true Δv values in km/s."""
        mask = y_test < 15.0
        if not np.any(mask):
            return SurrogateMetrics(auc_roc=0.0, accuracy=0.0,
                                    precision=0.0, recall=0.0, n_test_samples=0)
        x_f, y_f = x_test[mask], y_test[mask]
        preds = self.forward(x_f).flatten()
        errors = np.abs(preds - y_f)
        accuracy = float(np.mean(errors < 1.0))
        precision = float(np.mean(errors < 0.5))
        return SurrogateMetrics(
            auc_roc=0.0, accuracy=accuracy, precision=precision,
            recall=0.0, n_test_samples=int(np.sum(mask))
        )
