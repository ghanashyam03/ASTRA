"""Physics-Informed Neural Network (PINN) surrogate model for delta-v regression.

This model predicts departure and arrival velocity vectors (6 output dimensions)
and uses configurable combined physics residuals (vis-viva, energy, angular momentum).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from astra.physics.kernel import PhysicsKernel
    from astra.state.orbital_state import CelestialBody

from astra.neural.surrogate import (
    NeuralSurrogate,
    SurrogateMetrics,
    SurrogatePrediction,
)


class LambertPINN(NeuralSurrogate):
    """Physics-Informed Neural Network for continuous delta-v regression.

    This model learns the Lambert transfer velocity vectors as a continuous function
    over departure epoch x TOF space. It uses a pure NumPy MLP architecture and
    enforces physical conservation laws (vis-viva, energy, angular momentum).
    """

    def __init__(
        self,
        hidden_dims: list[int] = [64, 64, 32],
        lr: float = 3e-3,
        physics_weight: float = 0.1,
        mu_sun: float = 1.32712440018e11,
        use_vis_viva: bool = True,
        use_energy: bool = True,
        use_angular_momentum: bool = True,
        vis_viva_weight: float = 1.0,
        energy_weight: float = 0.1,
        angular_momentum_weight: float = 0.1,
    ) -> None:
        """Initialize the PINN regressor.

        Parameters
        ----------
        hidden_dims : list[int]
            List of hidden layer dimensions.
        lr : float
            Learning rate.
        physics_weight : float
            Weight for the physics residual loss term.
        mu_sun : float
            Gravitational parameter of the Sun in km^3/s^2.
        use_vis_viva : bool
            Whether to enforce vis-viva constraint.
        use_energy : bool
            Whether to enforce energy conservation constraint.
        use_angular_momentum : bool
            Whether to enforce angular momentum conservation constraint.
        vis_viva_weight : float
            Loss weight for vis-viva residual.
        energy_weight : float
            Loss weight for energy residual.
        angular_momentum_weight : float
            Loss weight for angular momentum residual.
        """
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.physics_weight = physics_weight
        self.mu_sun = mu_sun

        self.use_vis_viva = use_vis_viva
        self.use_energy = use_energy
        self.use_angular_momentum = use_angular_momentum
        self.vis_viva_weight = vis_viva_weight
        self.energy_weight = energy_weight
        self.angular_momentum_weight = angular_momentum_weight

        # Layer dimensions: 8 (input features) -> hidden_dims -> 6 (predicted velocities)
        dims = [8] + list(hidden_dims) + [6]

        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []

        # Initialize weights and biases using He initialization
        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            w = (np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)).astype(np.float32)
            b = np.zeros((1, out_dim), dtype=np.float32)
            self.weights.append(w)
            self.biases.append(b)

        # Activations and pre-activations cache for backpropagation
        self._activations: list[np.ndarray] = []
        self._preacts: list[np.ndarray] = []
        self._trained: bool = False

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Perform forward pass through the MLP.

        Parameters
        ----------
        x : np.ndarray
            Input features of shape (batch_size, 8) float32.

        Returns
        -------
        np.ndarray
            Predicted velocity components [v_dep, v_arr] of shape (batch_size, 6) float32.
        """
        x_cast = np.asarray(x, dtype=np.float32)
        self._activations = [x_cast]
        self._preacts = []
        current_act = x_cast

        num_layers = len(self.weights)
        for i in range(num_layers - 1):
            z = current_act @ self.weights[i] + self.biases[i]
            a = np.maximum(0.0, z)  # ReLU
            self._preacts.append(z)
            self._activations.append(a)
            current_act = a

        # Output layer with linear activation (velocities can be negative)
        z_out = current_act @ self.weights[-1] + self.biases[-1]
        out = z_out.astype(np.float32)

        self._preacts.append(z_out)
        self._activations.append(out)

        return out  # type: ignore[no-any-return]

    def physics_residual(
        self,
        features: np.ndarray,
        r1_norms_km: np.ndarray,
        r2_norms_km: np.ndarray,
        tof_seconds: np.ndarray,
        r1_vecs: np.ndarray | None = None,
        r2_vecs: np.ndarray | None = None,
        v_planet_depart: np.ndarray | None = None,
        v_planet_arrive: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute the combined physics residual based on active constraints.

        Parameters
        ----------
        features : np.ndarray
            Geometric feature matrix of shape (batch_size, 8).
        r1_norms_km : np.ndarray
            Departure position norms of shape (batch_size,).
        r2_norms_km : np.ndarray
            Arrival position norms of shape (batch_size,).
        tof_seconds : np.ndarray
            Times of flight in seconds of shape (batch_size,).
        r1_vecs : np.ndarray, optional
            Departure position 3D vectors of shape (batch_size, 3).
        r2_vecs : np.ndarray, optional
            Arrival position 3D vectors of shape (batch_size, 3).
        v_planet_depart : np.ndarray, optional
            Departure planet 3D velocity vectors of shape (batch_size, 3).
        v_planet_arrive : np.ndarray, optional
            Arrival planet 3D velocity vectors of shape (batch_size, 3).

        Returns
        -------
        np.ndarray
            Physics residuals of shape (batch_size, 1).
        """
        batch_size = features.shape[0]
        r1_norm = r1_norms_km.reshape(-1, 1).astype(np.float32)
        r2_norm = r2_norms_km.reshape(-1, 1).astype(np.float32)

        pred = self.forward(features)
        v_dep_pred = pred[:, :3]
        v_arr_pred = pred[:, 3:]

        total_res = np.zeros((batch_size, 1), dtype=np.float32)
        a_transfer = (r1_norm + r2_norm) / 2.0

        if self.use_vis_viva:
            # v_inf_pred = ||v_dep_pred - v_planet_depart||
            if v_planet_depart is not None:
                v_planet_dep = v_planet_depart
            else:
                v_planet_dep = np.zeros((batch_size, 3), dtype=np.float32)
            v_inf_pred = np.linalg.norm(v_dep_pred - v_planet_dep, axis=1, keepdims=True)
            v_transfer_r1 = np.sqrt(self.mu_sun * (2.0 / r1_norm - 1.0 / a_transfer))
            res_vis = (v_transfer_r1 - v_inf_pred) ** 2
            total_res += self.vis_viva_weight * res_vis

        if self.use_energy:
            # Specific orbital energy conservation: 1/2 v_dep^2 - mu/r_1 = -mu/(2a)
            # Using heliocentric departure velocity
            v_dep_sq = np.sum(v_dep_pred**2, axis=1, keepdims=True)
            energy_dep = 0.5 * v_dep_sq - self.mu_sun / r1_norm
            energy_theory = -self.mu_sun / (2.0 * a_transfer)
            res_energy = (energy_dep - energy_theory) ** 2
            total_res += self.energy_weight * res_energy

        if self.use_angular_momentum and r1_vecs is not None and r2_vecs is not None:
            # Angular momentum conservation: r1 x v_dep = r2 x v_arr
            h_dep = np.cross(r1_vecs, v_dep_pred)
            h_arr = np.cross(r2_vecs, v_arr_pred)
            res_h = np.sum((h_dep - h_arr) ** 2, axis=1, keepdims=True)
            total_res += self.angular_momentum_weight * res_h

        return total_res

    def train_on_dataset(
        self,
        x_data: np.ndarray,
        v_targets: np.ndarray,
        r1_norms: np.ndarray,
        r2_norms: np.ndarray,
        tof_seconds: np.ndarray,
        epochs: int = 50,
        batch_size: int = 128,
        r1_vecs: np.ndarray | None = None,
        r2_vecs: np.ndarray | None = None,
        v_planet_depart: np.ndarray | None = None,
        v_planet_arrive: np.ndarray | None = None,
    ) -> list[float]:
        """Train the PINN on a labeled transfer trajectory dataset."""
        n = x_data.shape[0]
        y_target = v_targets.astype(np.float32)
        if y_target.ndim == 1:
            y_target = y_target.reshape(-1, 1)

        r1_target = r1_norms.reshape(-1, 1).astype(np.float32)
        r2_target = r2_norms.reshape(-1, 1).astype(np.float32)
        tof_target = tof_seconds.reshape(-1, 1).astype(np.float32)

        loss_history = []

        for _ in range(epochs):
            indices = np.random.permutation(n)
            x_sh = x_data[indices]
            y_sh = y_target[indices]
            r1_sh = r1_target[indices]
            r2_sh = r2_target[indices]
            tof_sh = tof_target[indices]

            r1_v_sh = r1_vecs[indices] if r1_vecs is not None else None
            r2_v_sh = r2_vecs[indices] if r2_vecs is not None else None
            vp_dep_sh = v_planet_depart[indices] if v_planet_depart is not None else None
            vp_arr_sh = v_planet_arrive[indices] if v_planet_arrive is not None else None

            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, n, batch_size):
                xb = x_sh[i : i + batch_size]
                yb = y_sh[i : i + batch_size]
                r1b = r1_sh[i : i + batch_size]
                r2b = r2_sh[i : i + batch_size]
                tofb = tof_sh[i : i + batch_size]

                r1_vecs_b = r1_v_sh[i : i + batch_size] if r1_v_sh is not None else None
                r2_vecs_b = r2_v_sh[i : i + batch_size] if r2_v_sh is not None else None
                if vp_dep_sh is not None:
                    v_planet_dep_b = vp_dep_sh[i : i + batch_size]
                else:
                    v_planet_dep_b = np.zeros((xb.shape[0], 3), dtype=np.float32)
                if vp_arr_sh is not None:
                    v_planet_arr_b = vp_arr_sh[i : i + batch_size]
                else:
                    v_planet_arr_b = np.zeros((xb.shape[0], 3), dtype=np.float32)

                m = xb.shape[0]

                # Forward pass (outputs (batch, 6))
                pred = self.forward(xb)

                # Compute MSE loss (feasible only)
                if yb.shape[1] == 1:
                    # Target is 1D delta-v
                    v_dep_pred = pred[:, :3]
                    v_arr_pred = pred[:, 3:]
                    dv_dep = np.linalg.norm(v_dep_pred - v_planet_dep_b, axis=1, keepdims=True)
                    dv_arr = np.linalg.norm(v_arr_pred - v_planet_arr_b, axis=1, keepdims=True)
                    pred_dv = dv_dep + dv_arr
                    feasible_mask = (yb < 20.0).flatten()
                    count_feasible = np.sum(feasible_mask)
                    if count_feasible > 0:
                        mse_loss = float(np.mean((pred_dv[feasible_mask] - yb[feasible_mask]) ** 2))
                    else:
                        mse_loss = 0.0
                else:
                    # Compute corresponding delta-v to check feasibility
                    dv_from_yb = np.linalg.norm(
                        yb[:, :3] - v_planet_dep_b, axis=1
                    ) + np.linalg.norm(yb[:, 3:] - v_planet_arr_b, axis=1)
                    feasible_mask = dv_from_yb < 20.0
                    count_feasible = np.sum(feasible_mask)
                    if count_feasible > 0:
                        mse_loss = float(np.mean((pred[feasible_mask] - yb[feasible_mask]) ** 2))
                    else:
                        mse_loss = 0.0

                # Compute physics loss
                res = self.physics_residual(
                    xb, r1b, r2b, tofb, r1_vecs_b, r2_vecs_b, v_planet_dep_b, v_planet_arr_b
                )
                physics_loss = float(np.mean(res))

                total_loss = mse_loss + self.physics_weight * physics_loss
                epoch_loss += float(total_loss)
                num_batches += 1

                # Backpropagation
                dy = np.zeros_like(pred, dtype=np.float32)
                if count_feasible > 0:
                    if yb.shape[1] == 1:
                        # Gradient through delta-v norm back to 6D velocities
                        u_dep = (v_dep_pred - v_planet_dep_b) / (dv_dep + 1e-10)
                        u_arr = (v_arr_pred - v_planet_arr_b) / (dv_arr + 1e-10)
                        d_mse_d_pred_dv = 2.0 * (pred_dv - yb) / count_feasible
                        dy[feasible_mask, :3] = (d_mse_d_pred_dv * u_dep)[feasible_mask]
                        dy[feasible_mask, 3:] = (d_mse_d_pred_dv * u_arr)[feasible_mask]
                    else:
                        dy[feasible_mask] = (
                            2.0 * (pred[feasible_mask] - yb[feasible_mask]) / count_feasible
                        )

                # Physics loss gradients
                dy_physics = np.zeros_like(pred, dtype=np.float32)
                a_transfer = (r1b.reshape(-1, 1) + r2b.reshape(-1, 1)) / 2.0

                if self.use_vis_viva:
                    v_dep_pred = pred[:, :3]
                    v_inf_vec = v_dep_pred - v_planet_dep_b
                    v_inf_pred = np.linalg.norm(v_inf_vec, axis=1, keepdims=True)
                    u_dep = v_inf_vec / (v_inf_pred + 1e-10)
                    v_transfer_r1 = np.sqrt(
                        self.mu_sun * (2.0 / r1b.reshape(-1, 1) - 1.0 / a_transfer)
                    )
                    dy_vis = 2.0 * (v_inf_pred - v_transfer_r1) * u_dep
                    dy_physics[:, :3] += self.vis_viva_weight * dy_vis

                if self.use_energy:
                    v_dep_pred = pred[:, :3]
                    v_dep_sq = np.sum(v_dep_pred**2, axis=1, keepdims=True)
                    energy_dep = 0.5 * v_dep_sq - self.mu_sun / r1b.reshape(-1, 1)
                    energy_theory = -self.mu_sun / (2.0 * a_transfer)
                    dy_energy = 2.0 * (energy_dep - energy_theory) * v_dep_pred
                    dy_physics[:, :3] += self.energy_weight * dy_energy

                if self.use_angular_momentum and r1_vecs_b is not None and r2_vecs_b is not None:
                    v_dep_pred = pred[:, :3]
                    v_arr_pred = pred[:, 3:]
                    h_dep = np.cross(r1_vecs_b, v_dep_pred)
                    h_arr = np.cross(r2_vecs_b, v_arr_pred)
                    h_diff = h_dep - h_arr
                    dy_h_dep = 2.0 * np.cross(h_diff, r1_vecs_b)
                    dy_h_arr = -2.0 * np.cross(h_diff, r2_vecs_b)
                    dy_physics[:, :3] += self.angular_momentum_weight * dy_h_dep
                    dy_physics[:, 3:] += self.angular_momentum_weight * dy_h_arr

                dy += (self.physics_weight / m) * dy_physics

                # Backpropagate through layers (linear output layer derivative is 1.0)
                dz = dy

                dw_list = []
                db_list = []

                num_layers = len(self.weights)
                for layer_idx in reversed(range(num_layers)):
                    a_prev = self._activations[layer_idx]
                    dw = a_prev.T @ dz
                    db = np.sum(dz, axis=0, keepdims=True)

                    dw_list.append(dw)
                    db_list.append(db)

                    if layer_idx > 0:
                        da_prev = dz @ self.weights[layer_idx].T
                        z_prev = self._preacts[layer_idx - 1]
                        dz = da_prev * (z_prev > 0.0)

                dw_list.reverse()
                db_list.reverse()

                # Update parameters with gradient clipping
                for layer_idx in range(num_layers):
                    dw_clipped = np.clip(dw_list[layer_idx], -1.0, 1.0)
                    db_clipped = np.clip(db_list[layer_idx], -1.0, 1.0)

                    self.weights[layer_idx] -= self.lr * dw_clipped
                    self.biases[layer_idx] -= self.lr * db_clipped

            epoch_loss /= max(1, num_batches)
            loss_history.append(epoch_loss)

        self._trained = True
        return loss_history

    def predict(
        self,
        features: np.ndarray,
        v_planet_depart: np.ndarray | None = None,
        v_planet_arrive: np.ndarray | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> SurrogatePrediction:
        """Predict velocities and compute delta-v for a single features vector."""
        pred = self.forward(features.reshape(1, -1))[0]  # (6,)

        v_planet_dep = (
            v_planet_depart if v_planet_depart is not None else np.zeros(3, dtype=np.float32)
        )
        v_planet_arr = (
            v_planet_arrive if v_planet_arrive is not None else np.zeros(3, dtype=np.float32)
        )

        dv_dep = float(np.linalg.norm(pred[:3] - v_planet_dep))
        dv_arr = float(np.linalg.norm(pred[3:] - v_planet_arr))
        total_dv = dv_dep + dv_arr

        return SurrogatePrediction(
            prediction=total_dv,
            uncertainty=0.0,
            requires_physics_validation=True,
            mean=pred,
            variance=np.zeros(6, dtype=np.float32),
            std=np.zeros(6, dtype=np.float32),
            delta_v=total_dv,
        )

    def predict_batch(
        self,
        features: np.ndarray,
        v_planet_depart: np.ndarray | None = None,
        v_planet_arrive: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict total delta-v for a batch of features."""
        preds = self.forward(features)
        n_samples = features.shape[0]

        v_planet_dep = (
            v_planet_depart
            if v_planet_depart is not None
            else np.zeros((n_samples, 3), dtype=np.float32)
        )
        v_planet_arr = (
            v_planet_arrive
            if v_planet_arrive is not None
            else np.zeros((n_samples, 3), dtype=np.float32)
        )

        dv_dep = np.linalg.norm(preds[:, :3] - v_planet_dep, axis=1)
        dv_arr = np.linalg.norm(preds[:, 3:] - v_planet_arr, axis=1)
        return np.asarray(dv_dep + dv_arr, dtype=np.float32)

    def is_trained(self) -> bool:
        """Return True if the model has been trained."""
        return self._trained

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> SurrogateMetrics:
        """Evaluate performance on a test set (feasible samples only)."""
        feasible_mask = y_test < 20.0
        x_feas = x_test[feasible_mask]
        y_feas = y_test[feasible_mask]
        n_test_samples = int(np.sum(feasible_mask))

        if n_test_samples == 0:
            return SurrogateMetrics(
                auc_roc=0.0,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                n_test_samples=0,
            )

        preds = self.predict_batch(x_feas)
        errors = preds - y_feas
        abs_errors = np.abs(errors)

        within_1kms = float(np.mean(abs_errors < 1.0))
        within_05kms = float(np.mean(abs_errors < 0.5))

        return SurrogateMetrics(
            auc_roc=0.0,
            accuracy=within_1kms,
            precision=within_05kms,
            recall=0.0,
            n_test_samples=n_test_samples,
        )

    def is_likely_feasible(self, feat: np.ndarray, threshold: float = 12.0) -> bool:
        """Check if predicted delta-v is below the feasibility threshold."""
        pred_obj = self.predict(feat)
        return bool(pred_obj.prediction < threshold)


class LambertPINNEnsemble(NeuralSurrogate):
    """Ensemble of independent LambertPINN models for uncertainty estimation."""

    def __init__(
        self,
        hidden_dims: list[int] = [64, 64, 32],
        lr: float = 3e-3,
        physics_weight: float = 0.1,
        mu_sun: float = 1.32712440018e11,
        ensemble_size: int = 5,
        use_vis_viva: bool = True,
        use_energy: bool = True,
        use_angular_momentum: bool = True,
        vis_viva_weight: float = 1.0,
        energy_weight: float = 0.1,
        angular_momentum_weight: float = 0.1,
    ) -> None:
        self.ensemble_size = ensemble_size
        self.models = [
            LambertPINN(
                hidden_dims=hidden_dims,
                lr=lr,
                physics_weight=physics_weight,
                mu_sun=mu_sun,
                use_vis_viva=use_vis_viva,
                use_energy=use_energy,
                use_angular_momentum=use_angular_momentum,
                vis_viva_weight=vis_viva_weight,
                energy_weight=energy_weight,
                angular_momentum_weight=angular_momentum_weight,
            )
            for _ in range(ensemble_size)
        ]
        self._trained = False

    def train_on_dataset(
        self,
        x_data: np.ndarray,
        v_targets: np.ndarray,
        r1_norms: np.ndarray,
        r2_norms: np.ndarray,
        tof_seconds: np.ndarray,
        epochs: int = 50,
        batch_size: int = 128,
        r1_vecs: np.ndarray | None = None,
        r2_vecs: np.ndarray | None = None,
        v_planet_depart: np.ndarray | None = None,
        v_planet_arrive: np.ndarray | None = None,
    ) -> list[float]:
        """Train all ensemble members independently."""
        loss_histories = []
        for model in self.models:
            loss_history = model.train_on_dataset(
                x_data=x_data,
                v_targets=v_targets,
                r1_norms=r1_norms,
                r2_norms=r2_norms,
                tof_seconds=tof_seconds,
                epochs=epochs,
                batch_size=batch_size,
                r1_vecs=r1_vecs,
                r2_vecs=r2_vecs,
                v_planet_depart=v_planet_depart,
                v_planet_arrive=v_planet_arrive,
            )
            loss_histories.append(loss_history)
        self._trained = True
        avg_loss = [float(np.mean([lh[e] for lh in loss_histories])) for e in range(epochs)]
        return avg_loss

    def predict(
        self,
        features: np.ndarray,
        v_planet_depart: np.ndarray | None = None,
        v_planet_arrive: np.ndarray | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> SurrogatePrediction:
        """Run all models to compute mean and variance/uncertainty of velocity and delta-v."""
        preds_list = []
        for model in self.models:
            pred = model.forward(features.reshape(1, -1))[0]  # (6,)
            preds_list.append(pred)
        preds = np.array(preds_list)  # (ensemble_size, 6)

        mean_v = np.mean(preds, axis=0)  # (6,)
        var_v = np.var(preds, axis=0)  # (6,)
        std_v = np.sqrt(var_v)  # (6,)

        v_planet_dep = (
            v_planet_depart if v_planet_depart is not None else np.zeros(3, dtype=np.float32)
        )
        v_planet_arr = (
            v_planet_arrive if v_planet_arrive is not None else np.zeros(3, dtype=np.float32)
        )

        # Calculate delta-v for each ensemble member
        dvs = []
        for k in range(self.ensemble_size):
            v_dep_k = preds[k, :3]
            v_arr_k = preds[k, 3:]
            dv_dep = float(np.linalg.norm(v_dep_k - v_planet_dep))
            dv_arr = float(np.linalg.norm(v_arr_k - v_planet_arr))
            dvs.append(dv_dep + dv_arr)

        mean_dv = float(np.mean(dvs))
        std_dv = float(np.std(dvs))

        return SurrogatePrediction(
            prediction=mean_dv,
            uncertainty=std_dv,
            requires_physics_validation=True,
            mean=mean_v,
            variance=var_v,
            std=std_v,
            delta_v=mean_dv,
        )

    def predict_batch(
        self,
        features: np.ndarray,
        v_planet_depart: np.ndarray | None = None,
        v_planet_arrive: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict mean delta-v for a batch of inputs."""
        n_samples = features.shape[0]
        preds_list = []
        for model in self.models:
            pred = model.forward(features)  # (N, 6)
            preds_list.append(pred)
        preds = np.array(preds_list)  # (ensemble, N, 6)

        v_planet_dep = (
            v_planet_depart
            if v_planet_depart is not None
            else np.zeros((n_samples, 3), dtype=np.float32)
        )
        v_planet_arr = (
            v_planet_arrive
            if v_planet_arrive is not None
            else np.zeros((n_samples, 3), dtype=np.float32)
        )

        dvs = np.zeros((self.ensemble_size, n_samples), dtype=np.float32)
        for k in range(self.ensemble_size):
            v_dep_k = preds[k, :, :3]
            v_arr_k = preds[k, :, 3:]
            dv_dep = np.linalg.norm(v_dep_k - v_planet_dep, axis=1)
            dv_arr = np.linalg.norm(v_arr_k - v_planet_arr, axis=1)
            dvs[k] = dv_dep + dv_arr

        return np.asarray(np.mean(dvs, axis=0), dtype=np.float32)

    def is_trained(self) -> bool:
        """Return True if all models are trained."""
        return self._trained

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> SurrogateMetrics:
        """Evaluate performance on a test set using the ensemble mean prediction."""
        feasible_mask = y_test < 20.0
        x_feas = x_test[feasible_mask]
        y_feas = y_test[feasible_mask]
        n_test_samples = int(np.sum(feasible_mask))

        if n_test_samples == 0:
            return SurrogateMetrics(
                auc_roc=0.0,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                n_test_samples=0,
            )

        preds = self.predict_batch(x_feas)
        errors = preds - y_feas
        abs_errors = np.abs(errors)

        within_1kms = float(np.mean(abs_errors < 1.0))
        within_05kms = float(np.mean(abs_errors < 0.5))

        return SurrogateMetrics(
            auc_roc=0.0,
            accuracy=within_1kms,
            precision=within_05kms,
            recall=0.0,
            n_test_samples=n_test_samples,
        )

    def is_likely_feasible(self, feat: np.ndarray, threshold: float = 12.0) -> bool:
        """Check if ensemble predicted mean delta-v is below the feasibility threshold."""
        pred_obj = self.predict(feat)
        return bool(pred_obj.prediction < threshold)


class ActiveLearningManager:
    """Manages the active learning loop for neural trajectory surrogate models.

    Identifies regions of high model uncertainty, queries the physical simulator,
    grows the training dataset, and triggers retraining.
    """

    def __init__(
        self,
        surrogate: LambertPINNEnsemble | LambertPINN,
        kernel: PhysicsKernel,
        uncertainty_threshold: float = 1.0,
        retrain_every: int = 10,
        epochs: int = 20,
        batch_size: int = 32,
    ) -> None:
        self.surrogate = surrogate
        self.kernel = kernel
        self.uncertainty_threshold = uncertainty_threshold
        self.retrain_every = retrain_every
        self.epochs = epochs
        self.batch_size = batch_size

        # Training data buffer
        self.x_buffer: list[np.ndarray] = []
        self.v_buffer: list[np.ndarray] = []
        self.r1_norm_buffer: list[float] = []
        self.r2_norm_buffer: list[float] = []
        self.tof_buffer: list[float] = []

        self.r1_vec_buffer: list[np.ndarray] = []
        self.r2_vec_buffer: list[np.ndarray] = []
        self.v_planet_dep_buffer: list[np.ndarray] = []
        self.v_planet_arr_buffer: list[np.ndarray] = []

        self.new_samples_count = 0

    def query_and_learn(
        self,
        features: np.ndarray,
        origin_body: CelestialBody,
        destination_body: CelestialBody,
        dep_epoch: float,
        tof_seconds: float,
    ) -> float:
        """Query the surrogate. If uncertainty exceeds threshold, execute exact solve and learn.

        Parameters
        ----------
        features : np.ndarray
            8-dimensional feature vector.
        origin_body : CelestialBody
            The departure body.
        destination_body : CelestialBody
            The arrival body.
        dep_epoch : float
            Departure J2000 epoch in seconds.
        tof_seconds : float
            Time of flight in seconds.

        Returns
        -------
        float
            The delta-v of the transfer (either exact or predicted).
        """
        r1_state = self.kernel.get_body_state(origin_body, dep_epoch)
        r2_state = self.kernel.get_body_state(destination_body, dep_epoch + tof_seconds)

        v_planet_dep = r1_state.velocity
        v_planet_arr = r2_state.velocity

        pred_obj = self.surrogate.predict(
            features,
            v_planet_depart=v_planet_dep,
            v_planet_arrive=v_planet_arr,
        )

        if pred_obj.uncertainty > self.uncertainty_threshold:
            from astra.physics.lambert import find_best_transfer

            try:
                # Execute real Lambert solve
                sol = find_best_transfer(
                    r1=r1_state.position,
                    v1_body=v_planet_dep,
                    r2=r2_state.position,
                    v2_body=v_planet_arr,
                    tof=tof_seconds,
                    mu=getattr(self.kernel.ephemeris, "mu_sun", 1.32712440018e11),
                    max_revs=0,
                )
                v_dep_exact = sol.v1
                v_arr_exact = sol.v2
                exact_dv = sol.delta_v
            except Exception:
                # If exact solve fails, fallback to prediction
                return pred_obj.prediction

            # Use exact solution as ground truth and add to training buffers
            v_target = np.hstack([v_dep_exact, v_arr_exact])
            self.x_buffer.append(features)
            self.v_buffer.append(v_target)
            self.r1_norm_buffer.append(float(np.linalg.norm(r1_state.position)))
            self.r2_norm_buffer.append(float(np.linalg.norm(r2_state.position)))
            self.tof_buffer.append(tof_seconds)

            self.r1_vec_buffer.append(r1_state.position)
            self.r2_vec_buffer.append(r2_state.position)
            self.v_planet_dep_buffer.append(v_planet_dep)
            self.v_planet_arr_buffer.append(v_planet_arr)

            self.new_samples_count += 1

            # Trigger retraining if budget is met
            if self.new_samples_count >= self.retrain_every:
                self.retrain()
                self.new_samples_count = 0

            return exact_dv
        else:
            return pred_obj.prediction

    def retrain(self) -> None:
        """Retrain the surrogate model on the accumulated dataset in the training buffers."""
        if not self.x_buffer:
            return

        x_data = np.array(self.x_buffer, dtype=np.float32)
        v_targets = np.array(self.v_buffer, dtype=np.float32)
        r1_norms = np.array(self.r1_norm_buffer, dtype=np.float32)
        r2_norms = np.array(self.r2_norm_buffer, dtype=np.float32)
        tof_seconds = np.array(self.tof_buffer, dtype=np.float32)

        r1_vecs = np.array(self.r1_vec_buffer, dtype=np.float32)
        r2_vecs = np.array(self.r2_vec_buffer, dtype=np.float32)
        v_planet_dep = np.array(self.v_planet_dep_buffer, dtype=np.float32)
        v_planet_arr = np.array(self.v_planet_arr_buffer, dtype=np.float32)

        self.surrogate.train_on_dataset(
            x_data=x_data,
            v_targets=v_targets,
            r1_norms=r1_norms,
            r2_norms=r2_norms,
            tof_seconds=tof_seconds,
            epochs=self.epochs,
            batch_size=self.batch_size,
            r1_vecs=r1_vecs,
            r2_vecs=r2_vecs,
            v_planet_depart=v_planet_dep,
            v_planet_arrive=v_planet_arr,
        )
