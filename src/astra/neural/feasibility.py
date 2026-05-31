"""Neural feasibility classifier for trajectory validation using pure NumPy."""
from __future__ import annotations

import numpy as np


class FeasibilityClassifier:
    """Lightweight pure NumPy neural feasibility classifier for trajectory validation."""

    def __init__(self) -> None:
        # Initialize weights and biases using He initialization
        self.w1 = np.random.randn(8, 32).astype(np.float32) * np.sqrt(2.0 / 8, dtype=np.float32)
        self.b1 = np.zeros((1, 32), dtype=np.float32)
        
        self.w2 = np.random.randn(32, 16).astype(np.float32) * np.sqrt(2.0 / 32, dtype=np.float32)
        self.b2 = np.zeros((1, 16), dtype=np.float32)
        
        self.w3 = np.random.randn(16, 1).astype(np.float32) * np.sqrt(2.0 / 16, dtype=np.float32)
        # Optimistic Bias Initialization: default output probability is P ~ 0.88 (Sigmoid(2.0))
        self.b3 = np.ones((1, 1), dtype=np.float32) * 2.0
        
        self.lr = 0.01
        
        # Cache for backpropagation
        self.z1 = np.zeros((1, 32), dtype=np.float32)
        self.a1 = np.zeros((1, 32), dtype=np.float32)
        self.z2 = np.zeros((1, 16), dtype=np.float32)
        self.a2 = np.zeros((1, 16), dtype=np.float32)
        self.z3 = np.zeros((1, 1), dtype=np.float32)

    @property
    def requires_physics_validation(self) -> bool:
        return True

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform standard forward pass through MLP network layers."""
        self.z1 = x @ self.w1 + self.b1
        self.a1 = np.maximum(0.0, self.z1)  # ReLU
        
        self.z2 = self.a1 @ self.w2 + self.b2
        self.a2 = np.maximum(0.0, self.z2)  # ReLU
        
        self.z3 = self.a2 @ self.w3 + self.b3
        # Sigmoid activation with numerical clipping
        self.a3 = 1.0 / (1.0 + np.exp(-np.clip(self.z3, -20.0, 20.0)))
        return self.a1, self.a2, self.a3

    def train_on_dataset(
        self,
        x_data: np.ndarray,
        y: np.ndarray,
        epochs: int = 30,
        batch_size: int = 128
    ) -> None:
        """Pretrain the neural network model on generated physical samples using standard GD."""
        n = x_data.shape[0]
        y_target = y.reshape(-1, 1)
        
        for _ in range(epochs):
            indices = np.random.permutation(n)
            x_shuffled = x_data[indices]
            y_shuffled = y_target[indices]
            
            for i in range(0, n, batch_size):
                xb = x_shuffled[i : i + batch_size]
                yb = y_shuffled[i : i + batch_size]
                m = xb.shape[0]
                
                a1, a2, pred = self.forward(xb)
                
                # Backpropagate gradients
                dz3 = pred - yb
                dw3 = a2.T @ dz3 / m
                db3 = np.mean(dz3, axis=0, keepdims=True)
                
                da2 = dz3 @ self.w3.T
                dz2 = da2 * (self.z2 > 0.0)
                dw2 = a1.T @ dz2 / m
                db2 = np.mean(dz2, axis=0, keepdims=True)
                
                da1 = dz2 @ self.w2.T
                dz1 = da1 * (self.z1 > 0.0)
                dw1 = xb.T @ dz1 / m
                db1 = np.mean(dz1, axis=0, keepdims=True)
                
                # Parameter updates
                self.w3 -= self.lr * dw3
                self.b3 -= self.lr * db3
                self.w2 -= self.lr * dw2
                self.b2 -= self.lr * db2
                self.w1 -= self.lr * dw1
                self.b1 -= self.lr * db1

    def is_likely_feasible(self, feat: np.ndarray) -> bool:
        """Fast prediction with 20% epsilon-greedy exploration to ensure robust search coverage."""
        if np.random.random() < 0.20:
            return True
            
        _, _, pred = self.forward(feat.reshape(1, -1))
        return bool(pred[0, 0] >= 0.3)

    def update(self, feat: np.ndarray, val: float) -> None:
        """Online single-step update based on physical simulator feedback."""
        xb = feat.reshape(1, -1)
        yb = np.array([[val]], dtype=np.float32)
        
        # Forward pass
        a1, a2, pred = self.forward(xb)
        
        # Backpropagate gradients for single sample
        dz3 = pred - yb
        dw3 = a2.T @ dz3
        db3 = dz3
        
        da2 = dz3 @ self.w3.T
        dz2 = da2 * (self.z2 > 0.0)
        dw2 = a1.T @ dz2
        db2 = dz2
        
        da1 = dz2 @ self.w2.T
        dz1 = da1 * (self.z1 > 0.0)
        dw1 = xb.T @ dz1
        db1 = dz1
        
        # Parameter updates
        self.w3 -= self.lr * dw3
        self.b3 -= self.lr * db3
        self.w2 -= self.lr * dw2
        self.b2 -= self.lr * db2
        self.w1 -= self.lr * dw1
        self.b1 -= self.lr * db1
