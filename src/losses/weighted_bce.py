"""
weighted_bce.py — Weighted Binary Cross-Entropy loss.

Applies per-sample class weights to address class imbalance:

    L(w) = -(1/N) Σ [ w_pos * y_i * log(p_i) + w_neg * (1-y_i) * log(1-p_i) ]

where:
    p_i = σ(z_i),  z_i = x_i^T w
    w_pos = c_scale * R,   R = N_neg / N_pos   (computed from training data)
    w_neg = 1.0            (fixed)

The weight vector is broadcast over the sample dimension.

Gradient:
    ∇L(w) = (1/N) X^T diag(weights) (σ(z) - y)
    where weights_i = w_pos if y_i=1 else w_neg

Hessian:
    H(w) = (1/N) X^T diag(weights * s * (1-s)) X
"""

import numpy as np
from src.utils import sigmoid


class WeightedBCELoss:
    """
    Weighted Binary Cross-Entropy loss.

    Parameters
    ----------
    w_pos : weight for positive class (y=1), typically c_scale * R
    w_neg : weight for negative class (y=0), typically 1.0
    """

    name = "weighted_bce"

    def __init__(self, w_pos: float = 1.0, w_neg: float = 1.0):
        self.w_pos = w_pos
        self.w_neg = w_neg

    def _sample_weights(self, y: np.ndarray) -> np.ndarray:
        """Return per-sample weight vector based on class membership."""
        return np.where(y == 1, self.w_pos, self.w_neg)

    def __call__(self, w, X, y):
        return self.loss(w, X, y)

    def loss(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        z = X @ w
        log_p  = -np.logaddexp(0, -z)
        log_1p = -np.logaddexp(0, z)
        sample_w = self._sample_weights(y)
        return -float(np.mean(sample_w * (y * log_p + (1 - y) * log_1p)))

    def gradient(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        z = X @ w
        s = sigmoid(z)
        sample_w = self._sample_weights(y)
        residuals = sample_w * (s - y)
        return X.T @ residuals / X.shape[0]

    def hessian(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        z = X @ w
        s = sigmoid(z)
        sample_w = self._sample_weights(y)
        D = sample_w * s * (1 - s)
        return (X.T * D) @ X / X.shape[0]

    def __repr__(self) -> str:
        return f"WeightedBCELoss(w_pos={self.w_pos}, w_neg={self.w_neg})"
