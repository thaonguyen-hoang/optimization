"""
squared_hinge.py — Squared Hinge loss for binary classification.

Labels are assumed to be in {0, 1}; internally they are mapped to {-1, +1}.

Loss:     L(w) = (1/n) Σ max(0, 1 - ŷ_i * z_i)²
Gradient: ∇L(w) = -(2/n) X^T [ŷ * max(0, 1 - ŷ*z)]

where:
    z_i = x_i^T w          (raw logit / margin)
    ŷ_i = 2*y_i - 1        (mapped to {-1, +1})

Notes
-----
- This is a smooth loss (unlike standard hinge), so standard GD/NAG/SGD
  and Newton's method can all be applied without special handling.
- The Hessian is:
    H(w) = (2/n) X^T diag(1_{ŷ*z < 1}) X
  (diagonal indicator selects the samples in the margin violation region)
"""

import numpy as np


class SquaredHingeLoss:
    """Squared Hinge loss (smooth surrogate for SVM-style classification)."""

    name = "squared_hinge"

    def _margins(self, w: np.ndarray, X: np.ndarray, y: np.ndarray):
        """Return ŷ (±1 labels) and z (raw logits)."""
        y_signed = 2.0 * y - 1.0          # {0,1} → {-1,+1}
        z = X @ w
        return y_signed, z

    def __call__(self, w, X, y):
        return self.loss(w, X, y)

    def loss(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        y_signed, z = self._margins(w, X, y)
        margins = np.maximum(0.0, 1.0 - y_signed * z)
        return float(np.mean(margins ** 2))

    def gradient(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        y_signed, z = self._margins(w, X, y)
        margins = np.maximum(0.0, 1.0 - y_signed * z)
        # d/dw max(0, 1-ŷz)² = -2ŷ * max(0, 1-ŷz)
        coeff = -2.0 * y_signed * margins       # shape (n,)
        return X.T @ coeff / X.shape[0]

    def hessian(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        y_signed, z = self._margins(w, X, y)
        # Second derivative is 2 for samples with ŷ*z < 1, else 0
        D = 2.0 * (y_signed * z < 1.0).astype(float)   # shape (n,)
        return (X.T * D) @ X / X.shape[0]

    def __repr__(self) -> str:
        return "SquaredHingeLoss()"
