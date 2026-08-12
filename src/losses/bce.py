"""
bce.py — Binary Cross-Entropy (BCE) loss.

Loss:     L(w) = -(1/n) Σ [ y_i log(σ(z_i)) + (1-y_i) log(1-σ(z_i)) ]
Gradient: ∇L(w) = (1/n) X^T (σ(z) - y)
Hessian:  H(w)  = (1/n) X^T diag(σ(z) * (1-σ(z))) X

where z = X @ w  and  σ is the sigmoid function.

Numerical stability:
  log(σ(z)) = -log(1 + exp(-z))   [use log-sum-exp trick internally]
"""

import numpy as np
from src.utils import sigmoid


class BCELoss:
    """Binary Cross-Entropy loss with analytic gradient and Hessian."""

    name = "bce"

    def __call__(
        self,
        w: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        return self.loss(w, X, y)

    def loss(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """Compute BCE loss (scalar)."""
        z = X @ w
        # Numerically stable: log σ(z) = -log(1+exp(-z)) = -softplus(-z)
        log_p  = -np.logaddexp(0, -z)          # log σ(z)
        log_1p = -np.logaddexp(0, z)           # log(1 - σ(z))
        return -float(np.mean(y * log_p + (1 - y) * log_1p))

    def gradient(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute gradient ∇L(w) = (1/n) X^T (σ(z) - y)."""
        z = X @ w
        residuals = sigmoid(z) - y
        return X.T @ residuals / X.shape[0]

    def hessian(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute Hessian H(w) = (1/n) X^T diag(s*(1-s)) X."""
        z = X @ w
        s = sigmoid(z)
        D = s * (1 - s)                   # shape (n,)
        return (X.T * D) @ X / X.shape[0]

    def __repr__(self) -> str:
        return "BCELoss()"
