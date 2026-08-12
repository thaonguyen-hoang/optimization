"""
bce.py — Binary Cross-Entropy (BCE) loss.

Logit-based interface: z = X @ w + b (pre-sigmoid logits), y in {0, 1}.

Loss:     L(z, y) = -(1/n) Σ [ y_i * log(σ(z_i)) + (1-y_i) * log(1-σ(z_i)) ]
Gradient: ∇L/∂z   = σ(z) - y
Hessian:  ∂²L/∂z² = diag(σ(z) * (1-σ(z)))

Numerical stability:
  log(σ(z)) = -log(1 + exp(-z))   [use logaddexp trick]
"""

import numpy as np


def _sigmoid(z):
    """Numerically stable sigmoid."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


class BCELoss:
    """Standard binary cross-entropy. Convex and smooth in z."""

    name = "bce"

    def value(self, z, y):
        p = _sigmoid(z)
        eps = 1e-12
        p = np.clip(p, eps, 1 - eps)
        return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))

    def grad(self, z, y):
        p = _sigmoid(z)
        return p - y

    def hessian(self, z, y):
        p = _sigmoid(z)
        return p * (1 - p)

    def __repr__(self):
        return "BCELoss()"
