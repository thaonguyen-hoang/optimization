"""
weighted_bce.py — Weighted Binary Cross-Entropy loss.

Logit-based interface: z = X @ w + b (pre-sigmoid logits), y in {0, 1}.
Applies per-sample class weights to address class imbalance.
"""

import numpy as np


def _sigmoid(z):
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


class WeightedBCELoss:
    """Cost-sensitive BCE: scales each example's loss by its class weight.

    w_pos, w_neg: e.g. inverse class frequency, or a chosen cost ratio.
    Still convex: a positive-weighted sum of convex terms is convex.
    """

    name = "weighted_bce"

    def __init__(self, w_pos: float = 1.0, w_neg: float = 1.0):
        self.w_pos = w_pos
        self.w_neg = w_neg

    def value(self, z, y):
        p = _sigmoid(z)
        eps = 1e-12
        p = np.clip(p, eps, 1 - eps)
        w = np.where(y == 1, self.w_pos, self.w_neg)
        return float(np.mean(w * -(y * np.log(p) + (1 - y) * np.log(1 - p))))

    def grad(self, z, y):
        p = _sigmoid(z)
        w = np.where(y == 1, self.w_pos, self.w_neg)
        return w * (p - y)

    def hessian(self, z, y):
        p = _sigmoid(z)
        w = np.where(y == 1, self.w_pos, self.w_neg)
        return w * p * (1 - p)

    def __repr__(self):
        return f"WeightedBCELoss(w_pos={self.w_pos}, w_neg={self.w_neg})"
