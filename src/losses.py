"""
losses.py

Logit-based interface: z = X @ w + b (pre-sigmoid logits), y in {0, 1}.

Loss:     L(z, y) = -(1/n) Σ [ y_i * log(σ(z_i)) + (1-y_i) * log(1-σ(z_i)) ]
Gradient: ∇L/∂z   = σ(z) - y
Hessian:  ∂²L/∂z² = diag(σ(z) * (1-σ(z)))

Numerical stability:
  log(σ(z)) = -log(1 + exp(-z))   [use logaddexp trick]
"""

import numpy as np

from src.utils import sigmoid


class BCELoss:
    """Standard binary cross-entropy. Convex and smooth in z."""

    name = "bce"

    def value(self, z, y):
        p = sigmoid(z)
        eps = 1e-12
        p = np.clip(p, eps, 1 - eps)
        return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))

    def grad(self, z, y):
        p = sigmoid(z)
        return p - y

    def hessian(self, z, y):
        p = sigmoid(z)
        return p * (1 - p)

    def __repr__(self):
        return "BCELoss()"


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
        p = sigmoid(z)
        eps = 1e-12
        p = np.clip(p, eps, 1 - eps)
        w = np.where(y == 1, self.w_pos, self.w_neg)
        return float(np.mean(w * -(y * np.log(p) + (1 - y) * np.log(1 - p))))

    def grad(self, z, y):
        p = sigmoid(z)
        w = np.where(y == 1, self.w_pos, self.w_neg)
        return w * (p - y)

    def hessian(self, z, y):
        p = sigmoid(z)
        w = np.where(y == 1, self.w_pos, self.w_neg)
        return w * p * (1 - p)

    def __repr__(self):
        return f"WeightedBCELoss(w_pos={self.w_pos}, w_neg={self.w_neg})"


class SquaredHingeLoss:
    """Squared hinge, margin form: max(0, 1 - y'*z)^2 where y' in {-1,+1}."""

    name = "squared_hinge"

    @staticmethod
    def _to_pm1(y):
        return 2 * y - 1

    def value(self, z, y):
        y_pm = self._to_pm1(y)
        margin = np.maximum(0.0, 1 - y_pm * z)
        return float(np.mean(margin ** 2))

    def grad(self, z, y):
        y_pm = self._to_pm1(y)
        margin = np.maximum(0.0, 1 - y_pm * z)
        return -2 * y_pm * margin

    def hessian(self, z, y):
        y_pm = self._to_pm1(y)
        margin = np.maximum(0.0, 1 - y_pm * z)
        return 2.0 * (margin > 0).astype(float)

    def __repr__(self):
        return "SquaredHingeLoss()"


_EPS = 1e-8


class FocalLoss:
    """
    Focal Loss for binary classification.

    Parameters
    ----------
    gamma : focusing parameter (gamma=0 reduces to weighted BCE)
    alpha : class balancing factor (weight for positive class)
    """

    name = "focal"

    def __init__(self, gamma: float = 2.0, alpha: float = 0.5):
        self.gamma = gamma
        self.alpha = alpha

    def _pt_and_alpha(self, s, y):
        """Per-sample confidence p_t and class weight a_t."""
        p_t = np.where(y == 1, s, 1.0 - s)
        a_t = np.where(y == 1, self.alpha, 1.0 - self.alpha)
        return p_t, a_t

    def value(self, z, y):
        s = sigmoid(z)
        p_t, a_t = self._pt_and_alpha(s, y)
        fw = (1.0 - p_t) ** self.gamma
        return -float(np.mean(a_t * fw * np.log(p_t + _EPS)))

    def grad(self, z, y):
        s = sigmoid(z)
        p_t, a_t = self._pt_and_alpha(s, y)
        g = self.gamma

        fw = (1.0 - p_t) ** g
        log_pt = np.log(p_t + _EPS)
        one_minus_pt = np.clip(1.0 - p_t, _EPS, 1.0)

        if g > 0:
            dFL_dpt = a_t * (g * one_minus_pt ** (g - 1.0) * log_pt - fw / (p_t + _EPS))
        else:
            dFL_dpt = -a_t * fw / (p_t + _EPS)

        dpt_dz = np.where(y == 1, 1.0, -1.0) * s * (1.0 - s)
        return dFL_dpt * dpt_dz

    def hessian(self, z, y):
        raise NotImplementedError(
            "Focal loss Hessian is not implemented. "
            "Newton's method is not used with focal loss."
        )

    def __repr__(self):
        return f"FocalLoss(gamma={self.gamma}, alpha={self.alpha})"


LOSS_REGISTRY = {
    "bce": BCELoss,
    "weighted_bce": WeightedBCELoss,
    "squared_hinge": SquaredHingeLoss,
    "focal": FocalLoss,
}