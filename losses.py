"""
losses.py
=========
SHARED module. Every loss follows the same interface so optimizers.py
and train.py can treat them interchangeably:

    loss_value          = loss.value(z, y)          -> scalar (mean over batch)
    grad_wrt_logits     = loss.grad(z, y)            -> array, same shape as z

where z = X @ w + b are the RAW LOGITS (pre-sigmoid), and y is in {0,1}.

Working with logits (not probabilities) keeps gradients numerically
stable and keeps the chain rule to parameters trivial:
    dL/dw = (1/n) * X.T @ grad_wrt_logits(z, y)
    dL/db = mean(grad_wrt_logits(z, y))

Each person only needs to import THEIR assigned loss class, but keep
this file shared/untouched so the interface stays consistent for the
final merge.
"""

import numpy as np


def _sigmoid(z):
    # numerically stable sigmoid
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
        return p - y  # classic logistic gradient w.r.t. logits


class WeightedBCELoss:
    """Cost-sensitive BCE: scales each example's loss by its class weight.

    w_pos, w_neg: e.g. inverse class frequency, or a chosen cost ratio
    (false-negative-costlier framing -> w_pos > w_neg).
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


class FocalLoss:
    """Focal loss (Lin et al. 2017). NON-CONVEX in z for gamma > 0 —
    use this deliberately as your convex-vs-non-convex contrast case,
    not as a drop-in convex alternative.

    alpha: class weight (like weighted BCE's w_pos/w_neg combined into one term)
    gamma: focusing parameter; gamma=0 reduces to (weighted) BCE.
    """

    name = "focal"

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        self.alpha = alpha
        self.gamma = gamma

    def value(self, z, y):
        p = _sigmoid(z)
        eps = 1e-12
        p = np.clip(p, eps, 1 - eps)
        p_t = np.where(y == 1, p, 1 - p)
        alpha_t = np.where(y == 1, self.alpha, 1 - self.alpha)
        loss = -alpha_t * (1 - p_t) ** self.gamma * np.log(p_t)
        return float(np.mean(loss))

    def grad(self, z, y):
        # Numerically-differentiated-free closed form.
        p = _sigmoid(z)
        eps = 1e-12
        p = np.clip(p, eps, 1 - eps)
        p_t = np.where(y == 1, p, 1 - p)
        alpha_t = np.where(y == 1, self.alpha, 1 - self.alpha)
        sign = np.where(y == 1, 1.0, -1.0)
        # d/dz [-(1-p_t)^gamma log(p_t)]
        # dp_t/dz = sign * p * (1-p)
        dpt_dz = sign * p * (1 - p)
        term1 = self.gamma * (1 - p_t) ** (self.gamma - 1) * np.log(p_t) * dpt_dz
        term2 = -(1 - p_t) ** self.gamma * (1.0 / p_t) * dpt_dz
        return alpha_t * (term1 + term2)


class SquaredHingeLoss:
    """Squared hinge, margin form: max(0, 1 - y'*z)^2 where y' in {-1,+1}.
    Convex AND smooth (unlike plain hinge) in z. Requires y' in {-1,+1}
    internally even though the rest of the pipeline uses y in {0,1}.
    """

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


LOSS_REGISTRY = {
    "bce": BCELoss,
    "weighted_bce": WeightedBCELoss,
    "focal": FocalLoss,
    "squared_hinge": SquaredHingeLoss,
}
