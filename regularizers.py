"""
regularizers.py
===============
SHARED module. Interface:

    reg.penalty(w)          -> scalar, added to reported loss for logging
    reg.grad(w)              -> smooth-part gradient contribution (L2, or 0 for pure L1)
    reg.prox(w, lr)          -> proximal step for the non-smooth part (identity if none)

L2 is handled via grad() (it's smooth, so plain gradient descent works).
L1 is handled via prox() (it's non-smooth, so needs a proximal step —
this is exactly ISTA / proximal gradient descent). Elastic Net does both:
L2 part goes into grad(), L1 part goes into prox().

train.py calls these generically:
    w = w - lr * (data_grad + reg.grad(w))   # smooth gradient step
    w = reg.prox(w, lr)                       # then proximal step (no-op if no L1)
"""

import numpy as np


class NoReg:
    name = "none"
    def penalty(self, w): return 0.0
    def grad(self, w): return np.zeros_like(w)
    def prox(self, w, lr): return w
    def hessian(self, w): return np.zeros_like(w)


class L2Reg:
    """Ridge: (lambda/2) * ||w||_2^2. Smooth + convex; makes the total
    objective STRONGLY convex when added to a convex loss -> improves
    conditioning, gives linear convergence rate guarantees for GD.
    """
    name = "l2"

    def __init__(self, lam: float = 1e-2):
        self.lam = lam

    def penalty(self, w):
        return 0.5 * self.lam * float(np.sum(w ** 2))

    def grad(self, w):
        return self.lam * w

    def prox(self, w, lr):
        return w  # no non-smooth part

    def hessian(self, w):
        return self.lam * np.ones_like(w)


class L1Reg:
    """Lasso: lambda * ||w||_1. Convex but NON-SMOOTH (kink at 0).
    Handled via proximal gradient (soft-thresholding), NOT plain
    subgradient descent, for cleaner/faster convergence -> this is the
    "why do we need proximal methods" story for your report.
    """
    name = "l1"

    def __init__(self, lam: float = 1e-2):
        self.lam = lam

    def penalty(self, w):
        return self.lam * float(np.sum(np.abs(w)))

    def grad(self, w):
        return np.zeros_like(w)  # non-smooth part handled in prox()

    def prox(self, w, lr):
        # soft-thresholding operator, threshold = lr * lambda
        thresh = lr * self.lam
        return np.sign(w) * np.maximum(np.abs(w) - thresh, 0.0)

    def hessian(self, w):
        return np.zeros_like(w)


class ElasticNetReg:
    """lambda1 * ||w||_1 + (lambda2/2) * ||w||_2^2. L2 part via grad(),
    L1 part via prox() -> combines both mechanisms above.
    """
    name = "elastic_net"

    def __init__(self, lam_l1: float = 1e-2, lam_l2: float = 1e-2):
        self.lam_l1 = lam_l1
        self.lam_l2 = lam_l2

    def penalty(self, w):
        return (self.lam_l1 * float(np.sum(np.abs(w)))
                + 0.5 * self.lam_l2 * float(np.sum(w ** 2)))

    def grad(self, w):
        return self.lam_l2 * w

    def prox(self, w, lr):
        thresh = lr * self.lam_l1
        return np.sign(w) * np.maximum(np.abs(w) - thresh, 0.0)

    def hessian(self, w):
        return self.lam_l2 * np.ones_like(w)


REG_REGISTRY = {
    "none": NoReg,
    "l2": L2Reg,
    "l1": L1Reg,
    "elastic_net": ElasticNetReg,
}
