"""
regularizers.py
===============
Regularizers contribute to the objective F(w) = f(w) + r(w).

If r(w) is smooth (None, L2):
    is_smooth = True
    reg.grad(w) returns \nabla r(w).
    reg.prox(w, thresh) returns w (identity).

If r(w) is non-smooth (L1/Lasso):
    is_smooth = False
    reg.grad(w) raises ValueError (subgradient not supported).
    reg.prox(w, thresh) computes soft-thresholding with threshold.
"""

import numpy as np


class NoReg:
    name = "none"
    is_smooth = True

    def __init__(self):
        self.lam = 0.0

    def penalty(self, w):
        return 0.0

    def grad(self, w):
        return np.zeros_like(w)

    def prox(self, w, threshold):
        return w


class L2Reg:
    """Ridge: (lambda/2) * ||w||_2^2. Smooth."""
    name = "l2"
    is_smooth = True

    def __init__(self, lam: float = 1e-2):
        self.lam = lam

    def penalty(self, w):
        return 0.5 * self.lam * float(np.sum(w ** 2))

    def grad(self, w):
        return self.lam * w

    def prox(self, w, threshold):
        return w


class L1Reg:
    """Lasso: lambda * ||w||_1. Non-smooth.
    Uses proximal operator (soft-thresholding). Subgradient is omitted.
    """
    name = "l1"
    is_smooth = False

    def __init__(self, lam: float = 1e-2):
        self.lam = lam

    def penalty(self, w):
        return self.lam * float(np.sum(np.abs(w)))

    def grad(self, w):
        raise ValueError("L1 regularizer does not support gradient. Use proximal operator.")

    def prox(self, w, threshold):
        """Soft-thresholding operator."""
        return np.sign(w) * np.maximum(np.abs(w) - threshold, 0.0)


def build_regularizer(name: str, lam: float = 0.0):
    name = name.lower()
    if name == "none":
        return NoReg()
    if name == "l2":
        return L2Reg(lam=lam)
    if name == "l1":
        return L1Reg(lam=lam)
    raise ValueError(f"Unknown regularizer: {name}")
