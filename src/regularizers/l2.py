"""
L2 (Ridge) regularization: (λ/2) * ||w||_2^2

Smooth + convex; makes the total objective STRONGLY convex when added
to a convex loss -> improves conditioning, gives linear convergence
rate guarantees for GD.
"""

import numpy as np


class L2Regularizer:
    """L2 (Ridge) regularization: (λ/2) * ||w||_2^2."""

    def __init__(self, lam: float = 1e-2):
        self.lam = lam

    def penalty(self, w):
        """L(w) = (λ/2) * ||w||_2^2"""
        return 0.5 * self.lam * float(np.sum(w ** 2))

    def grad(self, w):
        """∇L(w) = λ * w"""
        return self.lam * w

    def prox(self, w, lr):
        """No non-smooth part; prox is identity."""
        return w

    def hessian(self, w):
        """Hessian is λ * I (diagonal)."""
        return self.lam * np.ones_like(w)

    def __repr__(self):
        return f"L2Regularizer(lam={self.lam})"
