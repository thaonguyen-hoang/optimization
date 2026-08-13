import numpy as np


class L1Regularizer:
    """
    L1 (Lasso) regularization: λ * ||w||_1

    Convex but non-smooth (kink at 0). Handled via proximal gradient
    (soft-thresholding).
    """

    def __init__(self, lam: float = 1e-2):
        self.lam = lam

    def penalty(self, w):
        """L(w) = λ * ||w||_1"""
        return self.lam * float(np.sum(np.abs(w)))

    def grad(self, w):
        """Non-smooth part; subgradient is zero (handled in prox())."""
        return np.zeros_like(w)

    def prox(self, w, lr):
        """Soft-thresholding operator, threshold = lr * λ"""
        thresh = lr * self.lam
        return np.sign(w) * np.maximum(np.abs(w) - thresh, 0.0)

    def hessian(self, w):
        """L1 Hessian is zero (non-smooth, prox handles it)."""
        return np.zeros_like(w)

    def __repr__(self):
        return f"L1Regularizer(lam={self.lam})"


class L2Regularizer:
    """
    L2 (Ridge) regularization: (λ/2) * ||w||_2^2

    Smooth + convex; makes the total objective STRONGLY convex when added
    to a convex loss -> improves conditioning, gives linear convergence
    rate guarantees for GD.
    """

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


REGISTRY = {
    "none": None,
    "l1": L1Regularizer,
    "l2": L2Regularizer,
}