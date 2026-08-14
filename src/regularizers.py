"""Regularizers for smooth and composite objectives."""

from abc import ABC, abstractmethod

import numpy as np


class Regularizer(ABC):
    name = "regularizer"
    is_smooth = True
    lam = 0.0

    @abstractmethod
    def penalty(self, w): ...

    @abstractmethod
    def grad(self, w): ...

    @abstractmethod
    def prox(self, w, threshold): ...


class NoReg(Regularizer):
    """No regularization: r(w) = 0."""

    name = "none"

    def penalty(self, w):
        return 0.0

    def grad(self, w):
        return np.zeros_like(w)

    def prox(self, w, threshold):
        return np.asarray(w)

    def __repr__(self):
        return "NoReg()"


class L2Reg(Regularizer):
    """Ridge: (lambda/2) * ||w||_2^2. Smooth and strongly convex."""

    name = "l2"

    def __init__(self, lam: float = 1e-2):
        if lam < 0:
            raise ValueError("lam must be nonnegative")
        self.lam = float(lam)

    def penalty(self, w):
        return 0.5 * self.lam * float(np.dot(w, w))

    def grad(self, w):
        return self.lam * w

    def prox(self, w, threshold):
        """L2 has no non-smooth part; prox is the identity."""
        return np.asarray(w)

    def __repr__(self):
        return f"L2Reg(lam={self.lam})"


class L1Reg(Regularizer):
    """Lasso: lambda * ||w||_1. Non-smooth; handled via proximal operator."""

    name = "l1"
    is_smooth = False

    def __init__(self, lam: float = 1e-2):
        if lam < 0:
            raise ValueError("lam must be nonnegative")
        self.lam = float(lam)

    def penalty(self, w):
        return self.lam * float(np.sum(np.abs(w)))

    def grad(self, w):
        raise ValueError("L1 is non-smooth; use the proximal operator instead.")

    def prox(self, w, threshold):
        """Soft-thresholding operator: sign(w) * max(|w| - threshold, 0)."""
        return np.sign(w) * np.maximum(np.abs(w) - threshold, 0.0)

    def __repr__(self):
        return f"L1Reg(lam={self.lam})"


def build_regularizer(name: str, lam: float = 0.0) -> Regularizer:
    """Instantiate a regularizer by name."""
    key = name.lower()
    if key == "none":
        return NoReg()
    if key == "l2":
        return L2Reg(lam)
    if key == "l1":
        return L1Reg(lam)
    raise ValueError(f"Unknown regularizer: {name!r}")
