"""
l1.py — L1 (Lasso) regularization with proximal operator.

Penalty:   R(w) = λ * ||w||_1
Gradient:  ∇R(w) = λ * sign(w)  [sub-gradient; undefined at 0]

L1 is non-smooth so standard gradient descent cannot be applied directly.
Instead, the proximal gradient method decomposes each update into:
  1. A gradient step on the smooth part (loss + L2 if combined):
       w_half = w - η * ∇f_smooth(w)
  2. Application of the proximal operator (soft-thresholding):
       prox(w_half) = sign(w_half) * max(|w_half| - η*λ, 0)

The proximal operator shrinks entries toward zero, setting small weights
to exactly zero (sparsity-inducing property of L1).
"""

import numpy as np


class L1Regularizer:
    """
    L1 (Lasso) regularization with soft-thresholding proximal operator.

    Parameters
    ----------
    lambda_reg : regularization strength λ ≥ 0
    """

    def __init__(self, lambda_reg: float):
        if lambda_reg < 0:
            raise ValueError(f"lambda_reg must be ≥ 0, got {lambda_reg}")
        self.lambda_reg = lambda_reg

    def penalty(self, w: np.ndarray) -> float:
        """R(w) = λ * ||w||_1"""
        return self.lambda_reg * float(np.sum(np.abs(w)))

    def subgradient(self, w: np.ndarray) -> np.ndarray:
        """Sub-gradient of L1: λ * sign(w).  (0 at w=0 by convention.)"""
        return self.lambda_reg * np.sign(w)

    def proximal(self, w: np.ndarray, eta: float) -> np.ndarray:
        """
        Soft-thresholding proximal operator.

            prox_{η λ ||·||_1}(w) = sign(w) * max(|w| - η*λ, 0)

        Parameters
        ----------
        w   : parameter vector after the gradient step (w_half)
        eta : current step size η

        Returns
        -------
        w_new : proximal-updated parameters with sparsity-inducing shrinkage
        """
        threshold = eta * self.lambda_reg
        return np.sign(w) * np.maximum(np.abs(w) - threshold, 0.0)

    def __repr__(self) -> str:
        return f"L1Regularizer(lambda={self.lambda_reg})"
