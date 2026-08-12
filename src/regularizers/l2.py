"""
l2.py — L2 (Ridge) regularization.

Penalty:   R(w) = (λ/2) * ||w||²
Gradient:  ∇R(w) = λ * w

The bias term (last element of w if bias is appended, or separate scalar)
is typically NOT regularized.  This module regularizes all elements of w;
the caller is responsible for zeroing out the bias gradient if needed.
"""

import numpy as np


class L2Regularizer:
    """
    L2 (Ridge) regularization.

    Parameters
    ----------
    lambda_reg : regularization strength λ ≥ 0
    """

    def __init__(self, lambda_reg: float):
        if lambda_reg < 0:
            raise ValueError(f"lambda_reg must be ≥ 0, got {lambda_reg}")
        self.lambda_reg = lambda_reg

    def penalty(self, w: np.ndarray) -> float:
        """R(w) = (λ/2) ||w||²"""
        return 0.5 * self.lambda_reg * float(np.dot(w, w))

    def gradient(self, w: np.ndarray) -> np.ndarray:
        """∇R(w) = λ w"""
        return self.lambda_reg * w

    def __repr__(self) -> str:
        return f"L2Regularizer(lambda={self.lambda_reg})"
