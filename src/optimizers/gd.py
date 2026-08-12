"""
Gradient Descent (full-batch).

Update rule:
    w_{t+1} = w_t - η * ∇f(w_t)

For L1 regularization (non-smooth), uses the proximal gradient variant:
    w_half   = w_t - η * ∇f_smooth(w_t)     (gradient step on smooth part)
    w_{t+1}  = prox_{η λ ||·||_1}(w_half)   (soft-thresholding)

The step size η is either:
  - Fixed: FixedLR object (returns η = c/L, with optional decay)
  - Backtracking: ArmijoLineSearch object (searches along the gradient)
"""

import numpy as np
from src.optimizers.base import BaseOptimizer


class GradientDescent(BaseOptimizer):
    """Full-batch Gradient Descent with optional proximal step for L1."""

    name = "gd"

    def __init__(self, step_size, regularizer=None, use_proximal: bool = False):
        self.step_size = step_size
        self.regularizer = regularizer
        self.use_proximal = use_proximal

    def step(self, w, b, grad_w, grad_b, **kwargs):
        if hasattr(self.step_size, "search"):
            obj_fn = kwargs.get("obj_fn")
            eta = self.step_size.search(w, b, grad_w, grad_b, obj_fn)
        else:
            eta = self.step_size.lr
            self.step_size.step()

        w_new = w - eta * grad_w
        b_new = b - eta * grad_b

        if self.use_proximal and self.regularizer is not None:
            w_new = self.regularizer.prox(w_new, eta)

        return w_new, b_new

    def reset(self, w_shape):
        self.step_size.reset()

    def __repr__(self):
        return f"GradientDescent(step_size={self.step_size}, proximal={self.use_proximal})"

