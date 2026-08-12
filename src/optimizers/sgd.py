"""
Stochastic Gradient Descent (mini-batch and single-sample).

One SGD "step" processes a single mini-batch (or one sample when batch_size=1).
One "epoch" = ceil(n / batch_size) such steps.

Update rule per batch (X_b, y_b):
    g_b    = ∇f(w; X_b, y_b)    (stochastic gradient)
    w_{t+1} = w_t - η_t * g_b

Proximal variant for L1:
    w_half   = w_t - η_t * g_b
    w_{t+1}  = prox_{η_t λ}(w_half)

Note: Backtracking + SGD is theoretically unsound (stochastic gradient
makes the Armijo guarantee meaningless).
"""

import numpy as np
from src.optimizers.base import BaseOptimizer


class SGD(BaseOptimizer):
    """Mini-batch (or single-sample) Stochastic Gradient Descent."""

    name = "sgd"

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
        return f"SGD(step_size={self.step_size}, proximal={self.use_proximal})"

