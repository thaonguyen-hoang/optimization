"""
Nesterov Accelerated Gradient (NAG) descent.

Standard momentum update (Heavy Ball):
    v_{t+1} = μ * v_t - η * ∇f(w_t)
    w_{t+1} = w_t + v_{t+1}

Nesterov's correction: evaluate the gradient at the "lookahead" point:
    y_t      = w_t + μ * v_t              (lookahead point)
    v_{t+1}  = μ * v_t - η * ∇f(y_t)
    w_{t+1}  = w_t + v_{t+1}

Proximal variant for L1 (ISTA/FISTA-style):
    y_t     = w_t + μ * v_t
    g       = ∇f_smooth(y_t)
    w_half  = y_t - η * g
    w_{t+1} = prox_{η λ}(w_half)
    v_{t+1} = w_{t+1} - w_t

Backtracking: the Armijo condition is evaluated at the lookahead point y_t.
"""

import numpy as np
from src.optimizers.base import BaseOptimizer


class NAG(BaseOptimizer):
    """Nesterov Accelerated Gradient descent."""

    name = "nag"

    def __init__(self, step_size, momentum: float = 0.9, regularizer=None, use_proximal: bool = False):
        self.step_size = step_size
        self.momentum = momentum
        self.regularizer = regularizer
        self.use_proximal = use_proximal
        self._v = None
        self._vb = 0.0

    def lookahead(self, w, b):
        """Return Nesterov lookahead point for gradient evaluation."""
        if self._v is None:
            return w, b
        y_look = w + self.momentum * self._v
        y_look_b = b + self.momentum * self._vb
        return y_look, y_look_b

    def step(self, w, b, grad_w, grad_b, **kwargs):
        if self._v is None:
            self._v = np.zeros_like(w)
            self._vb = 0.0

        if hasattr(self.step_size, "search"):
            obj_fn = kwargs.get("obj_fn")
            y_look, y_look_b = self.lookahead(w, b)
            eta = self.step_size.search(y_look, y_look_b, grad_w, grad_b, obj_fn)
        else:
            eta = self.step_size.lr
            self.step_size.step()

        y_look, y_look_b = self.lookahead(w, b)
        w_half = y_look - eta * grad_w
        b_new = y_look_b - eta * grad_b

        if self.use_proximal and self.regularizer is not None:
            w_new = self.regularizer.prox(w_half, eta)
        else:
            w_new = w_half

        self._v = w_new - w
        self._vb = b_new - b
        return w_new, b_new

    def reset(self, w_shape):
        self._v = np.zeros(w_shape)
        self._vb = 0.0
        self.step_size.reset()

    def __repr__(self):
        return f"NAG(momentum={self.momentum}, step_size={self.step_size}, proximal={self.use_proximal})"

