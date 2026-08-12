"""
Newton's method with Hessian damping for numerical stability.

Update rule:
    w_{t+1} = w_t - α * H(w_t)^{-1} ∇f(w_t)

where H(w_t) is the Hessian of the loss at w_t, and α is the step size.

Damping for numerical stability:
    H_damp = H + ε * I
to ensure H_damp is positive definite and invertible.

Notes:
- Each step requires solving a d×d linear system (O(d³) but exact).
- Newton is NOT run with L1 regularization (non-smooth Hessian undefined).
- For L2 regularization, the Hessian gains an extra λI term.
"""

import numpy as np
from src.optimizers.base import BaseOptimizer


class Newton(BaseOptimizer):
    """Newton's method with Hessian damping."""

    name = "newton"
    requires_hessian = True

    def __init__(self, step_size, epsilon_damp: float = 1e-6):
        self.step_size = step_size
        self.epsilon_damp = epsilon_damp

    def step(self, w, b, grad_w, grad_b, **kwargs):
        hess_joint = kwargs.get("hess_joint")
        if hess_joint is None:
            raise ValueError("Newton requires 'hess_joint' in kwargs.")

        grad_joint = np.concatenate([grad_w, [grad_b]])

        diag_max = float(np.max(np.diag(hess_joint))) if len(hess_joint) > 0 else 1.0
        ridge = max(1e-8, self.epsilon_damp * max(diag_max, 1.0))
        H_reg = hess_joint + ridge * np.eye(len(grad_joint))

        try:
            step_joint = np.linalg.solve(H_reg, grad_joint)
        except np.linalg.LinAlgError:
            step_joint = np.linalg.lstsq(H_reg, grad_joint, rcond=None)[0]

        if hasattr(self.step_size, "search"):
            obj_fn = kwargs.get("obj_fn")
            eta = self.step_size.search(w, b, grad_w, grad_b, obj_fn)
        else:
            eta = self.step_size.lr
            self.step_size.step()

        new_w = w - eta * step_joint[:-1]
        new_b = b - eta * step_joint[-1]

        return new_w, new_b

    def reset(self, w_shape):
        self.step_size.reset()

    def __repr__(self):
        return f"Newton(epsilon_damp={self.epsilon_damp}, step_size={self.step_size})"

