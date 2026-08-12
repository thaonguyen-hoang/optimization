"""
Armijo backtracking line search.

Finds the largest step size α = alpha_init * beta^k satisfying the
Armijo sufficient-decrease condition:

    f(w - α * d) ≤ f(w) - c_armijo * α * ∇f(w)^T d

where d is the descent direction (typically d = ∇f for gradient descent).

For gradient descent: d = grad, so ∇f^T d = ||grad||².

Notes:
- Newton / L-BFGS: pass direction via kwargs
- SGD backtracking is theoretically unsound (stochastic gradient breaks Armijo guarantee)
"""

import numpy as np


class ArmijoLineSearch:
    """Armijo backtracking line search."""

    def __init__(self, alpha_init: float = 1.0, beta: float = 0.5, c_armijo: float = 1e-4, max_iter: int = 50):
        self.alpha_init = alpha_init
        self.beta = beta
        self.c_armijo = c_armijo
        self.max_iter = max_iter

    def search(self, w, b, grad_w, grad_b, obj_fn, direction=None):
        """Run backtracking and return the accepted step size α.

        Parameters
        ----------
        w, b       : current parameters
        grad_w, grad_b : gradients
        obj_fn     : callable f(w, b) → scalar loss
        direction  : descent direction (joint w and b), default: joint grad

        Returns
        -------
        alpha : accepted step size
        """
        grad_joint = np.concatenate([grad_w, [grad_b]])
        if direction is None:
            direction = grad_joint

        f0 = obj_fn(w, b)
        slope = float(np.dot(grad_joint, direction))

        alpha = self.alpha_init
        for _ in range(self.max_iter):
            w_new = w - alpha * direction[:-1]
            b_new = b - alpha * direction[-1]
            if obj_fn(w_new, b_new) <= f0 - self.c_armijo * alpha * slope:
                return alpha
            alpha *= self.beta

        return alpha

    def __repr__(self):
        return f"ArmijoLineSearch(alpha_init={self.alpha_init}, beta={self.beta}, c_armijo={self.c_armijo})"

