"""
newton.py — Newton's method with Hessian damping for numerical stability.

Update rule:
    w_{t+1} = w_t - α * H(w_t)^{-1} ∇f(w_t)

where H(w_t) is the Hessian of the loss at w_t, and α is the step size.

Damping for numerical stability:
    H_damp = H + ε * I
to ensure H_damp is positive definite and invertible even when the true
Hessian is near-singular or ill-conditioned.

Step size:
  - Fixed (α=1 for pure Newton, or scaled): from FixedLR
  - Backtracking: Armijo line search along the Newton direction d = H^{-1} ∇f

Notes
-----
- Each step requires solving a d×d linear system (or inverting H_damp).
  We use np.linalg.solve(H_damp, grad) which is O(d³) but exact.
- Newton is NOT run with L1 regularization (non-smooth Hessian undefined).
- For L2 regularization, the Hessian gains an extra λI term (already included
  in hessian_fn if the model adds L2 to the Hessian).
"""

import numpy as np
from src.optimizers.base import BaseOptimizer


class Newton(BaseOptimizer):
    """
    Newton's method with Hessian damping.

    Parameters
    ----------
    step_size   : FixedLR | ArmijoLineSearch instance
    epsilon_damp: damping constant ε added to diagonal of Hessian
    """

    name = "newton"

    def __init__(self, step_size, epsilon_damp: float = 1e-6):
        self.step_size = step_size
        self.epsilon_damp = epsilon_damp

    def step(self, w, loss_fn, grad_fn, X, y, hessian_fn=None, **kwargs) -> np.ndarray:
        """
        Parameters
        ----------
        hessian_fn : callable (w, X, y) → ndarray of shape (d, d)
                     Must be provided; raises ValueError otherwise.
        """
        if hessian_fn is None:
            raise ValueError("Newton's method requires hessian_fn to be provided.")

        grad = grad_fn(w, X, y)
        H = hessian_fn(w, X, y)

        # Damping: H_damp = H + ε*I
        H_damp = H + self.epsilon_damp * np.eye(H.shape[0])

        # Solve H_damp @ d = grad  →  d = H_damp^{-1} grad (Newton direction)
        try:
            direction = np.linalg.solve(H_damp, grad)
        except np.linalg.LinAlgError:
            # Fallback to gradient descent step if solve fails
            direction = grad

        if hasattr(self.step_size, "search"):
            # Armijo along Newton direction
            eta = self.step_size.search(
                w, grad,
                loss_fn=lambda _w: loss_fn(_w, X, y),
                direction=direction,
            )
        else:
            eta = self.step_size.step()

        return w - eta * direction

    def reset(self) -> None:
        if hasattr(self.step_size, "reset"):
            self.step_size.reset()

    def __repr__(self) -> str:
        return f"Newton(epsilon_damp={self.epsilon_damp}, step_size={self.step_size})"
