"""
armijo.py — Armijo backtracking line search.

Finds the largest step size α = alpha_init * beta^k satisfying the
Armijo sufficient-decrease condition:

    f(w - α * d) ≤ f(w) - c_armijo * α * ∇f(w)^T d

where d is the descent direction (typically d = ∇f for gradient descent).

For gradient descent: d = grad, so ∇f^T d = ||grad||².

Notes
-----
- Newton / L-BFGS do NOT use this module; they rely on their own internal
  line searches (damped Newton step and scipy's internal Wolfe conditions).
- SGD backtracking is supported but flagged as theoretically unsound
  (stochastic gradient makes the Armijo guarantee meaningless).
"""

import numpy as np
from typing import Callable


class ArmijoLineSearch:
    """
    Armijo backtracking line search.

    Parameters
    ----------
    alpha_init : initial step size (default 1.0, as recommended)
    beta       : reduction factor per backtrack step (0 < β < 1)
    c_armijo   : sufficient decrease constant (0 < c < 1, typically 1e-4)
    max_iter   : maximum number of backtrack iterations
    """

    def __init__(
        self,
        alpha_init: float = 1.0,
        beta: float = 0.5,
        c_armijo: float = 1e-4,
        max_iter: int = 50,
    ):
        self.alpha_init = alpha_init
        self.beta = beta
        self.c_armijo = c_armijo
        self.max_iter = max_iter

    def search(
        self,
        w: np.ndarray,
        grad: np.ndarray,
        loss_fn: Callable[[np.ndarray], float],
        direction: np.ndarray | None = None,
    ) -> float:
        """
        Run backtracking and return the accepted step size α.

        Parameters
        ----------
        w         : current parameter vector
        grad      : gradient at w (∇f(w))
        loss_fn   : callable f(w) → scalar loss (evaluated on the same
                    data the gradient was computed on)
        direction : descent direction d (default: grad, i.e. steepest descent)

        Returns
        -------
        alpha : accepted step size
        """
        if direction is None:
            direction = grad  # steepest descent: d = ∇f

        f0 = loss_fn(w)
        slope = float(np.dot(grad, direction))   # ∇f^T d  (should be > 0)

        alpha = self.alpha_init
        for _ in range(self.max_iter):
            w_new = w - alpha * direction
            if loss_fn(w_new) <= f0 - self.c_armijo * alpha * slope:
                return alpha
            alpha *= self.beta

        # Return the smallest tried step if no condition was met
        return alpha

    def __repr__(self) -> str:
        return (f"ArmijoLineSearch(alpha_init={self.alpha_init}, "
                f"beta={self.beta}, c_armijo={self.c_armijo})")
