"""
gd.py — Gradient Descent (full-batch).

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
    """
    Full-batch Gradient Descent with optional proximal step for L1.

    Parameters
    ----------
    step_size   : FixedLR | ArmijoLineSearch instance
    regularizer : L1Regularizer | L2Regularizer | None
                  (L2 gradient is already added inside grad_fn by the model;
                   L1 proximal operator is applied here explicitly)
    use_proximal: if True, apply soft-thresholding after the gradient step
                  (must be True when regularizer is L1)
    """

    name = "gd"

    def __init__(self, step_size, regularizer=None, use_proximal: bool = False):
        self.step_size = step_size
        self.regularizer = regularizer
        self.use_proximal = use_proximal

    def step(self, w, loss_fn, grad_fn, X, y, **kwargs) -> np.ndarray:
        grad = grad_fn(w, X, y)

        if hasattr(self.step_size, "search"):
            # Armijo backtracking
            eta = self.step_size.search(
                w, grad,
                loss_fn=lambda _w: loss_fn(_w, X, y),
            )
        else:
            # Fixed LR (advances internal step counter)
            eta = self.step_size.step()

        w_new = w - eta * grad

        if self.use_proximal and self.regularizer is not None:
            w_new = self.regularizer.proximal(w_new, eta)

        return w_new

    def reset(self) -> None:
        if hasattr(self.step_size, "reset"):
            self.step_size.reset()

    def __repr__(self) -> str:
        return (f"GradientDescent(step_size={self.step_size}, "
                f"proximal={self.use_proximal})")
