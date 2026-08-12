"""
sgd.py — Stochastic Gradient Descent (mini-batch and single-sample).

One SGD "step" processes a single mini-batch (or one sample when batch_size=1).
One "epoch" = ceil(n / batch_size) such steps.

The model's train loop calls step() once per batch.  The caller (model.py)
handles the epoch-level loop and data shuffling (via iter_batches in utils).

Update rule per batch (X_b, y_b):
    g_b    = ∇f(w; X_b, y_b)    (stochastic gradient)
    w_{t+1} = w_t - η_t * g_b

Proximal variant for L1:
    w_half   = w_t - η_t * g_b
    w_{t+1}  = prox_{η_t λ}(w_half)

Backtracking + SGD (theoretically unsound):
    Armijo condition is evaluated on the CURRENT MINI-BATCH (not the full
    batch, which would be O(n²) per epoch and completely intractable for
    batch_size=1).  Even on the mini-batch the guarantee is broken: the
    stochastic gradient is a noisy direction that may not be a descent
    direction for either the mini-batch or the full-batch loss.  This run
    is kept to demonstrate and explain the failure mode.
"""

import numpy as np
from src.optimizers.base import BaseOptimizer


class SGD(BaseOptimizer):
    """
    Mini-batch (or single-sample) Stochastic Gradient Descent.

    Parameters
    ----------
    step_size    : FixedLR | ArmijoLineSearch instance
    regularizer  : L1Regularizer | L2Regularizer | None
    use_proximal : apply soft-thresholding for L1 after the gradient step
    """

    name = "sgd"

    def __init__(self, step_size, regularizer=None, use_proximal: bool = False):
        self.step_size = step_size
        self.regularizer = regularizer
        self.use_proximal = use_proximal

    def step(
        self,
        w: np.ndarray,
        loss_fn,
        grad_fn,
        X: np.ndarray,
        y: np.ndarray,
        X_full: np.ndarray | None = None,
        y_full: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        One mini-batch update.

        Parameters
        ----------
        X, y       : the current mini-batch
        X_full, y_full : full training data (used ONLY for Armijo evaluation
                         to demonstrate the theoretical inconsistency)
        """
        grad = grad_fn(w, X, y)

        if hasattr(self.step_size, "search"):
            # Armijo backtracking on the MINI-BATCH loss.
            #
            # Why not the full batch?  Evaluating the full-batch loss for every
            # single-sample step is O(n²) per epoch — completely intractable.
            #
            # Using the mini-batch loss is still theoretically unsound:
            # the stochastic gradient is a noisy estimate of the full-batch
            # gradient, so the Armijo sufficient-decrease condition has no
            # convergence guarantee even when evaluated consistently on the
            # same mini-batch.  This run exists to demonstrate the failure
            # mode, not to produce a useful optimizer.
            eta = self.step_size.search(
                w, grad,
                loss_fn=lambda _w: loss_fn(_w, X, y),   # mini-batch only
            )
        else:
            eta = self.step_size.step()

        w_new = w - eta * grad

        if self.use_proximal and self.regularizer is not None:
            w_new = self.regularizer.proximal(w_new, eta)

        return w_new

    def reset(self) -> None:
        if hasattr(self.step_size, "reset"):
            self.step_size.reset()

    def __repr__(self) -> str:
        return f"SGD(step_size={self.step_size}, proximal={self.use_proximal})"
