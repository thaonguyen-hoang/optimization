"""
nag.py — Nesterov Accelerated Gradient (NAG) descent.

Standard momentum update (Heavy Ball):
    v_{t+1} = μ * v_t - η * ∇f(w_t)
    w_{t+1} = w_t + v_{t+1}

Nesterov's correction: evaluate the gradient at the "lookahead" point:
    y_t      = w_t + μ * v_t              (lookahead point)
    v_{t+1}  = μ * v_t - η * ∇f(y_t)
    w_{t+1}  = w_t + v_{t+1}

Equivalently (Sutskever formulation, used here):
    w_{t+1} = w_t + μ * v_t - η * ∇f(w_t + μ * v_t)

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
    """
    Nesterov Accelerated Gradient descent.

    Parameters
    ----------
    step_size    : FixedLR | ArmijoLineSearch instance
    momentum     : momentum coefficient μ ∈ [0, 1)  (default 0.9)
    regularizer  : L1Regularizer | None (L2 handled in grad_fn)
    use_proximal : True when regularizer is L1
    """

    name = "nag"

    def __init__(
        self,
        step_size,
        momentum: float = 0.9,
        regularizer=None,
        use_proximal: bool = False,
    ):
        self.step_size = step_size
        self.momentum = momentum
        self.regularizer = regularizer
        self.use_proximal = use_proximal
        self._v = None   # velocity vector; initialized on first step

    def step(self, w, loss_fn, grad_fn, X, y, **kwargs) -> np.ndarray:
        if self._v is None:
            self._v = np.zeros_like(w)

        mu = self.momentum
        # Lookahead point
        y_look = w + mu * self._v

        grad = grad_fn(y_look, X, y)

        if hasattr(self.step_size, "search"):
            # Armijo at the lookahead point
            eta = self.step_size.search(
                y_look, grad,
                loss_fn=lambda _w: loss_fn(_w, X, y),
            )
        else:
            eta = self.step_size.step()

        w_half = y_look - eta * grad
        if self.use_proximal and self.regularizer is not None:
            w_new = self.regularizer.proximal(w_half, eta)
        else:
            w_new = w_half

        self._v = w_new - w
        return w_new

    def reset(self) -> None:
        self._v = None
        if hasattr(self.step_size, "reset"):
            self.step_size.reset()

    def __repr__(self) -> str:
        return (f"NAG(momentum={self.momentum}, "
                f"step_size={self.step_size}, proximal={self.use_proximal})")
