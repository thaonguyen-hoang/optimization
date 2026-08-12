"""
Base class for optimizers.

Interface:
    opt.step(w, b, grad_w, grad_b, **kwargs) -> new_w, new_b

Optional attributes:
    - lookahead(w, b) -> (w_eval, b_eval): Nesterov lookahead point
    - requires_hessian: bool (Newton needs joint Hessian in kwargs)
    - requires_obj_fn: bool (backtracking needs callable objective)
    - requires_prox_fn: bool (backtracking needs proximal operator)
    - handles_prox: bool (optimizer manages prox internally)
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseOptimizer(ABC):
    """Abstract base class for optimizers."""

    name: str = "base"

    @abstractmethod
    def step(self, w, b, grad_w, grad_b, **kwargs):
        """Perform one optimization step.

        Parameters
        ----------
        w, b       : current parameters
        grad_w     : gradient w.r.t. w
        grad_b     : gradient w.r.t. b
        **kwargs   : optional (hess_joint, obj_fn, prox_fn)

        Returns
        -------
        new_w, new_b : updated parameters
        """

    def reset(self, w_shape):
        """Reset internal state at start of each run."""

    def lookahead(self, w, b):
        """Return evaluation point for gradient (e.g., Nesterov lookahead)."""
        return w, b

