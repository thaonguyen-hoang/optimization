"""
base.py — Abstract base class for all optimizers.

Every optimizer must implement `step()`, which performs a single
parameter update given the current parameters, loss function, and data.
"""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class BaseOptimizer(ABC):
    """
    Abstract base class for gradient-based optimizers.

    All optimizers share a common interface:
      - `step(w, loss_fn, grad_fn, X, y)` → updated w
      - `reset()` → reset any internal state (e.g. momentum)
    """

    name: str = "base"

    @abstractmethod
    def step(
        self,
        w: np.ndarray,
        loss_fn,
        grad_fn,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Perform one optimization step.

        Parameters
        ----------
        w        : current parameter vector, shape (d,)
        loss_fn  : callable (w, X, y) → scalar
        grad_fn  : callable (w, X, y) → ndarray of shape (d,)
        X        : feature matrix, shape (n, d)
        y        : label vector, shape (n,)

        Returns
        -------
        w_new : updated parameter vector, shape (d,)
        """

    def reset(self) -> None:
        """Reset internal state (called at the start of each run)."""
