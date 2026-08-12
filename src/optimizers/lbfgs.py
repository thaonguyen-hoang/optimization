"""
lbfgs.py — L-BFGS optimizer via scipy.optimize.minimize.

This implementation wraps scipy's L-BFGS-B implementation, providing
a manual gradient function.  scipy handles its own internal Wolfe-condition
line search, so no external Armijo loop is needed or applied.

Experiment matrix note:
  - "Fixed step size" mode: Not applicable for L-BFGS in the traditional sense.
    scipy's L-BFGS-B always uses its internal line search.
    For the "fixed" column in the matrix, we run L-BFGS with maxiter=1 per
    outer epoch to simulate one quasi-Newton step per "epoch", which allows
    the epoch-level logging structure to remain consistent.
  - "Backtracking" mode: scipy's internal line search is Wolfe-conditions
    based, which is strictly stronger than Armijo.  We run with default
    scipy settings (full convergence within each call).

L1 is NOT supported (non-smooth; scipy L-BFGS-B supports bound constraints
but not non-differentiable penalties in this context).
"""

import numpy as np
from scipy.optimize import minimize
from src.optimizers.base import BaseOptimizer


class LBFGS(BaseOptimizer):
    """
    L-BFGS optimizer using scipy.optimize.minimize(method='L-BFGS-B').

    Parameters
    ----------
    maxiter    : maximum number of internal L-BFGS iterations per outer call
                 Use 1 for "one step per epoch" (fixed mode simulation),
                 or a large number for full convergence per outer epoch.
    m          : number of vector pairs to store (history size, default 10)
    gtol       : gradient tolerance for scipy's internal convergence
    """

    name = "lbfgs"

    def __init__(self, maxiter: int = 1, m: int = 10, gtol: float = 1e-5):
        self.maxiter = maxiter
        self.m = m
        self.gtol = gtol

    def step(self, w, loss_fn, grad_fn, X, y, **kwargs) -> np.ndarray:
        """
        Run scipy L-BFGS-B for `maxiter` iterations starting from w.

        Returns the updated parameter vector.
        """
        def objective(w_):
            return float(loss_fn(w_, X, y))

        def jac(w_):
            return grad_fn(w_, X, y).astype(np.float64)

        result = minimize(
            fun=objective,
            x0=w,
            jac=jac,
            method="L-BFGS-B",
            options={
                "maxiter": self.maxiter,
                "maxcor":  self.m,
                "gtol":    self.gtol,
                "ftol":    0.0,   # disable function tolerance to use maxiter
            },
        )
        return result.x

    def reset(self) -> None:
        pass   # No internal state to reset between runs

    def __repr__(self) -> str:
        return f"LBFGS(maxiter={self.maxiter}, m={self.m})"
