"""Fixed learning rates and loss-gradient Lipschitz bounds."""

import numpy as np


def lipschitz_constant(
    X,
    reg=None,
    lam: float = 0.0,
    loss_name: str = "bce",
    w_pos: float = 1.0,
    w_neg: float = 1.0,
    sigma_max: float | None = None,
    reg_type: str | None = None,
    lambda_reg: float | None = None,
) -> float:
    """
    Compute the Lipschitz constant L of the gradient.

    For sigmoid-based losses: L_base = σ_max(X)² / (4n)
    For squared hinge:        L_base = 2 * σ_max(X)² / n
    Weighted BCE:             L_base *= max(w_pos, w_neg)
    L2 regularization:        L += λ

    Parameters
    ----------
    X         : feature matrix (n_samples, n_features)
    lam       : regularization strength λ
    loss_name : 'bce' | 'weighted_bce' | 'squared_hinge'
    sigma_max : precomputed σ_max(X); pass to avoid repeated SVD
    reg_type  : 'none' | 'l2' | 'l1' (overrides reg.name if provided)
    lambda_reg: alternative to lam (takes precedence if not None)
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("X must be a non-empty 2-D array")

    if sigma_max is not None:
        sigma = float(sigma_max)
    else:
        sigma = float(np.linalg.svd(X, compute_uv=False)[0])

    base = sigma * sigma / X.shape[0]

    if loss_name.lower() == "squared_hinge":
        L = 2.0 * base
    else:
        L = 0.25 * base
        if loss_name.lower() == "weighted_bce":
            L *= max(w_pos, w_neg)

    kind = reg_type or getattr(reg, "name", reg if isinstance(reg, str) else "none")
    strength = lam if lambda_reg is None else lambda_reg

    if kind == "l2":
        L += strength if strength > 0 else getattr(reg, "lam", 0.0)

    return max(float(L), np.finfo(float).eps)


def build_step_size(step_type: str, **kwargs):
    """
    Return a float for fixed/diminishing steps (c/L), or None for backtracking.

    Backtracking is handled internally by each optimizer's step() method.
    """
    key = step_type.lower()
    if key in {"backtracking", "bt"}:
        return None
    if key in {"fixed", "diminishing"}:
        c = kwargs.get("c", 1.0)
        L = kwargs["L"]
        return float(c / L)
    raise ValueError(f"Unknown step-size type: {step_type!r}")
