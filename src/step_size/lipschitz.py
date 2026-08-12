"""
lipschitz.py — Compute the Lipschitz constant L of the gradient.

For logistic-regression-style losses the gradient is L-Lipschitz with:

    L_base = λ_max(X^T X) / 4          (logistic / sigmoid-based losses)
    L      = L_base + λ                 (add ridge strength when L2 reg is used)
    L      = L_base * max(w_pos, w_neg) (scale for weighted BCE)

λ_max is computed via the largest singular value of X (σ_max²).
"""

import numpy as np


def lipschitz_constant(
    X: np.ndarray,
    reg_type: str = "none",
    lambda_reg: float = 0.0,
    w_pos: float = 1.0,
    w_neg: float = 1.0,
) -> float:
    """
    Compute the Lipschitz constant L of the gradient for logistic regression.

    Parameters
    ----------
    X          : feature matrix (n_samples, n_features)
    reg_type   : 'none' | 'l2' | 'l1'
    lambda_reg : regularization strength λ (only added to L when reg_type='l2')
    w_pos      : positive class weight (for weighted BCE; default 1.0)
    w_neg      : negative class weight (for weighted BCE; default 1.0)

    Returns
    -------
    L : float — Lipschitz constant

    Notes
    -----
    We compute σ_max(X)² = λ_max(X^T X) using np.linalg.svd with
    compute_uv=False, which is more numerically stable than forming X^T X
    explicitly for large matrices.
    """
    # λ_max(X^T X) = σ_max(X)²
    # The loss is averaged over n samples: f(w) = (1/n) Σ f_i(w)
    # so H = (1/n) X^T D X,  λ_max(H) ≤ σ_max(X)² / (4n)
    n = X.shape[0]
    sigma_max = np.linalg.svd(X, compute_uv=False)[0]
    L = (sigma_max ** 2) / (4.0 * n)

    # Scale for weighted BCE: the dominant class weight scales the curvature
    weight_scale = max(w_pos, w_neg)
    L *= weight_scale

    # For L2 regularization the Hessian gains an extra λI, so L grows by λ
    if reg_type == "l2":
        L += lambda_reg

    return float(L)
