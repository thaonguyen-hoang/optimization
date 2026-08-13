import numpy as np


class FixedLR:
    """
    Fixed learning rate strategy with optional inverse-time decay.

    The learning rate is set to η = c / L where:
    c  : scaling constant from YAML config
    L  : Lipschitz constant of the gradient (computed from data)

    For SGD an optional multiplicative decay schedule is applied:
    η_t = η_0 / (1 + decay_rate * t)   where t is the global step count.
    """

    def __init__(self, c: float, L: float, decay_rate: float = 0.0):
        if L <= 0:
            raise ValueError(f"Lipschitz constant L must be > 0, got {L}")
        self.eta0 = c / L
        self.decay_rate = decay_rate
        self._step = 0

    @property
    def lr(self):
        """Current learning rate (accounts for decay)."""
        return self.eta0 / (1.0 + self.decay_rate * self._step)

    def step(self):
        """Advance the internal step counter."""
        self._step += 1

    def reset(self):
        self._step = 0

    def __repr__(self):
        return f"FixedLR(eta0={self.eta0:.6g}, decay_rate={self.decay_rate})"


class ArmijoLineSearch:
    """
    Armijo backtracking line search.

    Finds the largest step size α = alpha_init * beta^k satisfying the
    Armijo sufficient-decrease condition:

        f(w - α * d) ≤ f(w) - c_armijo * α * ∇f(w)^T d

    where d is the descent direction (typically d = ∇f for gradient descent).

    For gradient descent: d = grad, so ∇f^T d = ||grad||².

    Notes:
    - Newton / L-BFGS: pass direction via kwargs
    - SGD backtracking is theoretically unsound (stochastic gradient breaks Armijo guarantee)
    """

    def __init__(self, alpha_init: float = 1.0, beta: float = 0.5, c_armijo: float = 1e-4, max_iter: int = 50):
        self.alpha_init = alpha_init
        self.beta = beta
        self.c_armijo = c_armijo
        self.max_iter = max_iter

    def search(self, w, b, grad_w, grad_b, obj_fn, direction=None):
        """Run backtracking and return the accepted step size α.

        Parameters
        ----------
        w, b       : current parameters
        grad_w, grad_b : gradients
        obj_fn     : callable f(w, b) → scalar loss
        direction  : descent direction (joint w and b), default: joint grad

        Returns
        -------
        alpha : accepted step size
        """
        grad_joint = np.concatenate([grad_w, [grad_b]])
        if direction is None:
            direction = grad_joint

        f0 = obj_fn(w, b)
        slope = float(np.dot(grad_joint, direction))

        alpha = self.alpha_init
        for _ in range(self.max_iter):
            w_new = w - alpha * direction[:-1]
            b_new = b - alpha * direction[-1]
            if obj_fn(w_new, b_new) <= f0 - self.c_armijo * alpha * slope:
                return alpha
            alpha *= self.beta

        return alpha

    def reset(self):
        """Armijo search is stateless; nothing to reset."""

    def __repr__(self):
        return f"ArmijoLineSearch(alpha_init={self.alpha_init}, beta={self.beta}, c_armijo={self.c_armijo})"


def lipschitz_constant(
    X: np.ndarray,
    reg_type: str = "none",
    lambda_reg: float = 0.0,
    w_pos: float = 1.0,
    w_neg: float = 1.0,
    loss_name: str = "bce",
    sigma_max: float | None = None,
) -> float:
    """
    Compute the Lipschitz constant L of the gradient for logistic regression.

    For logistic-regression-style losses the gradient is L-Lipschitz with:
    
        L_base = λ_max(X^T X) / 4          (logistic / sigmoid-based losses)
        L_base = 2 * λ_max(X^T X) / n      (squared hinge: bounded 2nd deriv)
        L      = L_base + λ                 (add ridge strength when L2 reg is used)
        L      = L_base * max(w_pos, w_neg) (scale for weighted BCE)

    λ_max is computed via the largest singular value of X (σ_max²).

    Parameters
    ----------
    X          : feature matrix (n_samples, n_features)
    reg_type   : 'none' | 'l2' | 'l1'
    lambda_reg : regularization strength λ (only added to L when reg_type='l2')
    w_pos      : positive class weight (for weighted BCE; default 1.0)
    w_neg      : negative class weight (for weighted BCE; default 1.0)
    loss_name  : 'bce' | 'weighted_bce' | 'squared_hinge' | 'focal'
                 selects the curvature bound (sigmoid-based) / base factor.
    sigma_max  : precomputed σ_max(X) (largest singular value). Pass this to
                 avoid recomputing the SVD for every (λ, loss) combination —
                 σ_max depends only on X, which is fixed across a run.

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
    if sigma_max is None:
        sigma_max = np.linalg.svd(X, compute_uv=False)[0]
    if loss_name == "squared_hinge":
        # Squared hinge has per-sample second derivative ≤ 2, so the
        # gradient is 2·σ_max(X)²/n Lipschitz — twice the logistic bound.
        L = 2.0 * (sigma_max ** 2) / n
    else:
        # Sigmoid-based losses (bce, weighted_bce, focal): curvature ≤ 1/4.
        L = (sigma_max ** 2) / (4.0 * n)

    # Scale for weighted BCE: the dominant class weight scales the curvature
    weight_scale = max(w_pos, w_neg)
    L *= weight_scale

    # For L2 regularization the Hessian gains an extra λI, so L grows by λ
    if reg_type == "l2":
        L += lambda_reg

    return float(L)
