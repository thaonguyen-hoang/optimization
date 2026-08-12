"""
focal.py — Focal Loss for binary classification.

Proposed by Lin et al. (2017), "Focal Loss for Dense Object Detection".

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

where:
    p_t = p      if y = 1   (positive class probability)
    p_t = 1 - p  if y = 0   (use 1-p so "confidence" is always for true class)
    α_t = α      if y = 1
    α_t = 1 - α  if y = 0

    p = σ(z),   z = X @ w

Full loss (averaged over n samples):
    L(w) = -(1/n) Σ α_t_i * (1 - p_t_i)^γ * log(p_t_i + ε)

Gradient derivation:
    Let p = σ(z), then:

    For y=1:
        dFL/dz = α   * [(1-p)^γ * (-1/p) * p(1-p)     + γ*(1-p)^(γ-1)*(-1)*(1-p)*log(p)*... ]
               simplifies (via chain rule) to:
        dFL/dz = -α   * (1-p)^(γ-1) * [γ*p*log(p) + (1-p)] * (1-p)/p * p*(1-p) / ...

    Compact closed form (used in code):
        For y=1:   dFL/dz = α   * (1-p)^γ * [γ * p * log(p+ε) - (1-p)]  * (-1) * ... 

    We use the following clean and numerically stable formulation:

        w_t = α_t * (1 - p_t)^γ                          (modulating factor)
        For y=1:   dL/dp_t * dp_t/dz = -w_t/p_t * p(1-p)
                 + focal_correction_term

    Because the focal gradient is complex and error-prone, we implement it 
    using the explicit per-sample chain rule:

        dFL/dz_i = d/dz_i [ -α_t * (1-p_t)^γ * log(p_t) ]
    
    Let q = p_t (confidence), s = σ(z):
    
    Case y=1:  p_t = s,        α_t = α,     q = s
        dFL/dz = -α * [ γ*(1-q)^(γ-1)*(-s*(1-s))*log(q) + (1-q)^γ * (1/q)*s*(1-s) ]
               = -α * s*(1-s) * (1-q)^(γ-1) * [ -γ*log(q) + (1-q)/q ]  ... [*]

    Case y=0:  p_t = 1-s,      α_t = 1-α,   q = 1-s
        dFL/dz = -(1-α) * [ γ*(1-q)^(γ-1)*(s*(1-s))*log(q) + (1-q)^γ*(1/q)*(-s*(1-s)) ]
               = (1-α) * s*(1-s) * (1-q)^(γ-1) * [ γ*log(q) - (1-q)/q ]

    Vectorized form (both cases unified):
        sign_i  = +1 if y=1, -1 if y=0
        dFL/dz_i = sign_i * α_t_i * s_i*(1-s_i) * (1-p_t_i)^(γ-1) 
                   * [ -γ*log(p_t_i+ε) - (1-p_t_i)/p_t_i ] * ... 

    NOTE: Because focal loss has a complex gradient, this implementation
    provides a `gradient_check` flag recommendation.  Numerical gradient
    verification is strongly advised before running full experiments.

    Hessian:
    The exact focal Hessian is very complex. Newton's method is NOT run
    with focal loss in the experiment matrix. This method raises
    NotImplementedError if called.
"""

import numpy as np
from src.utils import sigmoid

_EPS = 1e-8   # numerical stability for log(0) avoidance


class FocalLoss:
    """
    Focal Loss for binary classification.

    Parameters
    ----------
    gamma : focusing parameter γ ≥ 0  (γ=0 reduces to weighted BCE)
    alpha : class balancing factor α ∈ (0, 1)  (weight for positive class)
    """

    name = "focal"

    def __init__(self, gamma: float = 2.0, alpha: float = 0.5):
        self.gamma = gamma
        self.alpha = alpha

    def _pt_and_alpha(self, s: np.ndarray, y: np.ndarray):
        """
        Compute per-sample:
          p_t  : confidence score for the true class
          a_t  : class-specific alpha weight
        """
        p_t = np.where(y == 1, s, 1.0 - s)
        a_t = np.where(y == 1, self.alpha, 1.0 - self.alpha)
        return p_t, a_t

    def __call__(self, w, b, X, y):
        return self.loss(w, b, X, y)

    def loss(self, w: np.ndarray, b: float, X: np.ndarray, y: np.ndarray) -> float:
        """Compute focal loss (scalar)."""
        z = X @ w + b
        s = sigmoid(z)
        p_t, a_t = self._pt_and_alpha(s, y)
        focal_weight = (1.0 - p_t) ** self.gamma
        return -float(np.mean(a_t * focal_weight * np.log(p_t + _EPS)))

    def gradient(self, w: np.ndarray, b: float, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Compute gradient ∇L(w), ∇L(b) using the per-sample chain rule.

        d(FL_i)/dz_i is derived from the chain rule:
            d(FL_i)/dz_i = d(FL_i)/dp_t * dp_t/ds * ds/dz
        """
        z = X @ w + b
        s = sigmoid(z)                          # σ(z), shape (n,)
        p_t, a_t = self._pt_and_alpha(s, y)
        g = self.gamma

        # Focusing weight and its derivative
        fw  = (1.0 - p_t) ** g                  # (1-p_t)^γ
        log_pt = np.log(p_t + _EPS)

        # Fix instability when g < 1 and p_t approaches 1.0
        one_minus_pt = np.clip(1.0 - p_t, _EPS, 1.0)
        
        if g > 0:
            dFL_dpt = a_t * (g * (one_minus_pt) ** (g - 1.0) * log_pt - fw / (p_t + _EPS))
        else:
            # γ=0: reduces to weighted BCE; first term vanishes
            dFL_dpt = -a_t * fw / (p_t + _EPS)

        # dp_t/ds:  +1 if y=1 (p_t=s),  -1 if y=0 (p_t=1-s)
        dpt_ds = np.where(y == 1, 1.0, -1.0)

        # ds/dz = s*(1-s)
        dFL_dz = dFL_dpt * dpt_ds * s * (1.0 - s)   # shape (n,)

        grad_w = X.T @ dFL_dz / X.shape[0]
        grad_b = float(np.sum(dFL_dz) / X.shape[0])
        return grad_w, grad_b

    def hessian(self, w, b, X, y):
        raise NotImplementedError(
            "Focal loss Hessian is not implemented. "
            "Newton's method is not used with focal loss."
        )

    def __repr__(self) -> str:
        return f"FocalLoss(gamma={self.gamma}, alpha={self.alpha})"
