"""
optimizers.py
=============
SHARED module. All optimizers expose the same interface so train.py
can swap them freely:

    opt.step(w, b, grad_w, grad_b, **kwargs) -> new_w, new_b

Capabilities / Protocols supported via attributes or methods:
  - lookahead(w, b) -> (w_eval, b_eval): returns point for gradient evaluation (e.g. NAG lookahead).
  - requires_hessian: requires hess_joint in kwargs (Newton).
  - requires_obj_fn / requires_prox_fn: requires objective/prox functions in kwargs (GDBacktracking).
  - handles_prox: optimizer manages proximal steps internally (GDBacktracking).

Each optimizer owns its own internal state, created fresh via reset() at start of every run.
"""

import numpy as np

# Shared LR search grid — sweep this per optimizer, don't invent your own.
LR_GRID = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]


class GD:
    """Full-batch gradient descent."""
    name = "gd"

    def __init__(self, lr: float = 1e-2):
        self.lr = lr

    def reset(self, w_shape):
        pass

    def lookahead(self, w, b):
        return w, b

    def step(self, w, b, grad_w, grad_b, **kwargs):
        return w - self.lr * grad_w, b - self.lr * grad_b


class SGD:
    """Mini-batch SGD, no momentum."""
    name = "sgd"

    def __init__(self, lr: float = 1e-2):
        self.lr = lr

    def reset(self, w_shape):
        pass

    def lookahead(self, w, b):
        return w, b

    def step(self, w, b, grad_w, grad_b, **kwargs):
        return w - self.lr * grad_w, b - self.lr * grad_b


class SGDMomentum:
    """SGD with (classical or true Nesterov accelerated) momentum."""
    name = "sgd_momentum"

    def __init__(self, lr: float = 1e-2, momentum: float = 0.9, nesterov: bool = False):
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.v_w = None
        self.v_b = 0.0

    def reset(self, w_shape):
        self.v_w = np.zeros(w_shape)
        self.v_b = 0.0

    def lookahead(self, w, b):
        if self.nesterov and self.v_w is not None:
            return w + self.momentum * self.v_w, b + self.momentum * self.v_b
        return w, b

    def step(self, w, b, grad_w, grad_b, **kwargs):
        self.v_w = self.momentum * self.v_w - self.lr * grad_w
        self.v_b = self.momentum * self.v_b - self.lr * grad_b
        new_w = w + self.v_w
        new_b = b + self.v_b
        return new_w, new_b


class Adam:
    """Adam optimizer, standard bias-corrected moment estimates."""
    name = "adam"

    def __init__(self, lr: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m_w = None
        self.v_w = None
        self.m_b = 0.0
        self.v_b = 0.0
        self.t = 0

    def reset(self, w_shape):
        self.m_w = np.zeros(w_shape)
        self.v_w = np.zeros(w_shape)
        self.m_b = 0.0
        self.v_b = 0.0
        self.t = 0

    def lookahead(self, w, b):
        return w, b

    def step(self, w, b, grad_w, grad_b, **kwargs):
        self.t += 1
        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * grad_w
        self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (grad_w ** 2)
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * grad_b
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (grad_b ** 2)

        m_w_hat = self.m_w / (1 - self.beta1 ** self.t)
        v_w_hat = self.v_w / (1 - self.beta2 ** self.t)
        m_b_hat = self.m_b / (1 - self.beta1 ** self.t)
        v_b_hat = self.v_b / (1 - self.beta2 ** self.t)

        new_w = w - self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.eps)
        new_b = b - self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.eps)
        return new_w, new_b


class Newton:
    """Newton's method using exact Hessian for convex smooth losses.
    Note: for L1 regularization, soft-thresholding prox is applied post-step.
    """
    name = "newton"
    requires_hessian = True

    def __init__(self, lr: float = 1.0):
        self.lr = lr

    def reset(self, w_shape):
        pass

    def lookahead(self, w, b):
        return w, b

    def step(self, w, b, grad_w, grad_b, hess_joint, **kwargs):
        grad_joint = np.concatenate([grad_w, [grad_b]])
        diag_max = float(np.max(np.diag(hess_joint))) if len(hess_joint) > 0 else 1.0
        ridge = max(1e-8, 1e-6 * max(diag_max, 1.0))
        H_reg = hess_joint + ridge * np.eye(len(grad_joint))
        
        try:
            step_joint = np.linalg.solve(H_reg, grad_joint)
        except np.linalg.LinAlgError:
            step_joint = np.linalg.lstsq(H_reg, grad_joint, rcond=None)[0]
        
        new_w = w - self.lr * step_joint[:-1]
        new_b = b - self.lr * step_joint[-1]
        return new_w, new_b


class GDBacktracking:
    """Proximal Gradient Descent with Backtracking Line Search (Armijo)."""
    name = "gd_backtracking"
    requires_obj_fn = True
    requires_prox_fn = True
    handles_prox = True

    def __init__(self, init_lr: float = 1.0, alpha: float = 0.5, beta: float = 0.5):
        self.init_lr = init_lr
        self.alpha = alpha  # sufficient decrease condition
        self.beta = beta    # step reduction factor
        self.lr = init_lr

    def reset(self, w_shape):
        self.lr = self.init_lr

    def lookahead(self, w, b):
        return w, b

    def step(self, w, b, grad_w, grad_b, obj_fn, prox_fn=None, **kwargs):
        t = getattr(self, 'lr', self.init_lr)
        current_obj = obj_fn(w, b)
        
        for _ in range(40):
            new_w = w - t * grad_w
            if prox_fn is not None:
                new_w = prox_fn(new_w, t)
            new_b = b - t * grad_b
            
            new_obj = obj_fn(new_w, new_b)
            
            # Generalized sufficient decrease for proximal gradient
            dist_sq = np.sum((w - new_w)**2) + (b - new_b)**2
            if new_obj <= current_obj - (self.alpha / t) * dist_sq:
                break
            t *= self.beta
            if t < 1e-12:
                break
                
        self.lr = t 
        return new_w, new_b


OPTIMIZER_REGISTRY = {
    "gd": GD,
    "sgd": SGD,
    "sgd_momentum": SGDMomentum,
    "adam": Adam,
    "newton": Newton,
    "gd_backtracking": GDBacktracking,
}
