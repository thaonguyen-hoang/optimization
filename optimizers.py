"""
optimizers.py
=============
SHARED module. All optimizers expose the same interface so train.py
can swap them freely:

    opt.step(w, b, grad_w, grad_b) -> new_w, new_b

Each optimizer owns its own internal state (momentum buffers, Adam
moments, etc.), created fresh via reset() at the start of every run.
Everyone uses the SAME optimizer classes and the SAME LR search grid
(see LR_GRID below) so "best optimizer for loss X" is comparable
across all three people's branches.
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

    def step(self, w, b, grad_w, grad_b):
        return w - self.lr * grad_w, b - self.lr * grad_b


class SGD:
    """Mini-batch SGD, no momentum. Same update rule as GD; the
    'stochastic' part comes from train.py feeding mini-batches instead
    of the full dataset each step.
    """
    name = "sgd"

    def __init__(self, lr: float = 1e-2):
        self.lr = lr

    def reset(self, w_shape):
        pass

    def step(self, w, b, grad_w, grad_b):
        return w - self.lr * grad_w, b - self.lr * grad_b


class SGDMomentum:
    """SGD with (classical or Nesterov) momentum."""
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

    def step(self, w, b, grad_w, grad_b):
        self.v_w = self.momentum * self.v_w - self.lr * grad_w
        self.v_b = self.momentum * self.v_b - self.lr * grad_b
        if self.nesterov:
            # look-ahead correction
            new_w = w + self.momentum * self.v_w - self.lr * grad_w
            new_b = b + self.momentum * self.v_b - self.lr * grad_b
        else:
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

    def step(self, w, b, grad_w, grad_b):
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


OPTIMIZER_REGISTRY = {
    "gd": GD,
    "sgd": SGD,
    "sgd_momentum": SGDMomentum,
    "adam": Adam,
}
