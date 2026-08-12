"""
L-BFGS optimizer (native two-loop recursion implementation).

Preserves history vectors (s and y) across epoch steps.
Implements Nocedal Algorithm 7.4 for the two-loop recursion.
"""

import numpy as np
from src.optimizers.base import BaseOptimizer


class LBFGS(BaseOptimizer):
    """L-BFGS optimizer (native implementation)."""

    name = "lbfgs"

    def __init__(self, step_size, m: int = 10, **kwargs):
        self.step_size = step_size
        self.m = m
        self._s_hist = []
        self._y_hist = []
        self._rho_hist = []
        self._prev_w = None
        self._prev_b = None
        self._prev_grad_w = None
        self._prev_grad_b = None

    def step(self, w, b, grad_w, grad_b, **kwargs):
        theta = np.concatenate([w, [b]])
        grad_joint = np.concatenate([grad_w, [grad_b]])

        if self._prev_w is not None:
            prev_theta = np.concatenate([self._prev_w, [self._prev_b]])
            prev_grad_joint = np.concatenate([self._prev_grad_w, [self._prev_grad_b]])

            s_k = theta - prev_theta
            y_k = grad_joint - prev_grad_joint

            sy = np.dot(s_k, y_k)
            if sy > 1e-10:
                self._s_hist.append(s_k)
                self._y_hist.append(y_k)
                self._rho_hist.append(1.0 / sy)
                if len(self._s_hist) > self.m:
                    self._s_hist.pop(0)
                    self._y_hist.pop(0)
                    self._rho_hist.pop(0)

        q = grad_joint.copy()
        alphas = []
        for i in reversed(range(len(self._s_hist))):
            alpha = self._rho_hist[i] * np.dot(self._s_hist[i], q)
            alphas.append(alpha)
            q -= alpha * self._y_hist[i]

        if len(self._s_hist) > 0:
            gamma = np.dot(self._s_hist[-1], self._y_hist[-1]) / np.dot(self._y_hist[-1], self._y_hist[-1])
            z = gamma * q
        else:
            z = q

        alphas.reverse()
        for i in range(len(self._s_hist)):
            beta = self._rho_hist[i] * np.dot(self._y_hist[i], z)
            z += self._s_hist[i] * (alphas[i] - beta)

        direction_joint = z

        if hasattr(self.step_size, "search"):
            obj_fn = kwargs.get("obj_fn")
            eta = self.step_size.search(w, b, grad_w, grad_b, obj_fn, direction=direction_joint)
        else:
            eta = self.step_size.lr
            self.step_size.step()

        w_new = w - eta * direction_joint[:-1]
        b_new = b - eta * direction_joint[-1]

        self._prev_w = w.copy()
        self._prev_b = b
        self._prev_grad_w = grad_w.copy()
        self._prev_grad_b = grad_b

        return w_new, b_new

    def reset(self, w_shape):
        self._s_hist = []
        self._y_hist = []
        self._rho_hist = []
        self._prev_w = None
        self._prev_b = None
        self._prev_grad_w = None
        self._prev_grad_b = None
        self.step_size.reset()

    def __repr__(self):
        return f"LBFGS(m={self.m}, step_size={self.step_size})"

