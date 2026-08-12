"""
squared_hinge.py — Squared Hinge loss for binary classification.

Logit-based interface: z = X @ w + b (margin), y in {0, 1} (mapped to {-1, +1}).
Convex AND smooth (unlike plain hinge) in z.
"""

import numpy as np


class SquaredHingeLoss:
    """Squared hinge, margin form: max(0, 1 - y'*z)^2 where y' in {-1,+1}."""

    name = "squared_hinge"

    @staticmethod
    def _to_pm1(y):
        return 2 * y - 1

    def value(self, z, y):
        y_pm = self._to_pm1(y)
        margin = np.maximum(0.0, 1 - y_pm * z)
        return float(np.mean(margin ** 2))

    def grad(self, z, y):
        y_pm = self._to_pm1(y)
        margin = np.maximum(0.0, 1 - y_pm * z)
        return -2 * y_pm * margin

    def hessian(self, z, y):
        y_pm = self._to_pm1(y)
        margin = np.maximum(0.0, 1 - y_pm * z)
        return 2.0 * (margin > 0).astype(float)

    def __repr__(self):
        return "SquaredHingeLoss()"
