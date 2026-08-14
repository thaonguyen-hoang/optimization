"""Convex binary losses defined on raw logits."""

from abc import ABC, abstractmethod
import numpy as np

from .utils import sigmoid


class Loss(ABC):
    name = "loss"

    @abstractmethod
    def value(self, z: np.ndarray, y: np.ndarray) -> float: ...

    @abstractmethod
    def grad(self, z: np.ndarray, y: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def hessian_diag(self, z: np.ndarray, y: np.ndarray) -> np.ndarray: ...


class BCELoss(Loss):
    name = "bce"

    def value(self, z, y):
        return float(np.mean(np.logaddexp(0.0, z) - y * z))

    def grad(self, z, y):
        return sigmoid(z) - y

    def hessian_diag(self, z, y):
        p = sigmoid(z)
        return p * (1.0 - p)


class WeightedBCELoss(Loss):
    name = "weighted_bce"

    def __init__(self, w_pos: float = 1.0, w_neg: float = 1.0):
        if w_pos <= 0 or w_neg <= 0:
            raise ValueError("Class weights must be positive")
        self.w_pos, self.w_neg = float(w_pos), float(w_neg)

    def _weights(self, y):
        return np.where(y == 1, self.w_pos, self.w_neg)

    def value(self, z, y):
        return float(np.mean(self._weights(y) * (np.logaddexp(0.0, z) - y * z)))

    def grad(self, z, y):
        return self._weights(y) * (sigmoid(z) - y)

    def hessian_diag(self, z, y):
        p = sigmoid(z)
        return self._weights(y) * p * (1.0 - p)


class SquaredHingeLoss(Loss):
    """Squared hinge; its Hessian exists almost everywhere."""

    name = "squared_hinge"

    @staticmethod
    def _pm1(y):
        return 2.0 * y - 1.0

    def value(self, z, y):
        margin = np.maximum(0.0, 1.0 - self._pm1(y) * z)
        return float(np.mean(margin * margin))

    def grad(self, z, y):
        yp = self._pm1(y)
        return -2.0 * yp * np.maximum(0.0, 1.0 - yp * z)

    def hessian_diag(self, z, y):
        return 2.0 * (1.0 - self._pm1(y) * z > 0.0).astype(np.float64)


def build_loss(name: str, **kwargs) -> Loss:
    key = name.lower()
    if key == "bce":
        return BCELoss()
    if key == "weighted_bce":
        return WeightedBCELoss(kwargs.get("w_pos", 1.0), kwargs.get("w_neg", 1.0))
    if key == "squared_hinge":
        return SquaredHingeLoss()
    raise ValueError(f"Unknown loss: {name}")
