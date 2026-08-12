from .l1 import L1Regularizer
from .l2 import L2Regularizer

REGISTRY = {
    "none": None,
    "l1": L1Regularizer,
    "l2": L2Regularizer,
}

__all__ = ["L1Regularizer", "L2Regularizer", "REGISTRY"]
