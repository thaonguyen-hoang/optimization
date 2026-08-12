from .base import BaseOptimizer
from .gd import GradientDescent
from .sgd import SGD
from .nag import NAG
from .newton import Newton
from .lbfgs import LBFGS

OPTIMIZER_REGISTRY = {
    "gd": GradientDescent,
    "sgd": SGD,
    "nag": NAG,
    "newton": Newton,
    "lbfgs": LBFGS,
}

__all__ = ["BaseOptimizer", "GradientDescent", "SGD", "NAG", "Newton", "LBFGS", "OPTIMIZER_REGISTRY"]
