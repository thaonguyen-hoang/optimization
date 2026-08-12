# src/losses/__init__.py
from .bce import BCELoss
from .weighted_bce import WeightedBCELoss
from .squared_hinge import SquaredHingeLoss
from .focal import FocalLoss

__all__ = ["BCELoss", "WeightedBCELoss", "SquaredHingeLoss", "FocalLoss"]
