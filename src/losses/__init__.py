from .bce import BCELoss
from .weighted_bce import WeightedBCELoss
from .squared_hinge import SquaredHingeLoss
from .focal import FocalLoss

LOSS_REGISTRY = {
    "bce": BCELoss,
    "weighted_bce": WeightedBCELoss,
    "squared_hinge": SquaredHingeLoss,
    "focal": FocalLoss,
}

__all__ = ["BCELoss", "WeightedBCELoss", "SquaredHingeLoss", "FocalLoss", "LOSS_REGISTRY"]
