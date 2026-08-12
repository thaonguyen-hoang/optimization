"""
metrics.py — Classification metrics for model evaluation.

All functions accept raw probability scores (not hard labels) where
applicable, and binary labels y ∈ {0, 1}.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute a standard suite of binary classification metrics.

    Parameters
    ----------
    y_true    : ground-truth labels, shape (n,), values in {0, 1}
    y_prob    : predicted probabilities for class 1, shape (n,)
    threshold : decision threshold for converting probabilities to labels

    Returns
    -------
    dict with keys: auroc, auprc, f1, precision, recall, accuracy
    """
    y_pred = (y_prob >= threshold).astype(int)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (tp + tn) / len(y_true)

    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    return {
        "auroc":     float(auroc),
        "auprc":     float(auprc),
        "f1":        float(f1),
        "precision": float(precision),
        "recall":    float(recall),
        "accuracy":  float(accuracy),
    }


def auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Convenience wrapper returning only the AUROC scalar."""
    return float(roc_auc_score(y_true, y_prob))
