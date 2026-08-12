"""
metrics.py — Classification metrics for model evaluation.

All functions accept raw probability scores (not hard labels) where
applicable, and binary labels y ∈ {0, 1}.
"""

import numpy as np


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute a standard suite of binary classification metrics.

    Returns: dict with keys: auroc, auprc, f1, precision, recall, accuracy
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

    return {
        "auroc":     auroc(y_true, y_prob),
        "auprc":     auprc(y_true, y_prob),
        "f1":        float(f1),
        "precision": float(precision),
        "recall":    float(recall),
        "accuracy":  float(accuracy),
    }


def auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """AUROC via Mann-Whitney U with midrank tie handling, no sklearn."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_prob)
    p_sorted = y_prob[order]
    unique_vals, idx, counts = np.unique(p_sorted, return_inverse=True, return_counts=True)
    midranks = np.cumsum(counts) - (counts - 1) / 2.0
    ranks = midranks[idx]

    orig_ranks = np.empty_like(ranks)
    orig_ranks[order] = ranks

    sum_ranks_pos = np.sum(orig_ranks[y_true == 1])
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Precision-recall AUC via trapezoidal rule over sorted thresholds."""
    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)
    n_pos = np.sum(y_true == 1)
    precision = tp_cum / (tp_cum + fp_cum)
    recall = tp_cum / n_pos if n_pos > 0 else np.zeros_like(tp_cum, dtype=float)
    recall = np.concatenate(([0.0], recall))
    precision = np.concatenate(([1.0], precision))
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(_trapz(precision, recall))

