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
    """Average precision (PR-AUC), tie-aware, matching sklearn conventions.

    ``AP = Σ_k (R_k − R_{k−1}) · P_k`` over unique score thresholds from
    highest to lowest, where ties are scored as a single threshold (the whole
    tied group is included before precision/recall are measured).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n_pos = np.sum(y_true == 1)
    if n_pos == 0:
        return 0.0
    if np.sum(y_true == 0) == 0:
        return 1.0

    order = np.argsort(-y_prob, kind="stable")
    y_sort = y_true[order].astype(float)
    cum_tp = np.cumsum(y_sort)

    # Last index of each tie group (score changes) in the sorted array.
    neg_p = -y_prob[order]
    first_idx = np.unique(neg_p, return_index=True)[1]
    group_ends = np.concatenate([first_idx[1:], [len(y_sort)]]) - 1

    precision = cum_tp[group_ends] / (group_ends + 1)
    recall = cum_tp[group_ends] / n_pos
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))

