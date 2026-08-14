"""Dependency-free binary classification metrics with tie handling."""

import numpy as np

from .utils import sigmoid


def logits_to_proba(z, loss_name=None):
    return sigmoid(np.asarray(z))


def auroc(y_true, scores):
    """AUROC via Mann-Whitney U with midrank tie handling, no sklearn."""
    y = np.asarray(y_true)
    s = np.asarray(scores)
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    start = 0
    while start < len(s):
        end = start + 1
        while end < len(s) and s[order[end]] == s[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end

    u = np.sum(ranks[y == 1]) - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def auprc(y_true, scores):
    """Tie-aware average precision (PR-AUC), matching sklearn conventions."""
    y = np.asarray(y_true)
    s = np.asarray(scores)
    n_pos = int(np.sum(y == 1))
    if n_pos == 0:
        return 0.0
    if np.sum(y == 0) == 0:
        return 1.0

    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    ss = s[order]
    # Index of the last element in each tied-score group.
    ends = np.r_[np.flatnonzero(ss[1:] != ss[:-1]) + 1, len(ss)]
    tp = np.cumsum(ys == 1)[ends - 1]
    fp = np.cumsum(ys == 0)[ends - 1]
    recall = tp / n_pos
    precision = tp / (tp + fp)
    # Each unique threshold contributes a rectangle (step function),
    # matching the code-v2 / sklearn average-precision convention.
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def compute_metrics(y_true, p_pred, threshold: float = 0.5):
    """Standard binary classification metrics from predicted probabilities."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p_pred)
    pred = (p >= threshold).astype(int)

    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall    = tp / (tp + fn) if tp + fn else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if precision + recall else 0.0)

    return {
        "accuracy":    (tp + tn) / len(y),
        "precision":   precision,
        "recall":      recall,
        "f1_minority": f1,
        "auroc":       auroc(y, p),
        "auprc":       auprc(y, p),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def cost_weighted_score(y_true, p_pred, cost_fn=5.0, cost_fp=1.0, threshold=0.5):
    """Total cost = cost_fn * FN + cost_fp * FP. Lower is better."""
    m = compute_metrics(y_true, p_pred, threshold)
    return float(cost_fn * m["fn"] + cost_fp * m["fp"])
