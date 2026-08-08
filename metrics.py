"""
metrics.py
==========
SHARED module. Since raw loss VALUES are not comparable across
different loss functions (BCE=0.3 and focal=0.3 don't mean the same
thing), every person reports the SAME task-metric set computed from
predicted probabilities, so the final 3-way table is apples-to-apples.

Everyone calls compute_metrics(y_true, p_pred) and reports the
returned dict as their "final setup" row.
"""

import numpy as np


def _sigmoid(z):
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def logits_to_proba(z, loss_name: str = None):
    """squared_hinge scores aren't probabilities; sigmoid them anyway
    for a comparable ranking-based score, but note this caveat in your
    write-up if you used squared hinge.
    """
    return _sigmoid(z)


def confusion_counts(y_true, y_pred_label):
    tp = int(np.sum((y_true == 1) & (y_pred_label == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred_label == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred_label == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred_label == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


def auroc(y_true, p_pred):
    """Rank-based AUROC via Mann-Whitney U with midrank tie handling, no sklearn dependency."""
    y_true = np.asarray(y_true)
    p_pred = np.asarray(p_pred)
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(p_pred)
    p_sorted = p_pred[order]
    unique_vals, idx, counts = np.unique(p_sorted, return_inverse=True, return_counts=True)
    midranks = np.cumsum(counts) - (counts - 1) / 2.0
    ranks = midranks[idx]

    orig_ranks = np.empty_like(ranks)
    orig_ranks[order] = ranks

    sum_ranks_pos = np.sum(orig_ranks[y_true == 1])
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def auprc(y_true, p_pred):
    """Precision-recall AUC via trapezoidal rule over sorted thresholds."""
    order = np.argsort(-p_pred)
    y_sorted = y_true[order]
    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)
    n_pos = np.sum(y_true == 1)
    precision = tp_cum / (tp_cum + fp_cum)
    recall = tp_cum / n_pos if n_pos > 0 else np.zeros_like(tp_cum, dtype=float)
    # prepend (recall=0, precision=1) for a clean starting point
    recall = np.concatenate(([0.0], recall))
    precision = np.concatenate(([1.0], precision))
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(_trapz(precision, recall))


def compute_metrics(y_true, p_pred, threshold: float = 0.5):
    y_true = np.asarray(y_true)
    p_pred = np.asarray(p_pred)
    y_pred_label = (p_pred >= threshold).astype(int)

    cc = confusion_counts(y_true, y_pred_label)
    precision, recall, f1 = precision_recall_f1(cc["tp"], cc["fp"], cc["fn"])
    acc = (cc["tp"] + cc["tn"]) / len(y_true)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_minority": f1,
        "auroc": auroc(y_true, p_pred),
        "auprc": auprc(y_true, p_pred),
        **cc,
    }


def cost_weighted_score(y_true, p_pred, cost_fn: float = 5.0, cost_fp: float = 1.0,
                         threshold: float = 0.5):
    """Optional: total cost = cost_fn * #FalseNegatives + cost_fp * #FalsePositives.
    Lower is better. Useful for your cost-sensitive framing/discussion.
    """
    y_pred_label = (np.asarray(p_pred) >= threshold).astype(int)
    cc = confusion_counts(np.asarray(y_true), y_pred_label)
    return cost_fn * cc["fn"] + cost_fp * cc["fp"]
