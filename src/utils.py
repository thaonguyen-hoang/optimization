"""
utils.py — Data loading, seeding, class-imbalance ratio, logging helpers.
"""

import random
import logging
import numpy as np
import pandas as pd
from pathlib import Path


# ─── Logging ────────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
                                datefmt="%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ─── Reproducibility ────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set random seeds for NumPy and Python random for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)


# ─── Data loading ───────────────────────────────────────────────────────────

def load_split(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a CSV split into (X, y).

    Assumes the **first column** is the binary label (0/1) and all
    remaining columns are numeric features.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)  — values in {0, 1}
    """
    df = pd.read_csv(path)
    X = df.iloc[:, 1:].values.astype(np.float64)  # target is the first col
    y = df.iloc[:, 0].values.astype(np.float64)
    return X, y


def load_data(data_dir: str | Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Load train / val / test splits from *data_dir*.

    Returns
    -------
    dict with keys 'train', 'val', 'test', each mapping to (X, y).
    """
    data_dir = Path(data_dir)
    return {
        "train": load_split(data_dir / "train.csv"),
        "val":   load_split(data_dir / "val.csv"),
        "test":  load_split(data_dir / "test.csv"),
    }


# ─── Class imbalance ────────────────────────────────────────────────────────

def compute_imbalance_ratio(y: np.ndarray) -> float:
    """
    Compute R = N_neg / N_pos from the training labels.

    This ratio is used as the baseline positive class weight so that
    w_pos = c_scale * R balances the class frequencies.

    Raises ValueError if there are no positive or no negative samples.
    """
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    if n_pos == 0:
        raise ValueError("No positive samples found in y.")
    if n_neg == 0:
        raise ValueError("No negative samples found in y.")
    return float(n_neg) / float(n_pos)


# ─── Sigmoid ────────────────────────────────────────────────────────────────

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z)),
    )


# ─── Hessian assembly ────────────────────────────────────────────────────────

def hessian_joint(X: np.ndarray, diag: np.ndarray, reg_diag=None,
                  X_tilde: np.ndarray | None = None) -> np.ndarray:
    """
    Assemble the joint (d+1)×(d+1) Hessian H = X̃ᵀ·diag(diag)·X̃ / n,
    where X̃ = [X, 1] includes the bias column and *diag* is the per-sample
    second derivative of the loss (evaluated at the current logits).

    Parameters
    ----------
    X         : feature matrix (n_samples, d)
    diag      : per-sample loss Hessian diagonal, shape (n,)
    reg_diag  : optional regularizer Hessian contribution. If 1-D of length d
                (e.g. ``regularizer.hessian(w)``), it is placed on the
                trailing (weight) block of H[:-1, :-1]. If a (d×d) matrix, it
                is added directly.
    X_tilde   : precomputed design matrix [X, 1] of shape (n, d+1). X is fixed
                across a training run, so pass it to avoid rebuilding the
                stacked matrix (and the O(n) allocation) on every Newton step.

    Returns
    -------
    H : joint Hessian of shape (d+1, d+1)
    """
    n = X.shape[0]
    if X_tilde is None:
        X_tilde = np.c_[X, np.ones(n)]
    H = X_tilde.T @ (diag[:, None] * X_tilde) / n

    if reg_diag is not None:
        reg = np.diag(reg_diag) if np.ndim(reg_diag) == 1 else reg_diag
        H[:-1, :-1] += reg
    return H


# ─── JSON sanitizing ─────────────────────────────────────────────────────────

def sanitize_json(obj):
    """Recursively convert non-finite floats (NaN/inf) to None for JSON."""
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj

