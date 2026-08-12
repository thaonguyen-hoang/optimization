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
    """Numerically stable sigmoid: σ(z) = 1 / (1 + exp(-z))."""
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z)),
    )


# ─── Mini-batch generator ────────────────────────────────────────────────────

def iter_batches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    rng: np.random.Generator | None = None,
) -> "Generator[tuple[np.ndarray, np.ndarray], None, None]":
    """
    Yield (X_batch, y_batch) mini-batches of size *batch_size*.

    If shuffle=True the data is permuted at the start of each call
    (i.e., once per epoch).
    """
    n = X.shape[0]
    if rng is None:
        rng = np.random.default_rng()
    indices = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        idx = indices[start: start + batch_size]
        yield X[idx], y[idx]
