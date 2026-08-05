"""
data_utils.py
=============
SHARED module — do not fork/modify per-person. All three people import
this as-is so the train/val/test split and preprocessing are IDENTICAL
across everyone's experiments. This is what makes the final 3-way
comparison valid.

If you change anything here, notify the group — it invalidates
everyone's already-run results.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Fixed, shared configuration. Everyone imports SEED from here — never
# hardcode your own seed in your personal notebook.
# ---------------------------------------------------------------------
SEED = 42
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15  # remaining ~0.70 goes to train

# Column name of the binary target in your merged BRFSS csv.
# Update once here if your column is named differently.
TARGET_COL = "Diabetes_binary"


def load_data(path: str) -> pd.DataFrame:
    """Load the merged multi-year BRFSS csv.

    Expects one row per respondent, TARGET_COL as 0/1 label, and the
    rest numeric/binary/ordinal survey features (already merged across
    years before this step — schema alignment across years is a
    one-time group task, not per-person).
    """
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Expected target column '{TARGET_COL}' not found. "
            f"Available columns: {list(df.columns)[:10]}..."
        )
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal shared cleaning: drop exact duplicate rows, drop rows
    with a missing label. Do NOT add per-person cleaning steps here —
    if you find you need extra cleaning, propose it to the group so
    everyone's data stays identical.
    """
    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET_COL])
    return df.reset_index(drop=True)


def split_data(df: pd.DataFrame, seed: int = SEED):
    """Stratified train/val/test split, fixed seed, shared by everyone.

    Returns (train_df, val_df, test_df) — all three people must load
    these three (or read them from the shared cached csvs, see
    make_shared_splits.py below) rather than re-splitting themselves.
    """
    rng = np.random.RandomState(seed)
    y = df[TARGET_COL].values

    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)

    def _split_indices(idx):
        n = len(idx)
        n_test = int(n * TEST_FRACTION)
        n_val = int(n * VAL_FRACTION)
        test_idx = idx[:n_test]
        val_idx = idx[n_test:n_test + n_val]
        train_idx = idx[n_test + n_val:]
        return train_idx, val_idx, test_idx

    tr_pos, va_pos, te_pos = _split_indices(idx_pos)
    tr_neg, va_neg, te_neg = _split_indices(idx_neg)

    train_idx = np.concatenate([tr_pos, tr_neg])
    val_idx = np.concatenate([va_pos, va_neg])
    test_idx = np.concatenate([te_pos, te_neg])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


def get_feature_columns(df: pd.DataFrame):
    return [c for c in df.columns if c != TARGET_COL]


def standardize(train_df, val_df, test_df, cols):
    """Fit mean/std on TRAIN ONLY, apply to val/test. Prevents leakage.
    Everyone must standardize the same way — continuous survey columns
    (BMI, MentHlth, PhysHlth, etc.) matter most here; binary 0/1
    columns are harmless to standardize too but you may choose to
    leave them untouched — agree on this as a group once.
    """
    mean = train_df[cols].mean()
    std = train_df[cols].std().replace(0, 1.0)

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df[cols] = (train_df[cols] - mean) / std
    val_df[cols] = (val_df[cols] - mean) / std
    test_df[cols] = (test_df[cols] - mean) / std
    return train_df, val_df, test_df, mean, std


def to_xy(df: pd.DataFrame, feature_cols):
    X = df[feature_cols].values.astype(np.float64)
    y = df[TARGET_COL].values.astype(np.float64)
    return X, y


def class_counts(y: np.ndarray):
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return {"n_pos": n_pos, "n_neg": n_neg, "pos_ratio": n_pos / len(y)}
