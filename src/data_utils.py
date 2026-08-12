"""
data_utils.py
=============
Shared data module. Loads the three pre-split BRFSS CSVs directly from
``data/`` — there is no re-splitting step. The split is fixed by the
files on disk so every run sees identical train/val/test rows.

Layout expected under ``data_dir``:
    diabetes_binary_health_indicators_BRFSS2015.csv   -> train
    diabetes_2021_val.csv                             -> val
    diabetes_2021_test.csv                            -> test

Standardization (mean/std) is fit on TRAIN ONLY and applied to val/test
to prevent leakage. The fitted mean/std are returned so callers can
persist them next to a run's artifacts.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Fixed, shared configuration.
# ---------------------------------------------------------------------
SEED = 42
TARGET_COL = "Diabetes_binary"

# Default file names inside data_dir.
TRAIN_CSV = "train.csv"
VAL_CSV = "val.csv"
TEST_CSV = "test.csv"


def load_data(path: str) -> pd.DataFrame:
    """Load one BRFSS csv. Expects TARGET_COL as 0/1 label."""
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Expected target column '{TARGET_COL}' not found. "
            f"Available columns: {list(df.columns)[:10]}..."
        )
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal shared cleaning: drop exact duplicate rows, drop rows with
    a missing label. Applied independently to each split so row counts
    stay reproducible regardless of caller order.
    """
    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET_COL])
    return df.reset_index(drop=True)


def load_splits(data_dir: str, clean: bool = True):
    """Load train/val/test dataframes directly from data_dir.

    Returns (train_df, val_df, test_df).
    """
    train_df = load_data(f"{data_dir}/{TRAIN_CSV}")
    val_df = load_data(f"{data_dir}/{VAL_CSV}")
    test_df = load_data(f"{data_dir}/{TEST_CSV}")
    if clean:
        train_df = basic_clean(train_df)
        val_df = basic_clean(val_df)
        test_df = basic_clean(test_df)
    return train_df, val_df, test_df


def get_feature_columns(df: pd.DataFrame):
    return [c for c in df.columns if c != TARGET_COL]


def standardize(train_df, val_df, test_df, cols):
    """Fit mean/std on TRAIN ONLY, apply to val/test. Prevents leakage."""
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


def load_processed(data_dir: str, standardize_cols: bool = True):
    """One-shot helper: load + clean + standardize + to_xy.

    Returns dict with X/y arrays for train/val/test plus feature_cols,
    mean, std, and class counts.
    """
    train_df, val_df, test_df = load_splits(data_dir)
    feature_cols = get_feature_columns(train_df)
    if standardize_cols:
        train_df, val_df, test_df, mean, std = standardize(
            train_df, val_df, test_df, feature_cols
        )
    else:
        mean, std = None, None

    X_train, y_train = to_xy(train_df, feature_cols)
    X_val, y_val = to_xy(val_df, feature_cols)
    X_test, y_test = to_xy(test_df, feature_cols)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "feature_cols": feature_cols,
        "mean": mean, "std": std,
        "class_counts": class_counts(y_train),
    }
