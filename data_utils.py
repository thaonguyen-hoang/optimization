"""
data_utils.py
=============
SHARED module — do not fork/modify per-person. All three people import
this as-is so the train/val/test data and preprocessing are IDENTICAL
across everyone's experiments. This is what makes the final 3-way
comparison valid.

If you change anything here, notify the group — it invalidates
everyone's already-run results.

--- Split scheme (decided by the group) ---
- TRAIN = first 3 survey years, merged into one file.
- VAL / TEST = last survey year, already split in half into two
  separate files (not a random split we do here — those files are
  taken as given).
This is a time-based split, not a random stratified split: it lets you
discuss distribution shift across years as part of your analysis
(does the model trained on years 1-3 generalize to the held-out year?).
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Fixed, shared configuration. Everyone imports SEED from here — never
# hardcode your own seed in your personal notebook.
# ---------------------------------------------------------------------
SEED = 42

# Column name of the binary target in the BRFSS csvs.
TARGET_COL = "Diabetes_binary"

# Continuous features to standardize. Everything else (binary 0/1 and
# ordinal survey-scale columns) is left as-is. Adjust this list once,
# as a group, to match your actual merged schema before running
# make_shared_splits.py.
CONTINUOUS_COLS = ["BMI", "MentHlth", "PhysHlth"]


def load_data(path: str) -> pd.DataFrame:
    """Load a single BRFSS year csv."""
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Expected target column '{TARGET_COL}' not found in {path}. "
            f"Available columns: {list(df.columns)[:10]}..."
        )
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal shared cleaning: drop exact duplicate rows, drop rows with a missing label."""
    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET_COL])
    return df.reset_index(drop=True)


def load_and_merge_train_years(paths: list) -> pd.DataFrame:
    """
    Load the 3 first-year csv files and concatenate them into one
    training set. Cleans (dedup, drop missing-label rows) AFTER
    merging, so duplicates that happen to span two different year
    files are also caught.

    paths: list of 3 file paths, e.g.
        ["brfss_2015.csv", "brfss_2017.csv", "brfss_2019.csv"]
    """
    dfs = [load_data(p) for p in paths]

    # sanity check: all 3 years must have the exact same schema before
    # concatenating, otherwise columns silently misalign / fill with NaN
    ref_cols = set(dfs[0].columns)
    for p, df in zip(paths[1:], dfs[1:]):
        if set(df.columns) != ref_cols:
            missing = ref_cols - set(df.columns)
            extra = set(df.columns) - ref_cols
            raise ValueError(
                f"Schema mismatch in {p}: missing cols {missing}, "
                f"extra cols {extra}. Align columns across years first."
            )

    merged = pd.concat(dfs, ignore_index=True)
    merged = basic_clean(merged)
    return merged


def load_val_test(val_path: str, test_path: str):
    """
    Load the already-split val/test files (last survey year, given
    as two separate files). Only basic_clean is applied — no further
    splitting happens here.
    """
    val_df = basic_clean(load_data(val_path))
    test_df = basic_clean(load_data(test_path))
    return val_df, test_df


def split_by_ratio(df: pd.DataFrame, ratio: float = 0.5, seed: int = SEED):
    """
    OPTIONAL utility — Stratified split based on a given ratio, fixed seed. 
    `ratio` determines the proportion of the dataset that goes into the first 
    returned DataFrame (e.g., 0.8 returns an 80/20 split).
    """
    rng = np.random.RandomState(seed)
    y = df[TARGET_COL].values

    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)

    def _split(idx):
        # Calculate the split point based on the provided ratio
        n_split = int(len(idx) * ratio)
        return idx[:n_split], idx[n_split:]

    va_pos, te_pos = _split(idx_pos)
    va_neg, te_neg = _split(idx_neg)

    val_idx = np.concatenate([va_pos, va_neg])
    test_idx = np.concatenate([te_pos, te_neg])
    
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return (
        df.iloc[val_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


def get_feature_columns(df: pd.DataFrame):
    return [c for c in df.columns if c != TARGET_COL]


def standardize(train_df, val_df, test_df, cols=None):
    """
    Fit mean/std on TRAIN ONLY, apply to val/test. Prevents leakage.

    cols defaults to CONTINUOUS_COLS — only continuous features
    (BMI, MentHlth, PhysHlth, ...) get standardized. Binary 0/1 and
    ordinal survey-scale columns are intentionally left untouched, so
    their raw 0/1 / category-level meaning stays interpretable (and
    L1's sparsity story stays meaningful on the original scale).
    """
    if cols is None:
        cols = CONTINUOUS_COLS
    missing = [c for c in cols if c not in train_df.columns]
    if missing:
        raise ValueError(
            f"CONTINUOUS_COLS not found in data: {missing}. "
            f"Update CONTINUOUS_COLS in data_utils.py to match your schema."
        )

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
