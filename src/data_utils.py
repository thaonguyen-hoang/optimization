"""Load fixed CSV splits and standardize using training statistics only."""

from pathlib import Path
import numpy as np
import pandas as pd

TARGET_COL = "Diabetes_binary"
TRAIN_CSV, VAL_CSV, TEST_CSV = "train.csv", "val.csv", "test.csv"


def load_data(path):
    frame = pd.read_csv(path)
    if frame.empty: raise ValueError(f"Empty data file: {path}")
    return frame


def _target(frame):
    return TARGET_COL if TARGET_COL in frame.columns else frame.columns[0]


def basic_clean(frame):
    target = _target(frame)
    return frame.drop_duplicates().dropna(subset=[target]).reset_index(drop=True)


def load_splits(data_dir, clean=True):
    root = Path(data_dir)
    frames = tuple(load_data(root / name) for name in (TRAIN_CSV, VAL_CSV, TEST_CSV))
    return tuple(basic_clean(f) for f in frames) if clean else frames


def standardize(train_df, val_df, test_df, cols):
    mean = train_df[cols].mean()
    std = train_df[cols].std().replace(0.0, 1.0).fillna(1.0)
    out = []
    for frame in (train_df, val_df, test_df):
        copy = frame.copy()
        copy[cols] = (copy[cols] - mean) / std
        out.append(copy)
    return *out, mean, std


def to_xy(frame, feature_cols=None):
    target = _target(frame)
    cols = list(feature_cols) if feature_cols is not None else [c for c in frame.columns if c != target]
    return frame[cols].to_numpy(dtype=np.float64), frame[target].to_numpy(dtype=np.float64)


def load_processed(data_dir, standardize=True, standardize_cols=None):
    if standardize_cols is not None: standardize = standardize_cols
    train, val, test = load_splits(data_dir)
    target = _target(train)
    features = [c for c in train.columns if c != target]
    for name, frame in (("validation", val), ("test", test)):
        if target not in frame or any(c not in frame for c in features):
            raise ValueError(f"{name} split has incompatible columns")
    if standardize:
        train, val, test, mean, std = globals()["standardize"](train, val, test, features)
    else:
        mean = std = None
    xt, yt = to_xy(train, features); xv, yv = to_xy(val, features); xs, ys = to_xy(test, features)
    n_pos, n_neg = int(np.sum(yt == 1)), int(np.sum(yt == 0))
    return {"X_train": xt, "y_train": yt, "X_val": xv, "y_val": yv,
            "X_test": xs, "y_test": ys, "feature_cols": features,
            "mean": mean, "std": std,
            "class_counts": {"n_pos": n_pos, "n_neg": n_neg,
                             "pos_ratio": n_pos / len(yt) if len(yt) else float("nan")}}
