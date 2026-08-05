"""
make_shared_splits.py
======================
RUN THIS ONCE, TOGETHER, AS A GROUP — before anyone starts their own
loss-function branch. It freezes train/val/test csvs to disk so all
three people load the exact same rows for the exact same features.

Usage:
    python make_shared_splits.py /path/to/merged_brfss.csv ./shared_splits/

After running, commit shared_splits/{train,val,test}.csv to your repo
(or shared drive) and have every person's notebook load from there
directly — do NOT re-run split_data() independently in each person's
notebook, since even the same seed can drift if pandas/numpy versions
differ across machines.
"""

import sys
import os
from data_utils import load_data, basic_clean, split_data, get_feature_columns, standardize


def main(raw_csv_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    df = load_data(raw_csv_path)
    df = basic_clean(df)

    train_df, val_df, test_df = split_data(df)

    feature_cols = get_feature_columns(df)
    # Decide as a group: standardize all features, or only continuous ones?
    # Default here: standardize everything.
    train_df, val_df, test_df, mean, std = standardize(
        train_df, val_df, test_df, feature_cols
    )

    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(out_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)
    mean.to_csv(os.path.join(out_dir, "feature_mean.csv"))
    std.to_csv(os.path.join(out_dir, "feature_std.csv"))

    print(f"Wrote train ({len(train_df)}), val ({len(val_df)}), "
          f"test ({len(test_df)}) to {out_dir}")
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python make_shared_splits.py <raw_csv_path> <out_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
