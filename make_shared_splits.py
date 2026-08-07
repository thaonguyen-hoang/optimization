"""
make_shared_splits.py
======================
RUN THIS ONCE, TOGETHER, AS A GROUP — before anyone starts their own
loss-function branch. It freezes train/val/test csvs to disk so all
three people load the exact same rows for the exact same features.

Split scheme:
    TRAIN = first 3 survey years, merged into one file
    VAL   = last survey year, first half (already split, given as-is)
    TEST  = last survey year, second half (already split, given as-is)

Only CONTINUOUS_COLS (see data_utils.py) are standardized; binary and
ordinal survey columns are left on their original scale.

Usage:
    python make_shared_splits.py \
        --train_years year1.csv year2.csv year3.csv \
        --val val.csv \
        --test test.csv \
        --out_dir ./shared_splits/
"""

import argparse
import os

from data_utils import (
    load_and_merge_train_years,
    load_val_test,
    get_feature_columns,
    standardize,
    CONTINUOUS_COLS,
)


def main(train_year_paths, val_path, test_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    train_df = load_and_merge_train_years(train_year_paths)
    val_df, test_df = load_val_test(val_path, test_path)

    feature_cols = get_feature_columns(train_df)

    # sanity check: val/test must have the same schema as train
    for name, df in [("val", val_df), ("test", test_df)]:
        missing = set(train_df.columns) - set(df.columns)
        extra = set(df.columns) - set(train_df.columns)
        if missing or extra:
            raise ValueError(
                f"Schema mismatch between train and {name}: "
                f"missing {missing}, extra {extra}."
            )

    train_df, val_df, test_df, mean, std = standardize(
        train_df, val_df, test_df, cols=CONTINUOUS_COLS
    )

    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(out_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)
    mean.to_csv(os.path.join(out_dir, "continuous_feature_mean.csv"))
    std.to_csv(os.path.join(out_dir, "continuous_feature_std.csv"))

    print(f"Wrote train ({len(train_df)}), val ({len(val_df)}), "
          f"test ({len(test_df)}) to {out_dir}")
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
    print(f"Standardized (continuous only): {CONTINUOUS_COLS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_years", nargs=3, required=True,
                         metavar=("YEAR1_CSV", "YEAR2_CSV", "YEAR3_CSV"),
                         help="paths to the 3 first-year csv files, merged into train")
    parser.add_argument("--val", required=True, help="path to the val csv (last year, first half)")
    parser.add_argument("--test", required=True, help="path to the test csv (last year, second half)")
    parser.add_argument("--out_dir", default="./shared_splits/")
    args = parser.parse_args()

    main(args.train_years, args.val, args.test, args.out_dir)
