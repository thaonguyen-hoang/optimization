"""
merge_results.py
=================
RUN THIS ONCE, TOGETHER, at the end -- after all three
{person}_final_row.csv files exist. Produces the single comparison
table for your report/slides (the "best setup" table your teacher
asked for).

Usage:
    python merge_results.py person1_final_row.csv person2_final_row.csv person3_final_row.csv
"""

import sys
import pandas as pd


def main(paths):
    rows = [pd.read_csv(p) for p in paths]
    combined = pd.concat(rows, ignore_index=True)

    cols_order = ["person", "loss", "loss_hparams", "optimizer", "lr",
                  "regularizer", "lambda", "accuracy", "precision",
                  "recall", "f1_minority", "auroc", "auprc",
                  "tp", "tn", "fp", "fn"]
    cols_order = [c for c in cols_order if c in combined.columns]
    combined = combined[cols_order]

    combined.to_csv("final_comparison_table.csv", index=False)
    print(combined.to_string(index=False))
    print("\nWrote final_comparison_table.csv")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python merge_results.py <row1.csv> <row2.csv> <row3.csv> ...")
        sys.exit(1)
    main(sys.argv[1:])
