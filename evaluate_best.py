"""Evaluate the validation-selected sweep winner once on held-out test data."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.data_utils import load_processed
from src.metrics import compute_metrics, cost_weighted_score, logits_to_proba
from src.utils import sanitize_json


def main():
    parser = argparse.ArgumentParser(
        description="Run final test evaluation for the best sweep run."
    )
    parser.add_argument("--summary",  required=True,
                        help="Path to the sweep summary.csv")
    parser.add_argument("--data-dir", required=True,
                        help="Path to data/processed/")
    parser.add_argument("--output",   default="final_test_results.json",
                        help="Where to write the test results JSON")
    parser.add_argument("--cost-fn",  type=float, default=5.0,
                        help="Cost of a false negative (missed positive)")
    parser.add_argument("--cost-fp",  type=float, default=1.0,
                        help="Cost of a false positive")
    args = parser.parse_args()

    # ── Load summary and select the best run by val AUPRC ────────────────────
    with Path(args.summary).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Summary CSV is empty")

    best = max(rows, key=lambda row: float(row["best_val_auprc"]))

    # ── Load the best run's artifact JSON ────────────────────────────────────
    artifact_path = Path(best["result_path"])
    if not artifact_path.is_absolute():
        artifact_path = Path(args.summary).parent / artifact_path
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    # ── Reload data and reconstruct probabilities ─────────────────────────────
    data  = load_processed(args.data_dir)
    w     = np.asarray(artifact["best_w"], dtype=float)
    b     = float(artifact["best_b"])
    proba = logits_to_proba(data["X_test"] @ w + b, artifact["loss_fn"])

    # ── Compute metrics ───────────────────────────────────────────────────────
    metrics = compute_metrics(data["y_test"], proba)
    cost    = cost_weighted_score(data["y_test"], proba, args.cost_fn, args.cost_fp)

    result = {
        "selected_run_id":   artifact["run_id"],
        "selection_metric":  "best_val_auprc",
        "best_val_auprc":    artifact["best_val_auprc"],
        "test_metrics":      metrics,
        "confusion_matrix":  [[metrics["tn"], metrics["fp"]],
                               [metrics["fn"], metrics["tp"]]],
        "cost_weighted_score": cost,
        "cost_fn":           args.cost_fn,
        "cost_fp":           args.cost_fp,
    }

    Path(args.output).write_text(
        json.dumps(sanitize_json(result), indent=2), encoding="utf-8"
    )
    print(json.dumps(sanitize_json(result), indent=2))


if __name__ == "__main__":
    main()
