"""
evaluate_best.py — Final test-set evaluation of the best config per loss.

For each loss function, this script:
  1. Reads all per-run JSON files in results/.
  2. Selects the config with the highest best_val_auroc per loss.
  3. Re-trains that config on the same train split (and seed) used by the
     sweep, reproducing the exact hyperparameters stored in ``training_cfg``.
  4. Evaluates on the held-out test set.
  5. Prints and saves a summary table.

Usage:
    python evaluate_best.py --data_dir data/processed --results_dir results
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import load_data, set_seed, compute_imbalance_ratio, get_logger, sigmoid, sanitize_json
from src.metrics import compute_metrics
from runner import (train_logreg, build_loss, build_regularizer,
                    build_step_size, build_optimizer, DEFAULT_BT_CFG)

logger = get_logger("evaluate_best")

EXCLUDED_RESULTS = {"final_test_results.json", "final_test_results.csv"}


def find_best_per_loss(results_dir: Path) -> dict[str, dict]:
    """Scan all JSON result files and return the best run per loss (by best_val_auroc)."""
    best = {}
    for jf in results_dir.rglob("*.json"):
        if jf.name in EXCLUDED_RESULTS:
            continue
        with open(jf) as f:
            r = json.load(f)
        cand = r.get("best_val_auroc")
        if cand is None:
            # e.g. sanitize_json wrote NaN -> null; can't rank it.
            continue
        loss = r.get("loss_fn", "")
        if loss not in best or cand > (best[loss].get("best_val_auroc") or 0):
            best[loss] = r
    return best


def rebuild_and_evaluate(best_run: dict, X_train, y_train, X_val, y_val,
                         X_test, y_test, R: float, cfg_seed: int = 42) -> dict:
    """Re-train the best config (reproducing training_cfg) and evaluate on test set."""
    hp    = best_run["hyperparams"]
    tcfg  = best_run.get("training_cfg", {})
    loss_name = best_run["loss_fn"]
    opt_name  = best_run["optimizer"]
    reg_type  = best_run["regularization"]
    step_type = best_run["step_size_type"]

    lambda_reg = hp.get("lambda", 0.0)
    batch_size = hp.get("batch_size", 32)
    gamma      = hp.get("gamma", 2.0)
    alpha      = hp.get("alpha", 0.5)
    c_scale    = hp.get("c_scale", 1.0)
    c_val      = hp.get("c", 1.0)
    L          = hp.get("L", 1.0)
    decay_rate = hp.get("decay_rate", 0.0)

    # Rebuild every component from the shared runner builders so the
    # re-evaluation matches the original run's settings.
    loss_fn = build_loss(
        {"loss": loss_name}, R=R,
        gamma=gamma, alpha=alpha, c_scale=c_scale,
    )
    regularizer = build_regularizer(reg_type, lambda_reg)

    bt_cfg = tcfg.get("backtracking", DEFAULT_BT_CFG)
    step_size = build_step_size(
        step_type, c=c_val, L=L, decay_rate=decay_rate,
        cfg_bt=bt_cfg, opt_name=opt_name,
    )

    opt_cfg = {}
    if "newton_epsilon_damp" in tcfg:
        opt_cfg["newton_epsilon_damp"] = tcfg["newton_epsilon_damp"]
    if "lbfgs_m" in tcfg or "lbfgs_maxiter" in tcfg:
        opt_cfg["lbfgs"] = {
            "m":      tcfg.get("lbfgs_m", 10),
            "maxiter": tcfg.get("lbfgs_maxiter", 1),
        }
    optimizer = build_optimizer(
        opt_name, step_size, regularizer, reg_type, opt_cfg,
        momentum=tcfg.get("nag_momentum"),
    )

    result = train_logreg(
        X_train, y_train, X_val, y_val,
        loss_fn, optimizer, regularizer,
        n_epochs=tcfg.get("n_epochs", best_run.get("epochs_run", 1000)),
        batch_size=tcfg.get("batch_size", batch_size),
        seed=tcfg.get("seed", cfg_seed),
        verbose_every=0,
        tol_grad=tcfg.get("tol_grad", 1e-4),
        patience_inner=tcfg.get("patience_inner", 5),
        patience_outer=tcfg.get("patience_outer", 10),
        dry_run=False,
    )

    w, b = result["w"], result["b"]
    z_val = X_val @ w + b
    z_test = X_test @ w + b

    test_metrics = compute_metrics(y_test, sigmoid(z_test))
    val_metrics  = compute_metrics(y_val,  sigmoid(z_val))
    return {
        "loss_fn":    loss_name,
        "optimizer":  opt_name,
        "reg_type":   reg_type,
        "step_type":  step_type,
        "hyperparams": hp,
        "training_cfg": tcfg,
        "val_metrics":  val_metrics,
        "test_metrics": test_metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default="data/processed")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--out",         default="results/final_test_results.json")
    args = parser.parse_args()

    data = load_data(args.data_dir)
    X_train, y_train = data["train"]
    X_val,   y_val   = data["val"]
    X_test,  y_test  = data["test"]
    R = compute_imbalance_ratio(y_train)

    results_dir = Path(args.results_dir)
    best_per_loss = find_best_per_loss(results_dir)

    final_results = {}
    rows = []

    for loss_name, best_run in best_per_loss.items():
        logger.info(f"Re-evaluating best {loss_name} config: {best_run['run_id']}")
        result = rebuild_and_evaluate(
            best_run, X_train, y_train, X_val, y_val, X_test, y_test, R
        )
        final_results[loss_name] = result
        tm = result["test_metrics"]
        rows.append({
            "loss_fn":   loss_name,
            "optimizer": result["optimizer"],
            "reg_type":  result["reg_type"],
            "step_type": result["step_type"],
            "val_auroc": round(result["val_metrics"]["auroc"], 4),
            "test_auroc":    round(tm["auroc"], 4),
            "test_auprc":   round(tm["auprc"], 4),
            "test_f1":      round(tm["f1"], 4),
            "test_precision": round(tm["precision"], 4),
            "test_recall":  round(tm["recall"], 4),
            "test_accuracy":round(tm["accuracy"], 4),
        })

    # Save JSON
    with open(args.out, "w") as f:
        json.dump(sanitize_json(final_results), f, indent=2)
    logger.info(f"Final results saved to {args.out}")

    # Print table
    df = pd.DataFrame(rows)
    print("\n=== Final Test Set Results (Best Config per Loss) ===")
    print(df.to_string(index=False))

    # Save CSV
    csv_path = Path(args.results_dir) / "final_test_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"CSV saved to {csv_path}")


if __name__ == "__main__":
    main()