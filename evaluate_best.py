"""
evaluate_best.py — Final test-set evaluation of the best config per loss.

For each loss function, this script:
  1. Reads all per-run JSON files in results/.
  2. Selects the config with the highest best_val_auroc per loss.
  3. Re-trains that config on train+val data (optional) or uses saved weights.
  4. Evaluates on the held-out test set.
  5. Prints and saves a summary table.

Usage:
    python evaluate_best.py --data_dir data/processed --results_dir results

Note: Since results JSON files don't store the final weights (only metrics and
curves), the best config is re-trained from scratch on train data and then
evaluated on test.  If you want to avoid re-training, save `model.w_` in
runner.py results.
"""

import argparse
import json
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import load_data, set_seed, compute_imbalance_ratio, get_logger, sigmoid
from src.metrics import compute_metrics
from src.step_size.lipschitz import lipschitz_constant
from src.step_size.fixed import FixedLR
from src.step_size.armijo import ArmijoLineSearch
from src.regularizers.l1 import L1Regularizer
from src.regularizers.l2 import L2Regularizer
from src.losses.bce import BCELoss
from src.losses.weighted_bce import WeightedBCELoss
from src.losses.squared_hinge import SquaredHingeLoss
from src.losses.focal import FocalLoss
from src.optimizers.gd import GradientDescent
from src.optimizers.sgd import SGD
from src.optimizers.nag import NAG
from src.optimizers.newton import Newton
from src.optimizers.lbfgs import LBFGS
from src.model import LogisticRegression

logger = get_logger("evaluate_best")

LOSS_NAMES = ["bce", "weighted_bce", "squared_hinge", "focal"]


def find_best_per_loss(results_dir: Path) -> dict[str, dict]:
    """
    Scan all JSON result files and return the best run per loss
    (by best_val_auroc).
    """
    best = {}
    # Use rglob to search recursively in subdirectories (e.g. results/A, results/D_controlled)
    for jf in results_dir.rglob("*.json"):
        with open(jf) as f:
            r = json.load(f)
        loss = r.get("loss_fn", "")
        if loss not in best or r.get("best_val_auroc", 0) > best[loss].get("best_val_auroc", 0):
            best[loss] = r
    return best


def rebuild_and_evaluate(best_run: dict, X_train, y_train, X_val, y_val,
                         X_test, y_test, R: float, cfg_seed: int = 42) -> dict:
    """Re-train the best config and evaluate on test set."""
    hp    = best_run["hyperparams"]
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

    # Build loss
    if loss_name == "bce":
        loss_fn = BCELoss()
    elif loss_name == "weighted_bce":
        loss_fn = WeightedBCELoss(w_pos=c_scale * R, w_neg=1.0)
    elif loss_name == "squared_hinge":
        loss_fn = SquaredHingeLoss()
    elif loss_name == "focal":
        loss_fn = FocalLoss(gamma=gamma, alpha=alpha)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

    # Build regularizer
    regularizer = None
    if reg_type == "l2":
        regularizer = L2Regularizer(lambda_reg)
    elif reg_type == "l1":
        regularizer = L1Regularizer(lambda_reg)

    # Build step size
    if step_type == "fixed":
        step_size = FixedLR(c=c_val, L=L, decay_rate=decay_rate)
    else:
        step_size = ArmijoLineSearch(alpha_init=1.0, beta=0.5, c_armijo=1e-4)

    # Build optimizer
    use_proximal = (reg_type == "l1")
    if opt_name == "gd":
        optimizer = GradientDescent(step_size, regularizer, use_proximal)
    elif opt_name == "sgd":
        optimizer = SGD(step_size, regularizer, use_proximal)
    elif opt_name == "nag":
        optimizer = NAG(step_size, momentum=0.9, regularizer=regularizer,
                        use_proximal=use_proximal)
    elif opt_name == "newton":
        optimizer = Newton(step_size, epsilon_damp=1e-6)
    elif opt_name == "lbfgs":
        optimizer = LBFGS(maxiter=1, m=10)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    model = LogisticRegression(
        loss_fn=loss_fn,
        optimizer=optimizer,
        regularizer=regularizer,
        reg_type=reg_type,
        max_epochs=best_run.get("epochs_run", 1000),
        tol_obj=1e-6,
        patience_inner=5,
        batch_size=batch_size,
        seed=cfg_seed,
        verbose=0,
    )
    model.fit(X_train, y_train, X_val, y_val)

    test_metrics = compute_metrics(y_test, model.predict_proba(X_test))
    val_metrics  = compute_metrics(y_val,  model.predict_proba(X_val))
    return {
        "loss_fn":    loss_name,
        "optimizer":  opt_name,
        "reg_type":   reg_type,
        "step_type":  step_type,
        "hyperparams": hp,
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
        json.dump(final_results, f, indent=2)
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
