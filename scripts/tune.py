"""
scripts/tune.py
===============
CLI grid-search tuning for one loss function across the three objectives
(loss, loss+L2, loss+L1) and all applicable optimizers / step sizes.

Strategy (per the agreed plan):
  Stage A (loss-hparam pre-sweep): only for weighted_bce. Pick the best
          w_pos using a cheap GD run with reg=none. Other losses skip.
  Stage B (full grid): for each objective (none / l2 / l1) sweep the
          eligible optimizers x step sizes x lambdas. Select the best
          configuration by Validation AUPRC.

  * Smooth objectives (none, l2): GD, SGD, AcceleratedGD (fixed lr grid)
    + GD_backtrack, AccGD_backtrack, Newton, L-BFGS (backtracking and
    fixed-step modes).
  * Non-smooth objective (l1): GD, SGD, AcceleratedGD with subgradient
    and fixed lr only. No backtracking, no second-order.

Each trial reuses the data arrays (loaded once) and writes a per-trial
row to a summary CSV plus a best-config JSON. Full per-run artifacts can
optionally be kept (``--keep-runs``) but are off by default to save disk.

Example:
    python -m scripts.tune --loss bce --tune-epochs 30 --parallel 4
"""

import argparse
import json
import os
import sys
import time
import itertools

import numpy as np
import pandas as pd

try:
    from ._log_utils import start_run_log  # noqa: E402  (python -m scripts.tune)
except ImportError:
    from _log_utils import start_run_log  # noqa: E402  (python scripts/tune.py)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_utils import load_processed  # noqa: E402
from losses import build_loss  # noqa: E402
from optimizers import (build_optimizer, LR_GRID, LAMBDA_GRID,  # noqa: E402
                        FULL_BATCH_OPTIMIZERS, NO_L1_OPTIMIZERS)
from regularizers import build_regularizer  # noqa: E402
from train import train_logreg, predict_logits  # noqa: E402
from metrics import compute_metrics, logits_to_proba  # noqa: E402

# Loss-hparam grid for the pre-sweep (only weighted_bce uses it).
W_POS_GRID = [1.0, 2.0, 4.0, 6.0, 10.0]

# Optimizer groups
SMOOTH_OPTS = ["gd", "nag", "newton"]
NONSMOOTH_OPTS = ["gd", "nag"]

DEFAULT_EPOCHS = 500
DEFAULT_BATCH = 256


def _val_auprc(X_val, y_val, w, b, loss_fn):
    p = logits_to_proba(predict_logits(X_val, w, b), loss_fn.name)
    return compute_metrics(y_val, p)["auprc"]


def _run_trial(X_train, y_train, X_val, y_val,
               loss_name, w_pos, w_neg,
               reg_name, lam, opt_name, lr, schedule, backtracking,
               epochs, batch_size, seed, loss_epsilon,
               initial_lr=1.0):
    """Run one training trial and return (val_auprc, val_metrics, val_loss, train_time, result)."""
    loss_fn = build_loss(loss_name, w_pos=w_pos, w_neg=w_neg)
    regularizer = build_regularizer(reg_name, lam=lam)
    optimizer = build_optimizer(
        opt_name, lr=lr, schedule=schedule,
        backtracking=backtracking,
        initial_lr=initial_lr,
    )
    is_full_batch = opt_name in FULL_BATCH_OPTIMIZERS
    t0 = time.time()
    result = train_logreg(
        X_train, y_train, X_val, y_val,
        loss_fn=loss_fn, optimizer=optimizer, regularizer=regularizer,
        n_epochs=epochs,
        batch_size=len(X_train) if is_full_batch else batch_size,
        seed=seed, verbose_every=0,
        loss_epsilon=loss_epsilon
    )
    train_time = time.time() - t0
    p_val = logits_to_proba(predict_logits(X_val, result["best_w"], result["best_b"]),
                            loss_fn.name)
    m_val = compute_metrics(y_val, p_val)
    # val loss at best epoch (epoch-level entries are non-NaN for val_loss)
    vl = np.asarray(result["history"]["val_loss"])
    vl = vl[~np.isnan(vl)]
    val_loss = float(vl[-1]) if len(vl) else np.nan
    return m_val["auprc"], m_val, val_loss, train_time, result


def stage_a_loss_hparam(data, loss_name, epochs, seed):
    """Pre-sweep loss hyperparameters (w_pos for weighted_bce only)."""
    if loss_name != "weighted_bce":
        return {"w_pos": 1.0, "w_neg": 1.0}
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    best = {"w_pos": 1.0, "score": -np.inf}
    print("[Stage A] sweeping w_pos:", W_POS_GRID)
    for w_pos in W_POS_GRID:
        loss_fn = build_loss("weighted_bce", w_pos=w_pos, w_neg=1.0)
        optimizer = build_optimizer("gd", lr=1e-2)
        result = train_logreg(X_train, y_train, X_val, y_val,
                              loss_fn=loss_fn, optimizer=optimizer,
                              regularizer=build_regularizer("none"),
                              n_epochs=epochs, batch_size=len(X_train),
                              seed=seed, verbose_every=0)
        auprc = _val_auprc(X_val, y_val, result["best_w"], result["best_b"], loss_fn)
        print(f"  w_pos={w_pos}: val AUPRC={auprc:.4f}")
        if auprc > best["score"]:
            best = {"w_pos": w_pos, "score": auprc}
    print(f"[Stage A] best w_pos={best['w_pos']} (AUPRC={best['score']:.4f})")
    return {"w_pos": best["w_pos"], "w_neg": 1.0}


def build_grid(reg_name, backtracking_too):
    """Yield (opt_name, lr, schedule, backtracking) trial specs."""
    trials = []
    
    if reg_name == "l1":
        # Non-smooth: GD/NAG with ISTA/FISTA (fixed & backtrack), SGD (fixed & diminishing)
        for opt in NONSMOOTH_OPTS:
            for lr in LR_GRID:
                trials.append((opt, lr, "fixed", False))
            if backtracking_too:
                trials.append((opt, 1.0, "fixed", True)) # initial_lr=1.0 fixed for bt
        # SGD
        for lr in LR_GRID:
            trials.append(("sgd", lr, "fixed", False))
            trials.append(("sgd", lr, "diminishing", False))
            
    else:  # none or l2 (smooth)
        for opt in SMOOTH_OPTS:
            for lr in LR_GRID:
                if opt == "newton" and lr != 1.0: continue # Pure Newton uses t=1.0
                trials.append((opt, lr, "fixed", False))
            if backtracking_too:
                trials.append((opt, 1.0, "fixed", True))
        # SGD
        for lr in LR_GRID:
            trials.append(("sgd", lr, "fixed", False))
            trials.append(("sgd", lr, "diminishing", False))

    return trials


def run_one(args, data, loss_hp, reg_name, lam, trial):
    opt_name, lr, schedule, backtracking = trial
    try:
        auprc, m_val, val_loss, ttime, _ = _run_trial(
            data["X_train"], data["y_train"], data["X_val"], data["y_val"],
            loss_name=args.loss, w_pos=loss_hp.get("w_pos", 1.0),
            w_neg=loss_hp.get("w_neg", 1.0),
            reg_name=reg_name, lam=lam, opt_name=opt_name, lr=lr,
            schedule=schedule, backtracking=backtracking, epochs=args.tune_epochs,
            batch_size=args.batch_size, seed=args.seed,
            loss_epsilon=args.loss_epsilon,
        )
        # Apply schedule to optimizer name for logging if SGD
        opt_log = opt_name
        if schedule == "diminishing": opt_log += "_diminishing"
        if backtracking: opt_log += "_backtrack"
        
        row = {
            "loss": args.loss, "reg": reg_name, "lam": lam,
            "optimizer": opt_log,
            "lr": lr, "w_pos": loss_hp.get("w_pos", 1.0),
            "val_auprc": auprc, "val_f1_minority": m_val["f1_minority"],
            "val_accuracy": m_val["accuracy"], "val_auroc": m_val["auroc"],
            "val_loss": val_loss,
            "train_time_sec": ttime,
        }
    except Exception as e:
        opt_log = opt_name
        if schedule == "diminishing": opt_log += "_diminishing"
        if backtracking: opt_log += "_backtrack"
        
        row = {
            "loss": args.loss, "reg": reg_name, "lam": lam,
            "optimizer": opt_log,
            "lr": lr, "w_pos": loss_hp.get("w_pos", 1.0),
            "val_auprc": np.nan, "error": str(e)[:120],
        }
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--loss", default="bce")
    p.add_argument("--tune-epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--loss-epsilon", type=float, default=1e-4, help="early stopping threshold for loss change")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--skip-stage-a", action="store_true")
    p.add_argument("--no-backtracking", action="store_true",
                   help="skip backtracking trials (smooth only)")
    p.add_argument("--keep-runs", action="store_true",
                   help="keep per-trial run artifacts (otherwise discarded)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    start_run_log(args.out_dir, f"tune_{args.loss}.log")
    data = load_processed(args.data_dir)
    print(f"train={data['X_train'].shape} val={data['X_val'].shape} "
          f"test={data['X_test'].shape}")
    print(f"class balance (train): {data['class_counts']}")

    # Stage A: loss hyperparameter
    if args.skip_stage_a:
        loss_hp = {"w_pos": 1.0, "w_neg": 1.0}
    else:
        loss_hp = stage_a_loss_hparam(data, args.loss, epochs=10, seed=args.seed)

    rows = []
    t0 = time.time()

    # Objective: none
    print("\n[Stage B] objective = loss (no regularization)")
    for trial in build_grid("none", backtracking_too=not args.no_backtracking):
        row = run_one(args, data, loss_hp, "none", 0.0, trial)
        print(f"  {row['optimizer']:<24} lr={row['lr']:<6} AUPRC={row.get('val_auprc')}")
        rows.append(row)

    # Objective: l2
    print("\n[Stage B] objective = loss + L2")
    for lam in LAMBDA_GRID:
        for trial in build_grid("l2", backtracking_too=not args.no_backtracking):
            row = run_one(args, data, loss_hp, "l2", lam, trial)
            print(f"  lam={lam:<6} {row['optimizer']:<22} lr={row['lr']:<6} AUPRC={row.get('val_auprc')}")
            rows.append(row)

    # Objective: l1
    print("\n[Stage B] objective = loss + L1 (proximal)")
    for lam in LAMBDA_GRID:
        for trial in build_grid("l1", backtracking_too=not args.no_backtracking):
            row = run_one(args, data, loss_hp, "l1", lam, trial)
            print(f"  lam={lam:<6} {row['optimizer']:<22} lr={row['lr']:<6} AUPRC={row.get('val_auprc')}")
            rows.append(row)

    # persist summary
    df = pd.DataFrame(rows)
    summary_path = os.path.join(args.out_dir, f"tune_{args.loss}_summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"\nsummary -> {summary_path}  ({len(df)} trials, {time.time()-t0:.1f}s)")

    # best by val AUPRC
    valid = df.dropna(subset=["val_auprc"])
    if len(valid) == 0:
        print("No valid trials."); return
    best = valid.sort_values("val_auprc", ascending=False).iloc[0]
    best_cfg = {k: (float(best[k]) if isinstance(best[k], (np.floating, float)) else best[k])
                for k in best.index if k != "error"}
    best_path = os.path.join(args.out_dir, f"best_{args.loss}.json")
    with open(best_path, "w") as f:
        json.dump(best_cfg, f, indent=2)
    print(f"best -> {best_path}")
    print(json.dumps(best_cfg, indent=2))


if __name__ == "__main__":
    main()
