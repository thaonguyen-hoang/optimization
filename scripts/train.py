"""
scripts/train.py
================
CLI for a single training run. Builds loss/optimizer/regularizer from
arguments, runs train_logreg, and writes all artifacts to
``runs/<run_id>/``:

    config.json        : full hyperparameter config
    history.npz        : logged train/val loss + metrics + grad norm + time
    checkpoints/best.npz, last.npz   : (w, b) arrays
    metrics.json       : best val metrics + final test metrics
    best.json          : pointer to best checkpoint + epoch

Example:
    python -m scripts.train --loss bce --reg l2 --lam 1e-2 \
        --optimizer gd --lr 1e-2 --epochs 50
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# Make src/ importable when run as a module from repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_utils import load_processed  # noqa: E402
from losses import LOSS_REGISTRY, build_loss  # noqa: E402
from optimizers import build_optimizer, FULL_BATCH_OPTIMIZERS  # noqa: E402
from regularizers import build_regularizer  # noqa: E402
from train import train_logreg, predict_logits, hessian_eigs_logreg  # noqa: E402
from metrics import compute_metrics, logits_to_proba  # noqa: E402

try:
    from plotting import plot_convergence_full  # noqa: E402
except Exception:
    plot_convergence_full = None


def parse_args():
    p = argparse.ArgumentParser(description="Single training run for logistic regression.")
    p.add_argument("--loss", choices=list(LOSS_REGISTRY.keys()), default="bce")
    p.add_argument("--reg", choices=["none", "l1", "l2"], default="none")
    p.add_argument("--lam", type=float, default=0.0, help="regularization strength")
    p.add_argument("--optimizer", default="gd",
                   help="gd|nag|newton|sgd")
    p.add_argument("--lr", type=float, default=1e-2, help="fixed step size (fixed-step mode)")
    p.add_argument("--lr-schedule", choices=["fixed", "diminishing"], default="fixed", help="lr schedule for SGD")
    p.add_argument("--backtracking", action="store_true",
                   help="use backtracking line search (Armijo/Parabol) for gd/nag/newton")
    p.add_argument("--initial_lr", type=float, default=1.0, help="backtracking initial step")
    p.add_argument("--w-pos", type=float, default=1.0, help="weighted BCE positive weight")
    p.add_argument("--w-neg", type=float, default=1.0, help="weighted BCE negative weight")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--loss-epsilon", type=float, default=0.0, help="early stopping threshold for loss change")
    p.add_argument("--batch-size", type=int, default=256, help="SGD mini-batch size")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every-iters", type=int, default=50)
    p.add_argument("--verbose-every", type=int, default=10)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--run-id", default=None, help="override auto-generated run id")
    p.add_argument("--no-standardize", action="store_true")
    p.add_argument("--eval-test", action="store_true",
                   help="evaluate on test set with best checkpoint")
    p.add_argument("--save-figures", action="store_true")
    p.add_argument("--hessian-spectrum", action="store_true",
                   help="compute and save Hessian eigenvalue spectrum (train subsample)")
    return p.parse_args()


def make_run_id(args):
    if args.run_id:
        return args.run_id
    tag = f"{args.loss}_{args.reg}_{args.optimizer}_lr{args.lr}_lam{args.lam}"
    stamp = time.strftime("%m-%d-%H-%M")
    return f"{tag}_{stamp}"


def build_config(args):
    return {
        "loss": args.loss, "reg": args.reg, "lam": args.lam,
        "optimizer": args.optimizer, "lr": args.lr,
        "lr_schedule": args.lr_schedule,
        "backtracking": bool(args.backtracking), "initial_lr": args.initial_lr,
        "w_pos": args.w_pos, "w_neg": args.w_neg,
        "epochs": args.epochs, "loss_epsilon": args.loss_epsilon, 
        "batch_size": args.batch_size,
        "seed": args.seed, "log_every_iters": args.log_every_iters,
        "standardize": not args.no_standardize,
    }


def main():
    args = parse_args()

    opt_name = args.optimizer
    
    loss_fn = build_loss(args.loss, w_pos=args.w_pos, w_neg=args.w_neg)
    regularizer = build_regularizer(args.reg, lam=args.lam)
    optimizer = build_optimizer(
        opt_name, lr=args.lr,
        backtracking=bool(args.backtracking),
        schedule=args.lr_schedule,
        initial_lr=args.initial_lr,
    )

    # enforce full-batch for second-order / backtracking first-order methods
    if opt_name in FULL_BATCH_OPTIMIZERS:
        batch_size_for_run = None  # train.py will use full set
    else:
        batch_size_for_run = args.batch_size

    # data
    data = load_processed(args.data_dir, standardize_cols=not args.no_standardize)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    print(f"train={X_train.shape} val={X_val.shape} test={X_test.shape}")
    print(f"class balance (train): {data['class_counts']}")

    t0 = time.time()
    result = train_logreg(
        X_train, y_train, X_val, y_val,
        loss_fn=loss_fn, optimizer=optimizer, regularizer=regularizer,
        n_epochs=args.epochs,
        batch_size=len(X_train) if batch_size_for_run is None else batch_size_for_run,
        seed=args.seed, log_every_iters=args.log_every_iters,
        verbose_every=args.verbose_every, loss_epsilon=args.loss_epsilon,
    )
    train_time = time.time() - t0

    # ----- write artifacts -----
    run_id = make_run_id(args)
    run_dir = os.path.join(args.out_dir, run_id)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    config = build_config(args)
    config["optimizer_resolved"] = opt_name
    config["train_time_sec"] = train_time
    config["feature_cols"] = data["feature_cols"]
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    np.savez(os.path.join(ckpt_dir, "best.npz"),
             w=result["best_w"], b=np.array(result["best_b"]))
    np.savez(os.path.join(ckpt_dir, "last.npz"),
             w=result["w"], b=np.array(result["b"]))

    np.savez(os.path.join(run_dir, "history.npz"), **result["history"])

    # val metrics at best checkpoint
    p_val_best = logits_to_proba(predict_logits(X_val, result["best_w"], result["best_b"]),
                                 loss_fn.name)
    val_metrics = compute_metrics(y_val, p_val_best)

    out = {
        "run_id": run_id,
        "best_epoch": result["best_epoch"],
        "best_val_auprc": result["best_val_auprc"],
        "val_metrics": val_metrics,
        "train_time_sec": train_time,
    }

    if args.eval_test:
        p_test = logits_to_proba(
            predict_logits(X_test, result["best_w"], result["best_b"]), loss_fn.name)
        out["test_metrics"] = compute_metrics(y_test, p_test)

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(run_dir, "best.json"), "w") as f:
        json.dump({"checkpoint": "checkpoints/best.npz",
                   "best_epoch": result["best_epoch"],
                   "best_val_auprc": result["best_val_auprc"]}, f, indent=2)

    if args.save_figures and plot_convergence_full is not None:
        import matplotlib
        matplotlib.use("Agg")
        fig = plot_convergence_full(result["history"], title=run_id, x_axis="epoch")
        fig.savefig(os.path.join(run_dir, "convergence.png"), dpi=130)
        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass

    if args.hessian_spectrum:
        eigs = hessian_eigs_logreg(X_train, result["best_w"], result["best_b"],
                                   lam_l2=args.lam if args.reg == "l2" else 0.0,
                                   sample_size=20000, seed=args.seed)
        np.save(os.path.join(run_dir, "hessian_eigs.npy"), eigs)

    print(json.dumps(out, indent=2))
    print(f"artifacts -> {run_dir}")


if __name__ == "__main__":
    main()
