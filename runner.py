"""
runner.py — Main experiment runner.

Usage:
    python runner.py --config configs/bce.yaml --data_dir data/processed
    python runner.py --config configs/focal.yaml --data_dir data/processed --dry_run

For each loss config, the runner:
  1. Loads train/val data and computes Lipschitz constant L.
  2. Builds the combinatorial sweep (optimizer × reg × step_size × hyperparams).
  3. Trains a model for each combination.
  4. Applies Level-2 early stopping per run (val AUROC patience).
  5. Saves per-run results to results/<run_id>.json.

Level-2 patience is applied PER config run, monitoring val AUROC.
After all runs, a summary CSV is saved to results/<loss>_summary.csv.
"""

import argparse
import time
import json
import csv
import warnings
import itertools
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from src.utils import load_data, set_seed, compute_imbalance_ratio, get_logger
from src.metrics import compute_metrics, auroc as compute_auroc
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

logger = get_logger("runner")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ─── Helper builders ─────────────────────────────────────────────────────────

def build_loss(cfg: dict, R: float = 1.0, gamma: float = 2.0, alpha: float = 0.5,
               c_scale: float = 1.0):
    """Instantiate the loss function from config."""
    loss_name = cfg["loss"]
    if loss_name == "bce":
        return BCELoss()
    elif loss_name == "weighted_bce":
        w_pos = c_scale * R
        return WeightedBCELoss(w_pos=w_pos, w_neg=1.0)
    elif loss_name == "squared_hinge":
        return SquaredHingeLoss()
    elif loss_name == "focal":
        return FocalLoss(gamma=gamma, alpha=alpha)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")


def build_regularizer(reg_type: str, lambda_reg: float):
    if reg_type == "none":
        return None
    elif reg_type == "l2":
        return L2Regularizer(lambda_reg)
    elif reg_type == "l1":
        return L1Regularizer(lambda_reg)
    else:
        raise ValueError(f"Unknown regularizer: {reg_type}")


def build_step_size(step_type: str, c: float, L: float, decay_rate: float,
                    cfg_bt: dict, opt_name: str = ""):
    if step_type == "fixed":
        # For Newton: c is the direct step multiplier, NOT c/L.
        # Newton already incorporates curvature via H^{-1}; using c/L
        # would make the step size depend on data scale in a meaningless way.
        L_eff = 1.0 if opt_name == "newton" else L
        return FixedLR(c=c, L=L_eff, decay_rate=decay_rate)
    elif step_type == "backtracking":
        return ArmijoLineSearch(
            alpha_init=cfg_bt["alpha_init"],
            beta=cfg_bt["beta"],
            c_armijo=cfg_bt["c_armijo"],
        )
    else:
        raise ValueError(f"Unknown step_type: {step_type}")


def build_optimizer(opt_name: str, step_size, regularizer, reg_type: str,
                    cfg: dict, lbfgs_maxiter: int | None = None):
    use_proximal = (reg_type == "l1")
    momentum = 0.9  # fixed NAG momentum

    if opt_name == "gd":
        return GradientDescent(step_size, regularizer, use_proximal)
    elif opt_name == "sgd":
        return SGD(step_size, regularizer, use_proximal)
    elif opt_name == "nag":
        return NAG(step_size, momentum=momentum, regularizer=regularizer,
                   use_proximal=use_proximal)
    elif opt_name == "newton":
        eps = cfg.get("newton_epsilon_damp", cfg.get("newton", {}).get("epsilon_damp", 1e-6))
        return Newton(step_size, epsilon_damp=eps)
    elif opt_name == "lbfgs":
        # lbfgs_maxiter can be overridden per-run (for epoch-level vs full-convergence comparison)
        maxiter = lbfgs_maxiter or cfg.get("lbfgs", {}).get("maxiter", 1)
        m = cfg.get("lbfgs", {}).get("m", 10)
        return LBFGS(maxiter=maxiter, m=m)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


# ─── Run ID builder ──────────────────────────────────────────────────────────

def make_run_id(loss_name, opt_name, reg_type, step_type, hyperparams: dict) -> str:
    parts = [loss_name, opt_name, reg_type, step_type]
    for k, v in sorted(hyperparams.items()):
        if isinstance(v, float):
            parts.append(f"{k}{v:.0e}" if v < 0.01 else f"{k}{v}")
        else:
            parts.append(f"{k}{v}")
    return "_".join(parts).replace(".", "p")


# ─── Single run ──────────────────────────────────────────────────────────────

def run_single(
    cfg: dict,
    X_train, y_train, X_val, y_val,
    loss_fn, optimizer, regularizer, reg_type,
    batch_size: int,
    run_id: str,
    dry_run: bool = False,
) -> dict:
    """Train one model configuration and return the results dict."""

    model = LogisticRegression(
        loss_fn=loss_fn,
        optimizer=optimizer,
        regularizer=regularizer,
        reg_type=reg_type,
        max_epochs=1 if dry_run else cfg["max_epochs"],
        tol_obj=cfg["tol_obj"],
        patience_inner=cfg["patience_inner"],
        batch_size=batch_size,
        seed=cfg["seed"],
        verbose=0,
    )

    # Level-2 early stopping state (val AUROC patience)
    patience_outer = cfg["patience_outer"]
    best_auroc = -1.0
    no_improve_count = 0
    best_w = None

    # Train epoch by epoch to enable Level-2 stopping
    # (We override the model's internal loop and drive it manually)
    # Re-initialize manually for epoch-level control:
    n, d = X_train.shape
    model.w_ = np.zeros(d)
    model.optimizer.reset()
    model.train_loss_curve_ = []
    model.val_loss_curve_ = []
    model.val_auroc_curve_ = []
    model.stop_reason_ = "max_epochs"

    from src.metrics import auroc as _auroc
    from src.utils import iter_batches, sigmoid

    rng = np.random.default_rng(cfg["seed"])
    prev_loss = None
    tol_counter = 0
    max_ep = 1 if dry_run else cfg["max_epochs"]

    # ── Timing ──────────────────────────────────────────────────────────────
    epoch_times: list[float] = []   # wall-clock seconds for optimizer step only
    run_start = time.perf_counter()

    hessian_fn = (model._total_hessian if optimizer.name == "newton" else None)

    for epoch in range(1, max_ep + 1):
        model.epochs_run_ = epoch

        # ── Time the optimizer step only (excludes metric computation) ────────
        t0 = time.perf_counter()
        if optimizer.name == "sgd":
            for X_b, y_b in iter_batches(X_train, y_train,
                                         batch_size=batch_size,
                                         shuffle=True, rng=rng):
                model.w_ = optimizer.step(
                    model.w_,
                    loss_fn=model._total_loss,
                    grad_fn=model._total_grad,
                    X=X_b, y=y_b,
                    X_full=X_train, y_full=y_train,
                )
        else:
            model.w_ = optimizer.step(
                model.w_,
                loss_fn=model._total_loss,
                grad_fn=model._total_grad,
                X=X_train, y=y_train,
                hessian_fn=hessian_fn,
            )
        epoch_times.append(time.perf_counter() - t0)

        train_loss = model._total_loss(model.w_, X_train, y_train)
        val_loss   = model._total_loss(model.w_, X_val, y_val)
        val_auc    = _auroc(y_val, sigmoid(X_val @ model.w_))

        model.train_loss_curve_.append(train_loss)
        model.val_loss_curve_.append(val_loss)
        model.val_auroc_curve_.append(val_auc)

        # Level-1 early stop
        if prev_loss is not None:
            if abs(train_loss - prev_loss) <= cfg["tol_obj"]:
                tol_counter += 1
                if tol_counter >= cfg["patience_inner"]:
                    model.stop_reason_ = "tol_obj"
                    break
            else:
                tol_counter = 0
        prev_loss = train_loss

        # Level-2 early stop (val AUROC patience)
        if val_auc > best_auroc + 1e-6:
            best_auroc = val_auc
            best_w = model.w_.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience_outer:
                model.stop_reason_ = "patience_outer"
                break

    # Use best_w if available
    if best_w is not None:
        model.w_ = best_w

    total_time_s = time.perf_counter() - run_start   # includes metric computation

    # Final metrics
    from src.metrics import compute_metrics
    train_metrics = compute_metrics(y_train, sigmoid(X_train @ model.w_))
    val_metrics   = compute_metrics(y_val,   sigmoid(X_val   @ model.w_))

    return {
        "run_id":              run_id,
        "epochs_run":          model.epochs_run_,
        "stopped_early":       model.stop_reason_ != "max_epochs",
        "stop_reason":         model.stop_reason_,
        # ── Timing ────────────────────────────────────────────────────────
        "total_time_s":        round(total_time_s, 4),
        "time_per_epoch_s":    [round(t, 6) for t in epoch_times],
        "mean_epoch_time_s":   round(float(np.mean(epoch_times)), 6) if epoch_times else 0.0,
        "cumulative_time_s":   [round(float(np.cumsum(epoch_times)[i]), 4)
                                for i in range(len(epoch_times))],
        # ── Curves ────────────────────────────────────────────────────────
        "train_loss_curve":    model.train_loss_curve_,
        "val_loss_curve":      model.val_loss_curve_,
        "val_auroc_curve":     model.val_auroc_curve_,
        "final_metrics": {
            "train": train_metrics,
            "val":   val_metrics,
        },
        "best_val_auroc": best_auroc,
    }


# ─── Sweep builder ───────────────────────────────────────────────────────────

def build_sweep(cfg: dict, X_train: np.ndarray, y_train: np.ndarray, R: float):
    """
    Generate all (config_dict, hyperparams_dict) combinations to sweep.
    Yields dicts describing one run.
    """
    loss_name    = cfg["loss"]
    opt_names    = cfg["optimizers"]
    reg_types    = cfg["regularizations"]
    lambdas      = cfg["regularization"]["lambda"]
    bt_cfg       = cfg["step_size"]["backtracking"]
    c_gd_nag     = cfg["step_size"]["fixed"]["c_gd_nag"]
    c_sgd_vals   = cfg["step_size"]["fixed"]["c_sgd"]
    decay_rates  = cfg["step_size"]["fixed"]["sgd_decay_rate"]
    batch_sizes  = cfg["sgd"]["batch_sizes"]

    # Loss-specific param grids
    focal_pairs  = cfg.get("focal", {}).get("gamma_alpha_pairs", [[2.0, 0.5]])
    c_scales     = cfg.get("weighted_bce", {}).get("c_scale", [1.0])

    step_types   = ["fixed", "backtracking"]

    for opt_name, reg_type in itertools.product(opt_names, reg_types):
        # Skip: Newton/LBFGS + L1
        if opt_name in ("newton", "lbfgs") and reg_type == "l1":
            continue
        # Skip: Newton + focal (no Hessian)
        if opt_name == "newton" and loss_name == "focal":
            continue

        lambda_list = lambdas if reg_type != "none" else [0.0]

        for lambda_reg, step_type in itertools.product(lambda_list, step_types):
            # Compute Lipschitz constant
            L = lipschitz_constant(
                X_train,
                reg_type=reg_type,
                lambda_reg=lambda_reg,
            )

            # c / batch_size ranges depend on optimizer type
            if opt_name == "sgd":
                c_vals   = c_sgd_vals if step_type == "fixed" else [None]
                dr_vals  = decay_rates if step_type == "fixed" else [0.0]
                bs_vals  = batch_sizes
            else:
                c_vals   = c_gd_nag if step_type == "fixed" else [None]
                dr_vals  = [0.0]
                bs_vals  = [32]  # irrelevant for non-SGD

            # Loss-specific hyperparams
            if loss_name == "focal":
                loss_hp_list = [{"gamma": g, "alpha": a} for g, a in focal_pairs]
            elif loss_name == "weighted_bce":
                loss_hp_list = [{"c_scale": cs} for cs in c_scales]
            else:
                loss_hp_list = [{}]

            for c_val, dr, bs, loss_hp in itertools.product(
                c_vals, dr_vals, bs_vals, loss_hp_list
            ):
                hp = {"lambda": lambda_reg, "batch_size": bs, **loss_hp}
                if step_type == "fixed":
                    hp["c"] = c_val
                    hp["decay_rate"] = dr
                    hp["L"] = round(L, 6)
                    hp["lr"] = round(c_val / L, 8)

                run_id = make_run_id(loss_name, opt_name, reg_type, step_type, hp)
                yield {
                    "run_id":     run_id,
                    "opt_name":   opt_name,
                    "reg_type":   reg_type,
                    "step_type":  step_type,
                    "lambda_reg": lambda_reg,
                    "batch_size": bs,
                    "c":          c_val,
                    "decay_rate": dr,
                    "L":          L,
                    "loss_hp":    loss_hp,
                    "hyperparams": hp,
                    "loss_name":  loss_name,       # explicit, matches cfg["loss"]
                    "lbfgs_maxiter": None,         # use cfg default
                    "_effective_cfg": cfg,
                }


# ─── Explicit run builder ─────────────────────────────────────────────────────

DEFAULT_BT_CFG = {"alpha_init": 1.0, "beta": 0.5, "c_armijo": 1e-4}


def build_explicit_runs(
    cfg: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    R: float,
):
    """
    Convert the ``runs:`` list in an explicit-mode YAML config into the same
    dict format as ``build_sweep``.  Each run spec is a flat dict; missing
    fields fall back to global config values.
    """
    bt_cfg = cfg.get("step_size", {}).get("backtracking", DEFAULT_BT_CFG)

    for spec in cfg["runs"]:
        opt_name   = spec["optimizer"]
        reg_type   = spec.get("reg", "none")
        lambda_reg = float(spec.get("lambda", 0.0))
        step_type  = spec.get("step", "fixed")
        loss_name  = spec.get("loss", cfg.get("loss", "bce"))
        batch_size = int(spec.get("batch_size", 32))
        c_val      = float(spec.get("c", 1.0))
        decay_rate = float(spec.get("decay_rate", 0.0))
        lbfgs_mi   = spec.get("lbfgs_maxiter", None)

        # Loss-specific params
        gamma   = float(spec.get("gamma",   2.0))
        alpha   = float(spec.get("alpha",   0.5))
        c_scale = float(spec.get("c_scale", 1.0))

        # Compute Lipschitz constant for this run's reg setting
        w_pos = c_scale * R if loss_name == "weighted_bce" else 1.0
        L = lipschitz_constant(
            X_train, reg_type=reg_type,
            lambda_reg=lambda_reg, w_pos=w_pos,
        )

        hyperparams: dict = {
            "lambda":     lambda_reg,
            "batch_size": batch_size,
            "L":          round(L, 6),
        }
        if step_type == "fixed":
            L_eff = 1.0 if opt_name == "newton" else L
            hyperparams["c"]          = c_val
            hyperparams["decay_rate"] = decay_rate
            hyperparams["lr"]         = round(c_val / L_eff, 8)
        if loss_name == "focal":
            hyperparams["gamma"] = gamma
            hyperparams["alpha"] = alpha
        if loss_name == "weighted_bce":
            hyperparams["c_scale"] = c_scale
        if lbfgs_mi is not None:
            hyperparams["lbfgs_maxiter"] = lbfgs_mi

        run_label = spec.get("id", "")
        base_id   = make_run_id(loss_name, opt_name, reg_type, step_type, hyperparams)
        run_id    = f"{run_label}_{base_id}" if run_label else base_id

        # Per-run effective config (inherits global, allows max_epochs override)
        effective_cfg = dict(cfg)
        if "max_epochs" in spec:
            effective_cfg["max_epochs"] = spec["max_epochs"]

        yield {
            "run_id":        run_id,
            "opt_name":      opt_name,
            "reg_type":      reg_type,
            "step_type":     step_type,
            "lambda_reg":    lambda_reg,
            "batch_size":    batch_size,
            "c":             c_val,
            "decay_rate":    decay_rate,
            "L":             L,
            "loss_hp":       {"gamma": gamma, "alpha": alpha, "c_scale": c_scale},
            "hyperparams":   hyperparams,
            "loss_name":     loss_name,
            "lbfgs_maxiter": lbfgs_mi,
            "_effective_cfg": effective_cfg,
        }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run optimization experiments.")
    parser.add_argument("--config",   required=True,  help="Path to YAML config")
    parser.add_argument("--data_dir", required=True,  help="Path to data/processed/")
    parser.add_argument("--dry_run",  action="store_true",
                        help="Run 1 epoch only (sanity check)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run and overwrite existing result files")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    data = load_data(args.data_dir)
    X_train, y_train = data["train"]
    X_val,   y_val   = data["val"]

    R = compute_imbalance_ratio(y_train)
    logger.info(f"Class imbalance ratio R = N_neg/N_pos = {R:.3f}")

    # ── Detect mode: explicit-run list or grid sweep ──────────────────────────
    explicit_mode = "runs" in cfg
    group_tag     = cfg.get("group", cfg.get("loss", "exp"))

    if explicit_mode:
        logger.info(f"Explicit-run mode | group: {group_tag} | "
                    f"{len(cfg['runs'])} runs defined")
        sweep = list(build_explicit_runs(cfg, X_train, y_train, R))
    else:
        loss_name = cfg["loss"]
        sweep = list(build_sweep(cfg, X_train, y_train, R))

    logger.info(f"Total runs to execute: {len(sweep)}")

    # ── Results sub-directory per group ──────────────────────────────────────
    group_results_dir = RESULTS_DIR / group_tag
    group_results_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for run_cfg in tqdm(sweep, desc=f"[{group_tag}]"):
        run_id        = run_cfg["run_id"]
        run_loss_name = run_cfg.get("loss_name", cfg.get("loss", "unknown"))
        effective_cfg = run_cfg.get("_effective_cfg", cfg)
        out_path      = group_results_dir / f"{run_id}.json"

        if out_path.exists() and not args.overwrite:
            logger.debug(f"Skipping (exists): {run_id}")
            continue

        # Build loss object (use run-level loss name for explicit mode)
        tmp_cfg = dict(effective_cfg)
        tmp_cfg["loss"] = run_loss_name
        loss_fn = build_loss(
            tmp_cfg, R=R,
            gamma=run_cfg["loss_hp"].get("gamma", 2.0),
            alpha=run_cfg["loss_hp"].get("alpha", 0.5),
            c_scale=run_cfg["loss_hp"].get("c_scale", 1.0),
        )

        bt_cfg = effective_cfg.get("step_size", {}).get("backtracking", DEFAULT_BT_CFG)
        regularizer = build_regularizer(run_cfg["reg_type"], run_cfg["lambda_reg"])
        step_size   = build_step_size(
            run_cfg["step_type"],
            c=run_cfg["c"] or 1.0,
            L=run_cfg["L"],
            decay_rate=run_cfg["decay_rate"],
            cfg_bt=bt_cfg,
            opt_name=run_cfg["opt_name"],
        )
        optimizer = build_optimizer(
            run_cfg["opt_name"], step_size, regularizer,
            run_cfg["reg_type"], effective_cfg,
            lbfgs_maxiter=run_cfg.get("lbfgs_maxiter"),
        )

        try:
            result = run_single(
                effective_cfg, X_train, y_train, X_val, y_val,
                loss_fn, optimizer, regularizer, run_cfg["reg_type"],
                batch_size=run_cfg["batch_size"],
                run_id=run_id,
                dry_run=args.dry_run,
            )
        except Exception as e:
            logger.warning(f"Run {run_id} failed: {e}")
            continue

        # Attach metadata
        result["group"]         = group_tag
        result["loss_fn"]       = run_loss_name
        result["optimizer"]     = run_cfg["opt_name"]
        result["regularization"]= run_cfg["reg_type"]
        result["step_size_type"]= run_cfg["step_type"]
        result["hyperparams"]   = run_cfg["hyperparams"]

        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        summary_rows.append({
            "run_id":             run_id,
            "group":              group_tag,
            "loss_fn":            run_loss_name,
            "optimizer":          run_cfg["opt_name"],
            "regularization":     run_cfg["reg_type"],
            "step_type":          run_cfg["step_type"],
            "epochs_run":         result["epochs_run"],
            "stop_reason":        result["stop_reason"],
            "total_time_s":       result["total_time_s"],
            "mean_epoch_time_s":  result["mean_epoch_time_s"],
            "best_val_auroc":     result["best_val_auroc"],
            "val_f1":             result["final_metrics"]["val"].get("f1", ""),
            "val_auprc":          result["final_metrics"]["val"].get("auprc", ""),
        })

    # Save summary CSV
    if summary_rows:
        csv_path = group_results_dir / f"{group_tag}_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        logger.info(f"Summary saved to {csv_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
