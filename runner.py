"""
runner.py — Main experiment runner with inline training loop.

Training loop structure (v1-style):
    For each epoch:
        For SGD: iterate over mini-batches
        For others: single full-batch step
        
        1. Compute logits z = Xw + b
        2. Get gradient from loss_fn: grad_z = loss_fn.grad(z, y)
        3. Chain rule to parameters:
           - grad_w = X.T @ grad_z / n
           - grad_b = mean(grad_z)
        4. Add regularizer gradient (L2) or prepare for prox (L1)
        5. Call optimizer.step(w, b, grad_w, grad_b, **kwargs)
        6. Apply proximal operator if needed (L1)
        7. Record metrics, check stopping criteria

Stops on:
    Level 1: |loss(t) - loss(t-1)| < tol_obj for patience_inner epochs
    Level 2: val AUROC no improvement for patience_outer epochs
"""

import argparse
import time
import json
import csv
import warnings
import itertools
from pathlib import Path
from collections import defaultdict

import numpy as np
import yaml
from tqdm import tqdm

from src.utils import load_data, set_seed, compute_imbalance_ratio, get_logger
from src.metrics import compute_metrics, auroc
from src.step_size.fixed import FixedLR
from src.step_size.armijo import ArmijoLineSearch
from src.step_size.lipschitz import lipschitz_constant
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

logger = get_logger("runner")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ─── Helper: sigmoid ─────────────────────────────────────────────────────────
def sigmoid(z):
    """Numerically stable sigmoid."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


# ─── Training loop (v1-style, inline) ────────────────────────────────────────
def train_logreg(
    X_train, y_train, X_val, y_val,
    loss_fn, optimizer, regularizer,
    n_epochs: int = 100,
    batch_size: int = 256,
    seed: int = 42,
    verbose_every: int = 10,
    tol_obj: float = 1e-6,
    patience_inner: int = 5,
    patience_outer: int = 10,
    dry_run: bool = False,
):
    """
    Train logistic regression: z = Xw + b → σ(z) → binary classification.
    
    Training loop structure:
        Outer loop: epochs (1 to n_epochs)
            For SGD: inner loop over mini-batches
            For others: single full-batch step
            - Record train loss and val AUROC per epoch
            - Level-1 stopping: |loss(t) - loss(t-1)| ≤ tol_obj for patience_inner steps
            - Level-2 stopping: val AUROC no improvement for patience_outer epochs
    
    Returns: dict with trained (w, b) and history.
    """
    rng = np.random.RandomState(seed)
    n, d = X_train.shape
    
    # Initialize parameters
    w = np.zeros(d)
    b = 0.0
    
    # Reset optimizer state
    optimizer.reset(w.shape)
    
    # History tracking
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_auroc": [],
        "wall_time": [],
    }
    
    # Early stopping state
    t0 = time.time()
    
    # Level-1: objective tolerance
    prev_loss = None
    tol_counter = 0
    
    # Level-2: validation AUROC patience
    best_val_auroc = 0.0
    no_improve_outer = 0
    best_w = w.copy()
    best_b = b
    
    epochs_run = 0
    stop_reason = "max_epochs"
    
    max_ep = 1 if dry_run else n_epochs
    
    for epoch in range(1, max_ep + 1):
        epochs_run = epoch
        
        # ── Compute gradients & update ─────────────────────────────────────
        if optimizer.name == "sgd":
            # Mini-batch SGD
            idx = rng.permutation(n)
            for start in range(0, n, batch_size):
                batch_idx = idx[start:start + batch_size]
                X_b = X_train[batch_idx]
                y_b = y_train[batch_idx]
                nb = len(batch_idx)
                
                # Forward pass (on batch)
                z_b = X_b @ w + b
                grad_z = loss_fn.grad(z_b, y_b)
                grad_w = X_b.T @ grad_z / nb
                grad_b = np.mean(grad_z)
                
                # Add L2 gradient
                if regularizer is not None and isinstance(regularizer, L2Regularizer):
                    grad_w += regularizer.grad(w)
                
                # Optimizer step
                w, b = optimizer.step(w, b, grad_w, grad_b)
                
                # Apply proximal operator (L1)
                if regularizer is not None and isinstance(regularizer, L1Regularizer):
                    eta = optimizer.step_size.lr if hasattr(optimizer, 'step_size') else 0.01
                    w = regularizer.prox(w, eta)
        
        else:
            # Full-batch (GD, NAG, Newton, L-BFGS)
            z = X_train @ w + b
            grad_z = loss_fn.grad(z, y_train)
            grad_w = X_train.T @ grad_z / n
            grad_b = np.mean(grad_z)
            
            # Add L2 gradient
            if regularizer is not None and isinstance(regularizer, L2Regularizer):
                grad_w += regularizer.grad(w)
            
            # Compute Hessian if needed (Newton)
            kwargs = {}
            if getattr(optimizer, 'requires_hessian', False):
                diag_H = loss_fn.hessian(z, y_train)
                X_tilde = np.c_[X_train, np.ones(n)]
                H_joint = X_tilde.T @ (diag_H[:, None] * X_tilde) / n
                
                # Add L2 Hessian for weights (not bias)
                if regularizer is not None and isinstance(regularizer, L2Regularizer):
                    reg_diag = regularizer.hessian(w)
                    if reg_diag.ndim == 1:
                        H_joint[:-1, :-1] += np.diag(reg_diag)
                    else:
                        H_joint[:-1, :-1] += reg_diag
                
                kwargs['hess_joint'] = H_joint
            
            w, b = optimizer.step(w, b, grad_w, grad_b, **kwargs)
            
            # Apply proximal operator (L1)
            if regularizer is not None and isinstance(regularizer, L1Regularizer):
                eta = optimizer.step_size.lr if hasattr(optimizer, 'step_size') else 0.01
                w = regularizer.prox(w, eta)
        
        # ── Compute epoch metrics (on full train/val sets) ─────────────────
        z_train = X_train @ w + b
        train_loss = loss_fn.value(z_train, y_train)
        if regularizer is not None:
            train_loss += regularizer.penalty(w)
        
        z_val = X_val @ w + b
        val_loss = loss_fn.value(z_val, y_val)
        if regularizer is not None:
            val_loss += regularizer.penalty(w)
        
        val_auc = auroc(y_val, sigmoid(z_val))
        
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auroc"].append(val_auc)
        history["wall_time"].append(time.time() - t0)
        
        if verbose_every and epoch % verbose_every == 0:
            print(f"epoch {epoch:4d}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_auroc={val_auc:.4f}")
        
        # ── Level-1 stopping: objective tolerance ──────────────────────────
        if prev_loss is not None and abs(train_loss - prev_loss) < tol_obj:
            tol_counter += 1
            if tol_counter >= patience_inner:
                if verbose_every:
                    print(f"--> Stopping early at epoch {epoch}: "
                          f"objective delta < {tol_obj} for {patience_inner} epochs")
                stop_reason = "tol_obj"
                break
        else:
            tol_counter = 0
        prev_loss = train_loss
        
        # ── Level-2 stopping: val AUROC patience ───────────────────────────
        if val_auc > best_val_auroc:
            best_val_auroc = val_auc
            best_w = w.copy()
            best_b = b
            no_improve_outer = 0
        else:
            no_improve_outer += 1
            if no_improve_outer >= patience_outer:
                if verbose_every:
                    print(f"--> Stopping early at epoch {epoch}: "
                          f"val AUROC no improvement for {patience_outer} epochs")
                stop_reason = "patience_outer"
                break
    
    # Restore best parameters if stopped early
    if stop_reason == "patience_outer":
        w = best_w
        b = best_b
    
    return {
        "w": w,
        "b": b,
        "history": history,
        "epochs_run": epochs_run,
        "stop_reason": stop_reason,
        "best_val_auroc": best_val_auroc,
    }


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
    momentum = 0.9
    
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
        m = cfg.get("lbfgs", {}).get("m", 10)
        return LBFGS(step_size, m=m)
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
    
    result = train_logreg(
        X_train, y_train, X_val, y_val,
        loss_fn, optimizer, regularizer,
        n_epochs=cfg["max_epochs"],
        batch_size=batch_size,
        seed=cfg["seed"],
        verbose_every=0,
        tol_obj=cfg["tol_obj"],
        patience_inner=cfg["patience_inner"],
        patience_outer=cfg["patience_outer"],
        dry_run=dry_run,
    )
    
    # Compute final metrics on train and val sets
    w, b = result["w"], result["b"]
    z_train_final = X_train @ w + b
    z_val_final = X_val @ w + b
    
    train_metrics = compute_metrics(y_train, sigmoid(z_train_final))
    val_metrics = compute_metrics(y_val, sigmoid(z_val_final))
    
    return {
        "run_id":              run_id,
        "epochs_run":          result["epochs_run"],
        "stopped_early":       result["stop_reason"] != "max_epochs",
        "stop_reason":         result["stop_reason"],
        # ── Timing ────────────────────────────────────────────────────────
        "wall_time_per_epoch": result["history"]["wall_time"],
        "total_wall_time":     result["history"]["wall_time"][-1] if result["history"]["wall_time"] else 0.0,
        # ── Curves ────────────────────────────────────────────────────────
        "train_loss_curve":    result["history"]["train_loss"],
        "val_loss_curve":      result["history"]["val_loss"],
        "val_auroc_curve":     result["history"]["val_auroc"],
        "final_metrics": {
            "train": train_metrics,
            "val":   val_metrics,
        },
        "best_val_auroc": result["best_val_auroc"],
    }


# ─── Sweep builder ───────────────────────────────────────────────────────────

def build_sweep(cfg: dict, X_train: np.ndarray, y_train: np.ndarray, R: float):
    """Generate all (config_dict, hyperparams_dict) combinations to sweep."""
    loss_name    = cfg["loss"]
    opt_names    = cfg["optimizers"]
    reg_types    = cfg["regularizations"]
    lambdas      = cfg["regularization"]["lambda"]
    bt_cfg       = cfg["step_size"]["backtracking"]
    c_gd_nag     = cfg["step_size"]["fixed"]["c_gd_nag"]
    c_sgd_vals   = cfg["step_size"]["fixed"]["c_sgd"]
    decay_rates  = cfg["step_size"]["fixed"]["sgd_decay_rate"]
    batch_sizes  = cfg["sgd"]["batch_sizes"]
    
    focal_pairs  = cfg.get("focal", {}).get("gamma_alpha_pairs", [[2.0, 0.5]])
    c_scales     = cfg.get("weighted_bce", {}).get("c_scale", [1.0])
    
    step_types   = ["fixed", "backtracking"]
    
    for opt_name, reg_type in itertools.product(opt_names, reg_types):
        if opt_name in ("newton", "lbfgs") and reg_type == "l1":
            continue
        if opt_name == "newton" and loss_name == "focal":
            continue
        
        lambda_list = lambdas if reg_type != "none" else [0.0]
        
        for lambda_reg, step_type in itertools.product(lambda_list, step_types):
            if opt_name == "sgd":
                c_vals   = c_sgd_vals if step_type == "fixed" else [None]
                dr_vals  = decay_rates if step_type == "fixed" else [0.0]
                bs_vals  = batch_sizes
            else:
                c_vals   = c_gd_nag if step_type == "fixed" else [None]
                dr_vals  = [0.0]
                bs_vals  = [32]
            
            if loss_name == "focal":
                loss_hp_list = [{"gamma": g, "alpha": a} for g, a in focal_pairs]
            elif loss_name == "weighted_bce":
                loss_hp_list = [{"c_scale": cs} for cs in c_scales]
            else:
                loss_hp_list = [{}]
            
            for loss_hp in loss_hp_list:
                w_pos = loss_hp.get("c_scale", 1.0) * R if loss_name == "weighted_bce" else 1.0
                L = lipschitz_constant(
                    X_train,
                    reg_type=reg_type,
                    lambda_reg=lambda_reg,
                    w_pos=w_pos,
                )
                
                for c_val, dr, bs in itertools.product(c_vals, dr_vals, bs_vals):
                    hp = {"lambda": lambda_reg, "batch_size": bs, **loss_hp}
                    if step_type == "fixed":
                        L_eff = 1.0 if opt_name == "newton" else L
                        hp["c"] = c_val
                        hp["decay_rate"] = dr
                        hp["L"] = round(L, 6)
                        hp["lr"] = round(c_val / L_eff, 8)
                    else:
                        hp["decay_rate"] = dr
                        hp["L"] = round(L, 6)
                    
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
                        "loss_name":  loss_name,
                        "lbfgs_maxiter": None,
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
    """Convert the ``runs:`` list in explicit-mode YAML into run dicts."""
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
        
        gamma   = float(spec.get("gamma",   2.0))
        alpha   = float(spec.get("alpha",   0.5))
        c_scale = float(spec.get("c_scale", 1.0))
        
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
            "total_wall_time":    result["total_wall_time"],
            "mean_epoch_time":    np.mean(np.diff([0.0] + result["wall_time_per_epoch"])) if result["wall_time_per_epoch"] else 0.0,
            "best_val_auroc":     result["best_val_auroc"],
            "val_f1":             result["final_metrics"]["val"].get("f1", ""),
            "val_auprc":          result["final_metrics"]["val"].get("auprc", ""),
        })
    
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
