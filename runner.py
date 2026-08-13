"""
runner.py - Main experiment runner with inline training loop.

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
    Level 1: ||∇F(w)||₂ ≤ tol_grad · ||∇F(w0)||₂ for patience_inner epochs,
             with train_loss non-increasing (gradient-norm stationarity,
             relative to the initial gradient so it is scale-free)
    Level 2: val AUROC no improvement for patience_outer epochs
"""

import argparse
import time
import json
import csv
import itertools
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from src.utils import (load_data, set_seed, compute_imbalance_ratio, get_logger,
                       sigmoid, hessian_joint, sanitize_json)
from src.metrics import compute_metrics, auroc
from src.step_sizes import FixedLR, ArmijoLineSearch, lipschitz_constant
from src.regularizers import L1Regularizer, L2Regularizer
from src.losses import BCELoss, WeightedBCELoss, SquaredHingeLoss, FocalLoss
from src.optimizers import GradientDescent, SGD, NAG, Newton, LBFGS

logger = get_logger("runner")

RESULTS_DIR = Path("results")

# Inner-loop convergence tolerance for optimizers that run several steps per
# epoch (L-BFGS with lbfgs_maxiter > 1): stop when the step makes no progress.
LBFGS_STEP_TOL = 1e-10


class _BatchObjective:
    """Armijo objective evaluated on the current mini-batch.

    Reused across batches: the (X_b, y_b) pair is swapped in before each
    optimizer step instead of redefining a closure. That matters for
    batch_size=1 sweeps, which iterate ~n times per epoch.
    """

    def __init__(self, loss_fn, regularizer):
        self.loss_fn = loss_fn
        self.regularizer = regularizer
        self.X_b = None
        self.y_b = None

    def __call__(self, w_, b_):
        zz = self.X_b @ w_ + b_
        val = self.loss_fn.value(zz, self.y_b)
        if self.regularizer is not None:
            val += self.regularizer.penalty(w_)
        return float(val)


# ─── Training loop (v1-style, inline) ────────────────────────────────────────
def train_logreg(
    X_train, y_train, X_val, y_val,
    loss_fn, optimizer, regularizer,
    n_epochs: int = 100,
    batch_size: int = 256,
    seed: int = 42,
    verbose_every: int = 10,
    tol_grad: float = 1e-4,
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
            - Level-1 stopping: ||∇F(w)||₂ ≤ tol_grad·||∇F(w0)||₂ for
              patience_inner consecutive epochs (train_loss non-increasing)
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
    
    # Level-1: gradient-norm stationarity, relative to the initial gradient
    # so the threshold is scale-free w.r.t. loss weighting and feature scale.
    grad_z0 = loss_fn.grad(X_train @ w + b, y_train)
    grad_w0 = X_train.T @ grad_z0 / n
    grad_b0 = float(np.mean(grad_z0))
    if regularizer is not None and isinstance(regularizer, L2Regularizer):
        grad_w0 += regularizer.grad(w)
    grad_norm0 = float(np.linalg.norm(np.concatenate([grad_w0, [grad_b0]])))
    if grad_norm0 < 1e-12:
        grad_norm0 = 1.0
    grad_norm = float("inf")
    prev_loss = None
    tol_counter = 0
    
    # Level-2: validation AUROC patience
    best_val_auroc = 0.0
    no_improve_outer = 0
    best_w = w.copy()
    best_b = b
    best_epoch = 0
    
    epochs_run = 0
    stop_reason = "max_epochs"
    
    max_ep = 1 if dry_run else n_epochs
    
    for epoch in range(1, max_ep + 1):
        epochs_run = epoch
        
        # ── Compute gradients & update ─────────────────────────────────────
        if optimizer.name == "sgd":
            # Mini-batch SGD
            # Armijo objective evaluated on the current batch; created once
            # and reused across batches (avoids a new closure per step).
            batch_obj = _BatchObjective(loss_fn, regularizer)
            idx = rng.permutation(n)
            for start in range(0, n, batch_size):
                batch_idx = idx[start:start + batch_size]
                X_b = X_train[batch_idx]
                y_b = y_train[batch_idx]
                nb = len(batch_idx)
                batch_obj.X_b = X_b
                batch_obj.y_b = y_b

                # Forward pass at the optimizer's evaluation point
                # (identity for GD/SGD; Nesterov lookahead for NAG).
                y_w, y_b_pt = optimizer.lookahead(w, b)
                z_b = X_b @ y_w + y_b_pt
                grad_z = loss_fn.grad(z_b, y_b)
                grad_w = X_b.T @ grad_z / nb
                grad_b = np.mean(grad_z)

                # Add L2 gradient (evaluated at the lookahead point too)
                if regularizer is not None and isinstance(regularizer, L2Regularizer):
                    grad_w += regularizer.grad(y_w)

                # Optimizer step (includes prox for L1)
                w, b = optimizer.step(w, b, grad_w, grad_b, obj_fn=batch_obj)
        
        else:
            # Full-batch (GD, NAG, Newton, L-BFGS)
            def full_obj(w_, b_):
                zz = X_train @ w_ + b_
                val = loss_fn.value(zz, y_train)
                if regularizer is not None:
                    val += regularizer.penalty(w_)
                return float(val)

            # L-BFGS may take several quasi-Newton steps per epoch
            # (lbfgs_maxiter); everyone else takes exactly one.
            inner_iters = max(1, int(getattr(optimizer, "maxiter", 1)))

            # Cache the [X, 1] design matrix for Newton's Hessian (X fixed,
            # so the O(n·d) stacked matrix only needs building once).
            X_tilde = None
            if getattr(optimizer, "requires_hessian", False):
                X_tilde = np.c_[X_train, np.ones(n)]

            for _ in range(inner_iters):
                # Forward pass at the optimizer's evaluation point
                # (identity for GD/SGD/Newton/L-BFGS; Nesterov lookahead for NAG).
                y_w, y_b_pt = optimizer.lookahead(w, b)
                z = X_train @ y_w + y_b_pt
                grad_z = loss_fn.grad(z, y_train)
                grad_w = X_train.T @ grad_z / n
                grad_b = np.mean(grad_z)

                # Add L2 gradient (evaluated at the lookahead point too)
                if regularizer is not None and isinstance(regularizer, L2Regularizer):
                    grad_w += regularizer.grad(y_w)

                kwargs = {"obj_fn": full_obj}
                if X_tilde is not None:
                    diag_H = loss_fn.hessian(z, y_train)
                    reg_diag = None
                    if regularizer is not None and isinstance(regularizer, L2Regularizer):
                        reg_diag = regularizer.hessian(y_w)
                    H_joint = hessian_joint(X_train, diag_H, reg_diag, X_tilde=X_tilde)
                    kwargs['hess_joint'] = H_joint

                prev_w, prev_b = w, b
                w, b = optimizer.step(w, b, grad_w, grad_b, **kwargs)

                # Inner-loop convergence for multi-step optimizers: stop when
                # the step makes no numerical progress (L-BFGS wall done).
                if inner_iters > 1:
                    max_delta = max(float(np.max(np.abs(w - prev_w))), abs(b - prev_b))
                    if max_delta < LBFGS_STEP_TOL:
                        break
        
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
        
        # ── Level-1 stopping: gradient-norm stationarity ───────────────────
        # Stationarity gradient at the (new) iterate, full-batch. The epoch
        # metrics above already computed z_train, so this is one extra pass.
        grad_z = loss_fn.grad(z_train, y_train)
        grad_w_st = X_train.T @ grad_z / n
        grad_b_st = float(np.mean(grad_z))
        if regularizer is not None and isinstance(regularizer, L2Regularizer):
            grad_w_st += regularizer.grad(w)
        grad_norm = float(np.linalg.norm(np.concatenate([grad_w_st, [grad_b_st]])))

        # Guard against counting a diverging/spiky epoch (NAG overshoot,
        # SGD noise): only count consecutive epochs where the train loss is
        # non-increasing and the gradient is below the scale-free threshold.
        if (prev_loss is not None and train_loss <= prev_loss
                and grad_norm <= tol_grad * grad_norm0):
            tol_counter += 1
            if tol_counter >= patience_inner:
                if verbose_every:
                    print(f"--> Stopping early at epoch {epoch}: "
                          f"||grad||={grad_norm:.3e} ≤ "
                          f"{tol_grad}·||grad0|| for {patience_inner} epochs")
                stop_reason = "tol"
                break
        else:
            tol_counter = 0
        prev_loss = train_loss
        
        # ── Level-2 stopping: val AUROC patience ───────────────────────────
        if val_auc > best_val_auroc:
            best_val_auroc = val_auc
            best_w = w.copy()
            best_b = b
            best_epoch = epoch
            no_improve_outer = 0
        else:
            no_improve_outer += 1
            if no_improve_outer >= patience_outer:
                if verbose_every:
                    print(f"--> Stopping early at epoch {epoch}: "
                          f"val AUROC no improvement for {patience_outer} epochs")
                stop_reason = "patience_outer"
                break
    
    # Always restore the best validation-set parameters so the returned model
    # is defined consistently across all stop reasons (early stop or budget
    # exhausted), not just for early-stopped runs.
    final_epoch_w = w.copy()
    final_epoch_b = b
    if best_val_auroc > 0.0:
        w = best_w
        b = best_b
    
    return {
        "w": w,
        "b": b,
        "history": history,
        "epochs_run": epochs_run,
        "stop_reason": stop_reason,
        "best_val_auroc": best_val_auroc,
        "best_epoch": best_epoch,
        "final_grad_norm": grad_norm,
        "final_epoch_w": final_epoch_w,
        "final_epoch_b": final_epoch_b,
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
                    cfg: dict, lbfgs_maxiter: int | None = None,
                    momentum: float | None = None):
    use_proximal = (reg_type == "l1")
    if momentum is None:
        momentum = cfg.get("nag", {}).get("momentum", 0.9)

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
        if lbfgs_maxiter is None:
            lbfgs_maxiter = cfg.get("lbfgs", {}).get("maxiter", 1)
        return LBFGS(step_size, m=m, maxiter=lbfgs_maxiter)
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

    # Training/optimizer settings actually used - lets evaluate_best.py
    # faithfully reproduce the run instead of re-guessing hardcoded values.
    training_cfg = {
        "n_epochs":        cfg["max_epochs"],
        "tol_grad":        cfg.get("tol_grad", 1e-4),
        "patience_inner":  cfg["patience_inner"],
        "patience_outer":  cfg["patience_outer"],
        "seed":            cfg["seed"],
        "batch_size":      batch_size,
    }
    if isinstance(optimizer, NAG):
        training_cfg["nag_momentum"] = optimizer.momentum
    if isinstance(optimizer, Newton):
        training_cfg["newton_epsilon_damp"] = optimizer.epsilon_damp
    if isinstance(optimizer, LBFGS):
        training_cfg["lbfgs_m"] = optimizer.m
        training_cfg["lbfgs_maxiter"] = optimizer.maxiter
    ss = getattr(optimizer, "step_size", None)
    if ss is not None and hasattr(ss, "search"):
        training_cfg["backtracking"] = {
            "alpha_init": ss.alpha_init,
            "beta":       ss.beta,
            "c_armijo":   ss.c_armijo,
        }

    result = train_logreg(
        X_train, y_train, X_val, y_val,
        loss_fn, optimizer, regularizer,
        n_epochs=cfg["max_epochs"],
        batch_size=batch_size,
        seed=cfg["seed"],
        verbose_every=0,
        tol_grad=cfg.get("tol_grad", 1e-4),
        patience_inner=cfg["patience_inner"],
        patience_outer=cfg["patience_outer"],
        dry_run=dry_run,
    )
    
    # Final metrics at the best-val model (w, b is restored to best); also
    # report the last-epoch model explicitly for unambiguity.
    w, b = result["w"], result["b"]
    z_train_final = X_train @ w + b
    z_val_final = X_val @ w + b
    
    train_metrics = compute_metrics(y_train, sigmoid(z_train_final))
    val_metrics = compute_metrics(y_val, sigmoid(z_val_final))
    
    wf, bf = result["final_epoch_w"], result["final_epoch_b"]
    z_train_fe = X_train @ wf + bf
    z_val_fe = X_val @ wf + bf
    final_epoch_metrics = {
        "train": compute_metrics(y_train, sigmoid(z_train_fe)),
        "val":   compute_metrics(y_val, sigmoid(z_val_fe)),
    }
    
    return {
        "run_id":              run_id,
        "epochs_run":          result["epochs_run"],
        "stopped_early":       result["stop_reason"] != "max_epochs",
        "stop_reason":         result["stop_reason"],
        "best_epoch":          result["best_epoch"],
        "final_grad_norm":     result["final_grad_norm"],
        "training_cfg":        training_cfg,
        # ── Timing ────────────────────────────────────────────────────────
        "wall_time_per_epoch": result["history"]["wall_time"],
        "cumulative_time_s":   result["history"]["wall_time"],
        "total_wall_time":     result["history"]["wall_time"][-1] if result["history"]["wall_time"] else 0.0,
        # ── Curves ────────────────────────────────────────────────────────
        "train_loss_curve":    result["history"]["train_loss"],
        "val_loss_curve":      result["history"]["val_loss"],
        "val_auroc_curve":     result["history"]["val_auroc"],
        "final_metrics": {
            "train": train_metrics,
            "val":   val_metrics,
        },
        "final_epoch_metrics": final_epoch_metrics,
        "best_val_auroc": result["best_val_auroc"],
    }


# ─── Sweep builder ───────────────────────────────────────────────────────────

def build_sweep(cfg: dict, X_train: np.ndarray, y_train: np.ndarray, R: float,
                sigma_max: float | None = None):
    """Generate all (config_dict, hyperparams_dict) combinations to sweep."""
    loss_name    = cfg["loss"]
    opt_names    = cfg["optimizers"]
    reg_types    = cfg["regularizations"]
    lambdas      = cfg["regularization"]["lambda"]
    bt_cfg       = cfg["step_size"]["backtracking"]
    fixed_cfg    = cfg["step_size"]["fixed"]
    c_gd_nag     = fixed_cfg.get("c_gd_nag", [1.0])
    c_sgd_vals   = fixed_cfg.get("c_sgd", [0.01])
    decay_rates  = fixed_cfg.get("sgd_decay_rate", [0.0])
    # Newton fixed: keep c within the safe damped range (pure α=1 already
    # included); GD's c-grid up to 1.9 makes raw Newton diverge.
    c_newton     = fixed_cfg.get("c_newton", [1.0])
    # Focal loss is not 1/4-smooth, so the sigmoid Lipschitz bound does not
    # hold; use a smaller default c range.
    c_focal      = fixed_cfg.get("c_focal", [0.05, 0.1])
    batch_sizes  = cfg["sgd"]["batch_sizes"]
    
    focal_pairs  = cfg.get("focal", {}).get("gamma_alpha_pairs", [[2.0, 0.5]])
    c_scales     = cfg.get("weighted_bce", {}).get("c_scale", [1.0])
    
    step_types   = ["fixed", "backtracking"]
    
    for opt_name, reg_type in itertools.product(opt_names, reg_types):
        if opt_name in ("newton", "lbfgs") and reg_type == "l1":
            logger.warning(
                f"Sweep: skipping {opt_name} + l1 - requires a prox operator "
                f"that {opt_name} does not implement.")
            continue
        if opt_name == "newton" and loss_name == "focal":
            logger.warning(
                "Sweep: skipping newton + focal - focal Hessian is not implemented.")
            continue
        
        lambda_list = lambdas if reg_type != "none" else [0.0]
        
        for lambda_reg, step_type in itertools.product(lambda_list, step_types):
            if opt_name == "sgd":
                c_vals   = c_sgd_vals if step_type == "fixed" else [None]
                dr_vals  = decay_rates if step_type == "fixed" else [0.0]
                bs_vals  = batch_sizes
            elif step_type == "backtracking":
                c_vals   = [None]
                dr_vals  = [0.0]
                bs_vals  = [32]
            elif opt_name == "newton":
                c_vals   = c_newton
                dr_vals  = [0.0]
                bs_vals  = [32]
            elif loss_name == "focal":
                c_vals   = c_focal
                dr_vals  = [0.0]
                bs_vals  = [32]
            else:
                c_vals   = c_gd_nag
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
                    loss_name=loss_name,
                    sigma_max=sigma_max,
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
    sigma_max: float | None = None,
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

        # Enforce the same guards the sweep applies: Newton/L-BFGS cannot do
        # L1 (no prox) and Newton has no focal Hessian.
        if opt_name in ("newton", "lbfgs") and reg_type == "l1":
            logger.warning(
                f"Explicit: skipping run {spec.get('id', '')} - {opt_name} + l1 "
                "requires a prox operator that is not implemented.")
            continue
        if opt_name == "newton" and loss_name == "focal":
            logger.warning(
                f"Explicit: skipping run {spec.get('id', '')} - newton + focal "
                "Hessian is not implemented.")
            continue
        # Focal loss is not 1/4-smooth: the sigmoid Lipschitz bound the fixed
        # step relies on does not hold, so large c can make GD/NAG unstable.
        if loss_name == "focal" and step_type == "fixed" and c_val >= 0.5:
            logger.warning(
                f"Explicit: '{spec.get('id', '')}' uses focal + fixed c={c_val} "
                "- the sigmoid Lipschitz bound does not hold for focal loss; "
                "large steps may be unstable. Consider c < 0.5 or "
                "step: backtracking.")
        
        gamma   = float(spec.get("gamma",   2.0))
        alpha   = float(spec.get("alpha",   0.5))
        c_scale = float(spec.get("c_scale", 1.0))
        
        w_pos = c_scale * R if loss_name == "weighted_bce" else 1.0
        L = lipschitz_constant(
            X_train, reg_type=reg_type,
            lambda_reg=lambda_reg, w_pos=w_pos, loss_name=loss_name,
            sigma_max=sigma_max,
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


def epochs_to_target(val_auroc_curve, target: float) -> int | str:
    """First epoch (1-based) at which val AUROC reaches ``target``, else ''."""
    hit = next((i for i, v in enumerate(val_auroc_curve, 1) if v >= target), None)
    return hit if hit is not None else ""


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
    
    RESULTS_DIR.mkdir(exist_ok=True)
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    set_seed(cfg["seed"])
    data = load_data(args.data_dir)
    X_train, y_train = data["train"]
    X_val,   y_val   = data["val"]
    
    R = compute_imbalance_ratio(y_train)
    logger.info(f"Class imbalance ratio R = N_neg/N_pos = {R:.3f}")

    # σ_max depends only on X — compute once and pass it down so the SVD is
    # not repeated for every (optimizer, reg, λ, loss) combination.
    sigma_max = float(np.linalg.svd(X_train, compute_uv=False)[0])

    explicit_mode = "runs" in cfg
    group_tag     = cfg.get("group", cfg.get("loss", "exp"))
    
    if explicit_mode:
        logger.info(f"Explicit-run mode | group: {group_tag} | "
                    f"{len(cfg['runs'])} runs defined")
        sweep = list(build_explicit_runs(cfg, X_train, y_train, R, sigma_max=sigma_max))
    else:
        loss_name = cfg["loss"]
        sweep = list(build_sweep(cfg, X_train, y_train, R, sigma_max=sigma_max))
    
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
            json.dump(sanitize_json(result), f, indent=2)
        
        summary_rows.append({
            "run_id":             run_id,
            "group":              group_tag,
            "loss_fn":            run_loss_name,
            "optimizer":          run_cfg["opt_name"],
            "regularization":     run_cfg["reg_type"],
            "step_type":          run_cfg["step_type"],
            "epochs_run":         result["epochs_run"],
            "stop_reason":        result["stop_reason"],
            "best_epoch":         result["best_epoch"],
            "final_grad_norm":    result["final_grad_norm"],
            "total_wall_time":    result["total_wall_time"],
            "mean_epoch_time":    np.mean(np.diff([0.0] + result["wall_time_per_epoch"])) if result["wall_time_per_epoch"] else 0.0,
            "best_val_auroc":     result["best_val_auroc"],
            "val_f1":             result["final_metrics"]["val"].get("f1", ""),
            "val_auprc":          result["final_metrics"]["val"].get("auprc", ""),
            "val_auroc_curve":    result["val_auroc_curve"],
        })
    
    if summary_rows:
        # Epochs to reach the group's best val AUROC — a fairer per-epoch
        # comparison than "epochs_run" (which conflates method speed with
        # early-stopping luck).
        target_auroc = max(r["best_val_auroc"] for r in summary_rows)
        for r in summary_rows:
            curve = r.pop("val_auroc_curve")
            r["epochs_to_target"] = epochs_to_target(curve, target_auroc)
        csv_path = group_results_dir / f"{group_tag}_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        logger.info(f"Summary saved to {csv_path}")
    
    logger.info("Done.")


if __name__ == "__main__":
    main()
