"""Shared training loop for linear binary classifiers."""

import time

import numpy as np

from src.metrics import compute_metrics, logits_to_proba
from src.optimizers import FULL_BATCH_OPTIMIZERS, NO_L1_OPTIMIZERS


def train_logreg(
    X_train, y_train,
    X_val,   y_val,
    loss_fn,
    optimizer,
    regularizer,
    n_epochs=100,
    batch_size=256,
    seed=42,
    verbose_every=0,
    tol=1e-4,
    atol=1e-8,
    tol_stat=1e-2,
    atol_stat=1e-4,
    patience=3,
):
    """
    Train a logistic-regression model.

    Stopping has two layers, both on the optimization problem itself:

    1. Trigger (relative smoothed-loss change): EMA of the full objective
       F(w, b) with alpha = 0.1; the loss "stopped changing" when
       |delta| < tol * |smoothed| + atol for `patience` consecutive epochs.
    2. Acceptance (stationarity check): when the trigger fires, verify the
       first-order stationarity condition at the current iterate —
       ‖∇F(w,b)‖ <= max(tol_stat * ‖∇F(w0,b0)‖, atol_stat) for smooth
       objectives, or the proximal residual for L1. If the trigger fires but
       stationarity is NOT yet met, the run is given grace and continues
       until `max_epochs` (a stalled run is never silently reported as
       converged). For mini-batch SGD, the full-batch gradient is evaluated
       at each acceptance check.

    Validation is only used to track the best model for reporting; it never
    stops the run.

    Parameters
    ----------
    X_train, y_train : training features and labels
    X_val, y_val     : validation features and labels (used for best-model tracking)
    loss_fn          : Loss instance (BCELoss, WeightedBCELoss, SquaredHingeLoss)
    optimizer        : Optimizer instance (GD, NAG, Newton, SGD)
    regularizer      : Regularizer instance (NoReg, L2Reg, L1Reg)
    n_epochs         : maximum number of training epochs
    batch_size       : mini-batch size (used only by SGD; others use full batch)
    seed             : RNG seed for mini-batch shuffling
    verbose_every    : print progress every N epochs (0 = silent)
    tol              : relative tolerance for the smoothed-loss trigger
    atol             : absolute floor for the smoothed-loss trigger
    tol_stat         : relative stationarity acceptance threshold
    atol_stat        : absolute floor for stationarity acceptance
    patience         : consecutive epochs satisfying the trigger before checking

    Returns
    -------
    dict with keys: w, b, history, epochs_run, stop_reason, converged,
                    best_epoch, best_val_auprc, final_grad_norm, final_objective
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    X_val   = np.asarray(X_val,   dtype=float)
    y_val   = np.asarray(y_val,   dtype=float)

    if X_train.ndim != 2 or X_train.shape[0] != len(y_train):
        raise ValueError("Invalid training shapes")
    if X_val.shape[1] != X_train.shape[1] or X_val.shape[0] != len(y_val):
        raise ValueError("Invalid validation shapes")
    if optimizer.name in NO_L1_OPTIMIZERS and not regularizer.is_smooth:
        raise ValueError("Newton is invalid with L1 regularization")
    if n_epochs <= 0 or batch_size <= 0:
        raise ValueError("n_epochs and batch_size must be positive")

    n, d = X_train.shape
    rng  = np.random.RandomState(seed)
    w    = np.zeros(d)
    b    = 0.0
    optimizer.reset(w.shape, np.shape(b))
    full_batch = optimizer.name in FULL_BATCH_OPTIMIZERS

    # ── Helper: read the current step size from the optimizer ─────────────────
    def current_step_size():
        if hasattr(optimizer, "last_lr"):
            return float(optimizer.last_lr)
        return float(getattr(optimizer, "lr", getattr(optimizer, "initial_lr", 1.0)))

    # ── First-order stationarity measure (smooth: grad norm; L1: prox residual)
    def compute_stationarity(ww, bb):
        if regularizer.is_smooth:
            sw = grad_w(ww, bb) + regularizer.grad(ww)
            sb = grad_b(ww, bb)
            return float(np.sqrt(np.dot(sw, sw) + sb * sb))
        step  = max(current_step_size(), np.finfo(float).eps)
        mapping = (ww - regularizer.prox(ww - step * grad_w(ww, bb), step * regularizer.lam)) / step
        return float(np.linalg.norm(mapping))

    # ── Closures for loss / gradient / Hessian on the full training set ───────
    def f_val(ww, bb):
        return float(loss_fn.value(X_train @ ww + bb, y_train))

    def F_val(ww, bb):
        return f_val(ww, bb) + regularizer.penalty(ww)

    def grad_w(ww, bb):
        return X_train.T @ loss_fn.grad(X_train @ ww + bb, y_train) / n

    def grad_b(ww, bb):
        return float(np.mean(loss_fn.grad(X_train @ ww + bb, y_train)))

    def hessian_wb(ww, bb):
        diag   = loss_fn.hessian_diag(X_train @ ww + bb, y_train)
        design = np.column_stack((X_train, np.ones(n)))
        H      = design.T @ (diag[:, None] * design) / n
        if regularizer.name == "l2":
            H[:d, :d] += regularizer.lam * np.eye(d)
        g = np.r_[grad_w(ww, bb) + regularizer.grad(ww), grad_b(ww, bb)]
        return H, g

    # ── Reference stationarity at the initial point w0 = 0, b0 = 0 ────────────
    norm0 = max(compute_stationarity(w, b), np.finfo(float).eps)

    # ── History and state initialisation ─────────────────────────────────────
    hist = {key: [] for key in (
        "train_loss", "val_loss", "val_auprc", "val_auroc",
        "val_f1_minority", "grad_norm", "wall_time", "iter",
    )}
    best_score    = -np.inf
    best_w        = w.copy()
    best_b        = b
    best_epoch    = -1
    global_iter   = 0
    smoothed_loss = None
    wait_count    = 0
    stop_reason   = "max_epochs"
    started       = time.perf_counter()

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(n_epochs):

        if full_batch:
            batches = [np.arange(n)]
        else:
            perm    = rng.permutation(n)
            batches = [perm[s:s + batch_size] for s in range(0, n, batch_size)]

        for idx in batches:
            if full_batch:
                gw_batch = None
                gb_batch = None
            else:
                gz       = loss_fn.grad(X_train[idx] @ w + b, y_train[idx])
                gw_batch = X_train[idx].T @ gz / len(idx)
                gb_batch = float(np.mean(gz))

            ctx = {
                "w":           w,
                "b":           b,
                "smooth":      regularizer.is_smooth,
                "lam":         regularizer.lam,
                "f_val":       f_val,
                "F_val":       F_val,
                "grad_w":      grad_w,
                "grad_b":      grad_b,
                "grad_w_batch": gw_batch,
                "grad_b_batch": gb_batch,
                "reg_grad":    regularizer.grad if regularizer.is_smooth else None,
                "hessian_wb":  hessian_wb      if regularizer.is_smooth else None,
                "prox":        regularizer.prox,
            }
            w, b = optimizer.step(ctx)
            global_iter += 1

        # ── Per-epoch metrics ─────────────────────────────────────────────────
        train_loss  = F_val(w, b)
        val_loss    = float(loss_fn.value(X_val @ w + b, y_val) + regularizer.penalty(w))
        val_proba   = logits_to_proba(X_val @ w + b, loss_fn.name)
        val_metrics = compute_metrics(y_val, val_proba)

        if full_batch:
            stationarity = compute_stationarity(w, b)
            grad_norm    = stationarity
        else:
            stationarity = float("nan")
            grad_norm    = float("nan")

        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)
        hist["val_auprc"].append(val_metrics["auprc"])
        hist["val_auroc"].append(val_metrics["auroc"])
        hist["val_f1_minority"].append(val_metrics["f1_minority"])
        hist["grad_norm"].append(grad_norm)
        hist["wall_time"].append(time.perf_counter() - started)
        hist["iter"].append(global_iter)

        # ── Best-model tracking (reporting only, never stops early) ────────────
        score = val_metrics["auprc"]
        if np.isfinite(score) and score > best_score + 1e-12:
            best_score = score
            best_w     = w.copy()
            best_b     = float(b)
            best_epoch = epoch

        if verbose_every and (epoch % verbose_every == 0 or epoch == n_epochs - 1):
            print(f"epoch={epoch:4d}  train_loss={train_loss:.6g}  val_auprc={score:.6g}")

        # ── Trigger: relative EMA smoothed-loss change ─────────────────────────
        if tol > 0.0:
            if smoothed_loss is None:
                smoothed_loss = train_loss
            else:
                prev_smoothed = smoothed_loss
                smoothed_loss = 0.9 * smoothed_loss + 0.1 * train_loss
                delta = abs(prev_smoothed - smoothed_loss)
                if delta < tol * abs(smoothed_loss) + atol:
                    wait_count += 1
                    # ── Acceptance: verify stationarity before declaring converged
                    if wait_count >= patience:
                        stat = compute_stationarity(w, b)
                        if stat <= max(tol_stat * norm0, atol_stat):
                            stop_reason = "converged"
                            break
                        # Grace: loss stalled but gradient still large → keep going
                        # until max_epochs, then the final check decides the tag.
                        wait_count = 0
                else:
                    wait_count = 0

    # ── Final verdict at the stopping iterate ─────────────────────────────────
    final_stat = compute_stationarity(w, b)
    converged  = final_stat <= max(tol_stat * norm0, atol_stat)
    if stop_reason == "max_epochs" and converged:
        stop_reason = "converged"

    # ── Fallback: if no improving epoch was logged, use the last one ──────────
    if best_epoch < 0:
        best_w     = w.copy()
        best_b     = float(b)
        best_epoch = len(hist["train_loss"]) - 1
        best_score = hist["val_auprc"][-1]

    return {
        "w":               best_w,
        "b":               best_b,
        "history":         hist,
        "epochs_run":      len(hist["train_loss"]),
        "stop_reason":     stop_reason,
        "converged":       bool(converged),
        "best_epoch":      int(best_epoch),
        "best_val_auprc":  float(best_score),
        "final_grad_norm": final_stat,
        "final_objective": float(F_val(w, b)),
    }
