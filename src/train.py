"""
train.py
=======
Shared training loop for logistic regression z = Xw + b.

Modified to provide a robust `ctx` matching the exact mathematical
requirements of optimizers (Armijo for F, Parabol for f, Proximal ops, etc.).
"""

import time
import numpy as np

from metrics import compute_metrics, logits_to_proba
from optimizers import FULL_BATCH_OPTIMIZERS, NO_L1_OPTIMIZERS


def _sigmoid(z):
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def train_logreg(X_train, y_train, X_val, y_val,
                 loss_fn, optimizer, regularizer,
                 n_epochs: int = 100, batch_size: int = 256,
                 seed: int = 42, verbose_every: int = 5, loss_epsilon: float = 0.0,
                 patience: int = 3):
    
    opt_name = optimizer.name
    is_full_batch = opt_name in FULL_BATCH_OPTIMIZERS
    smooth = regularizer.is_smooth

    if opt_name in NO_L1_OPTIMIZERS and not smooth:
        raise ValueError(
            f"Optimizer '{opt_name}' is not supported with non-smooth L1 "
            f"regularizer (math invalid). Use GD / NAG / SGD with proximal."
        )

    rng = np.random.RandomState(seed)
    n, d = X_train.shape
    w = np.zeros(d)
    b = 0.0
    optimizer.reset(w.shape, np.shape(b))

    # --- Callables for Backtracking / Newton ---
    def f_val(ww, bb):
        z = X_train @ ww + bb
        return float(loss_fn.value(z, y_train))
        
    def F_val(ww, bb):
        return f_val(ww, bb) + float(regularizer.penalty(ww))

    def grad_w(ww, bb):
        z = X_train @ ww + bb
        gz = loss_fn.grad(z, y_train)
        return X_train.T @ gz / n

    def grad_b(ww, bb):
        z = X_train @ ww + bb
        gz = loss_fn.grad(z, y_train)
        return float(np.mean(gz))

    def hessian_wb(ww, bb):
        z = X_train @ ww + bb
        p = _sigmoid(z)
        weights = p * (1 - p)
        
        # Block Hessian matrix
        # H_ww: (d, d), H_wb: (d, 1), H_bw: (1, d), H_bb: (1, 1)
        H_ww = (X_train * weights[:, None]).T @ X_train / n
        H_wb = X_train.T @ weights / n
        H_bw = H_wb.T
        H_bb = np.sum(weights) / n
        
        # Combine blocks into full Hessian of size (d+1, d+1)
        H_full = np.zeros((d + 1, d + 1))
        H_full[:d, :d] = H_ww
        H_full[:d, d] = H_wb
        H_full[d, :d] = H_bw
        H_full[d, d] = H_bb
        
        if regularizer.name == "l2":
            # Apply L2 penalty only to the w block, bias is generally not regularized
            H_full[:d, :d] += regularizer.lam * np.eye(d)
        
        gw = grad_w(ww, bb)
        if regularizer.name == "l2":
            gw += regularizer.grad(ww)
            
        gb = grad_b(ww, bb)
        
        # Combine full gradient
        g_full = np.append(gw, gb)
        
        return H_full, g_full

    # ----- history buffers -----
    hist = {
        "iter": [], "epoch": [],
        "train_loss": [], "val_loss": [],
        "grad_norm": [], "wall_time": [],
        "val_auprc": [], "val_f1_minority": [], "val_accuracy": [],
        "val_auroc": [],
    }

    best_val_auprc = -np.inf
    best_w, best_b, best_epoch = w.copy(), b, -1

    t0 = time.time()
    global_iter = 0
    
    smoothed_loss = None
    wait_count = 0
    early_stopped = False

    for epoch in range(n_epochs):
        if is_full_batch:
            batches = [(np.arange(n),)]
        else:
            perm = rng.permutation(n)
            batches = [perm[start:start + batch_size]
                       for start in range(0, n, batch_size)]

        for batch_idx in batches:
            xb, yb = X_train[batch_idx], y_train[batch_idx]
            m = len(batch_idx)

            if not is_full_batch:
                z = xb @ w + b
                grad_z = loss_fn.grad(z, yb)
                gw_batch = xb.T @ grad_z / m
                gb_batch = float(np.mean(grad_z))
            else:
                gw_batch = None
                gb_batch = None

            ctx = {
                "w": w, "b": b,
                "smooth": smooth,
                "lam": getattr(regularizer, "lam", 0.0),
                "f_val": f_val,
                "F_val": F_val,
                "grad_w": grad_w,
                "grad_b": grad_b,
                "grad_w_batch": gw_batch,
                "grad_b_batch": gb_batch,
                "hessian_wb": hessian_wb if smooth else None,
                "reg_grad": regularizer.grad if smooth else None,
                "prox": regularizer.prox
            }
            
            w, b = optimizer.step(ctx)
            global_iter += 1

        # end-of-epoch
        z_val = X_val @ w + b
        val_loss = float(loss_fn.value(z_val, y_val) + regularizer.penalty(w))
        p_val = logits_to_proba(z_val, loss_fn.name)
        m_val = compute_metrics(y_val, p_val)

        # compute train loss for this epoch
        ep_train_loss = F_val(w, b)

        hist["iter"].append(global_iter)
        hist["epoch"].append(epoch)
        hist["train_loss"].append(ep_train_loss)
        hist["val_loss"].append(val_loss)
        
        if is_full_batch:
            if smooth:
                gn = float(np.linalg.norm(grad_w(w, b) + regularizer.grad(w)))
            else:
                gn = float(np.linalg.norm(grad_w(w, b)))
        else:
            gn = np.nan
            
        hist["grad_norm"].append(gn)
        hist["wall_time"].append(time.time() - t0)
        hist["val_auprc"].append(m_val["auprc"])
        hist["val_f1_minority"].append(m_val["f1_minority"])
        hist["val_accuracy"].append(m_val["accuracy"])
        hist["val_auroc"].append(m_val["auroc"])

        if m_val["auprc"] > best_val_auprc:
            best_val_auprc = m_val["auprc"]
            best_w, best_b, best_epoch = w.copy(), b, epoch

        if verbose_every and (epoch % verbose_every == 0 or epoch == n_epochs - 1):
            print(f"epoch {epoch:4d}  "
                  f"train_loss={hist['train_loss'][-1]:.4f}  "
                  f"val_loss={val_loss:.4f}  "
                  f"val_auprc={m_val['auprc']:.4f}  "
                  f"val_f1={m_val['f1_minority']:.4f}  "
                  f"val_acc={m_val['accuracy']:.4f}  "
                  f"t={time.time() - t0:.1f}s")
                  
        # Check early stopping at the end of epoch
        current_train_loss = F_val(w, b)
        if loss_epsilon > 0.0:
            if smoothed_loss is None:
                smoothed_loss = current_train_loss
            else:
                prev_smoothed = smoothed_loss
                smoothed_loss = 0.9 * smoothed_loss + 0.1 * current_train_loss
                delta = abs(prev_smoothed - smoothed_loss)
                
                if delta < loss_epsilon:
                    wait_count += 1
                    if wait_count >= patience:
                        print(f"Early stopping at epoch {epoch}: Smoothed loss change ({delta:.6e}) < epsilon ({loss_epsilon}) for {patience} consecutive epochs.")
                        if not (verbose_every and (epoch % verbose_every == 0 or epoch == n_epochs - 1)):
                            print(f"epoch {epoch:4d}  "
                                  f"train_loss={hist['train_loss'][-1]:.4f}  "
                                  f"val_loss={val_loss:.4f}  "
                                  f"val_auprc={m_val['auprc']:.4f}  "
                                  f"val_f1={m_val['f1_minority']:.4f}  "
                                  f"val_acc={m_val['accuracy']:.4f}  "
                                  f"t={time.time() - t0:.1f}s")
                        early_stopped = True
                        break
                else:
                    wait_count = 0

    for k in hist:
        hist[k] = np.asarray(hist[k], dtype=np.float64)

    return {
        "w": w, "b": b,
        "best_w": best_w, "best_b": best_b,
        "best_epoch": int(best_epoch),
        "best_val_auprc": float(best_val_auprc),
        "history": hist,
    }

def predict_logits(X, w, b):
    return X @ w + b

def hessian_eigs_logreg(X, w, b, lam_l2: float = 0.0, sample_size: int = None, seed: int = 0):
    if sample_size is not None and sample_size < X.shape[0]:
        rng = np.random.RandomState(seed)
        idx = rng.choice(X.shape[0], sample_size, replace=False)
        X = X[idx]

    z = X @ w + b
    p = _sigmoid(z)
    weights = p * (1 - p)
    H = (X * weights[:, None]).T @ X / X.shape[0]
    H += lam_l2 * np.eye(X.shape[1])
    eigvals = np.linalg.eigvalsh(H)
    return np.sort(eigvals)
