"""
train.py
========
SHARED module. This is the one training loop everyone calls — it wires
together a loss (losses.py), an optimizer (optimizers.py), and a
regularizer (regularizers.py) around a plain logistic regression model
(z = X w + b). Keeping this generic and shared means the only thing
that differs between people's experiments is the objects they pass in,
not the training mechanics -- which is what makes the comparison fair.

If you later add the optional MLP (Stage 4), that needs its own
forward/backward and won't reuse this file directly -- treat it as a
separate, clearly-labeled extension, not a modification of this one.
"""

import time
import numpy as np


def train_logreg(X_train, y_train, X_val, y_val,
                  loss_fn, optimizer, regularizer,
                  n_epochs: int = 100, batch_size: int = 256,
                  seed: int = 0, verbose_every: int = 10,
                  tol_grad: float = None,
                  tol_obj: float = None,
                  early_stopping_patience: int = None):
    """Trains z = Xw + b via mini-batch iterations using the given
    loss/optimizer/regularizer. Set batch_size = len(X_train) to get
    full-batch GD behavior for the same code path.

    Returns: dict with trained (w, b) and a history log for plotting.
    """
    rng = np.random.RandomState(seed)
    n, d = X_train.shape
    w = np.zeros(d)
    b = 0.0
    optimizer.reset(w.shape)

    history = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "grad_norm": [], "wall_time": [],
    }
    t0 = time.time()
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(n_epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch_idx = perm[start:start + batch_size]
            xb, yb = X_train[batch_idx], y_train[batch_idx]

            w_eval, b_eval = optimizer.lookahead(w, b) if hasattr(optimizer, 'lookahead') else (w, b)
            z = xb @ w_eval + b_eval
            grad_z = loss_fn.grad(z, yb)  # shape (batch,)
            grad_w = xb.T @ grad_z / len(batch_idx) + regularizer.grad(w_eval)
            grad_b = np.mean(grad_z)

            kwargs = {}
            if getattr(optimizer, 'requires_hessian', False):
                try:
                    diag_H_z = loss_fn.hessian(z, yb)
                except NotImplementedError as e:
                    raise NotImplementedError(
                        f"Optimizer '{getattr(optimizer, 'name', 'newton')}' requires exact Hessian, "
                        f"but loss '{getattr(loss_fn, 'name', 'loss')}' does not support it."
                    ) from e

                X_tilde = np.c_[xb, np.ones(len(xb))]
                H_joint = X_tilde.T @ (diag_H_z[:, None] * X_tilde) / len(batch_idx)
                if hasattr(regularizer, 'hessian'):
                    reg_diag = regularizer.hessian(w_eval)
                    if reg_diag.ndim == 1:
                        H_joint[:-1, :-1] += np.diag(reg_diag)
                    else:
                        H_joint[:-1, :-1] += reg_diag
                kwargs['hess_joint'] = H_joint
                
            if getattr(optimizer, 'requires_obj_fn', False):
                def obj_fn(w_eval_fn, b_eval_fn):
                    z_eval = xb @ w_eval_fn + b_eval_fn
                    return loss_fn.value(z_eval, yb) + regularizer.penalty(w_eval_fn)
                kwargs['obj_fn'] = obj_fn
                
            if getattr(optimizer, 'requires_prox_fn', False):
                kwargs['prox_fn'] = regularizer.prox

            w, b = optimizer.step(w, b, grad_w, grad_b, **kwargs)
            
            if not getattr(optimizer, 'handles_prox', False):
                w = regularizer.prox(w, getattr(optimizer, 'lr', 1e-2))

        # end-of-epoch logging (on full sets, for clean curves)
        z_train_full = X_train @ w + b
        z_val_full = X_val @ w + b
        train_loss = loss_fn.value(z_train_full, y_train) + regularizer.penalty(w)
        val_loss = loss_fn.value(z_val_full, y_val) + regularizer.penalty(w)
        grad_full = X_train.T @ loss_fn.grad(z_train_full, y_train) / n + regularizer.grad(w)
        grad_norm = float(np.linalg.norm(grad_full))

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["grad_norm"].append(grad_norm)
        history["wall_time"].append(time.time() - t0)

        if verbose_every and epoch % verbose_every == 0:
            print(f"epoch {epoch:4d}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  grad_norm={grad_norm:.4f}")

        # --- Dynamic Stopping Criteria ---
        # 1. Gradient Norm Stopping
        if tol_grad is not None and grad_norm < tol_grad:
            if verbose_every:
                print(f"--> Stopping early at epoch {epoch}: grad_norm ({grad_norm:.6f}) < tol_grad ({tol_grad})")
            break

        # 2. Objective Delta Stopping
        if tol_obj is not None and len(history["train_loss"]) > 1:
            delta_obj = abs(history["train_loss"][-2] - train_loss)
            if delta_obj < tol_obj:
                if verbose_every:
                    print(f"--> Stopping early at epoch {epoch}: objective delta ({delta_obj:.6f}) < tol_obj ({tol_obj})")
                break

        # 3. Validation Early Stopping (Patience)
        if early_stopping_patience is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    if verbose_every:
                        print(f"--> Early stopping at epoch {epoch}: val_loss has not improved for {early_stopping_patience} epochs.")
                    break

    return {"w": w, "b": b, "history": history}


def predict_logits(X, w, b):
    return X @ w + b


def hessian_eigs_logreg(X, y, w, b, loss_fn, lam_l2: float = 0.0, sample_size: int = None,
                         seed: int = 0):
    """Computes the Hessian eigenvalue spectrum of (unregularized or
    L2-regularized) logistic regression at the current w, for the
    'conditioning' plot referenced in the group discussion.

    H = X^T diag(H_z) X / n + lam_l2 * I

    sample_size: optionally subsample rows for speed on large n.
    """
    if sample_size is not None and sample_size < X.shape[0]:
        rng = np.random.RandomState(seed)
        idx = rng.choice(X.shape[0], sample_size, replace=False)
        X = X[idx]
        y = y[idx]

    z = X @ w + b
    
    diag_H = loss_fn.hessian(z, y)
    H = (X * diag_H[:, None]).T @ X / X.shape[0]
    H += lam_l2 * np.eye(X.shape[1])
    eigvals = np.linalg.eigvalsh(H)
    return np.sort(eigvals)
