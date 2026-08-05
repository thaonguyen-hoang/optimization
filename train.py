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
                  seed: int = 0, verbose_every: int = 10):
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

    for epoch in range(n_epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch_idx = perm[start:start + batch_size]
            xb, yb = X_train[batch_idx], y_train[batch_idx]

            z = xb @ w + b
            grad_z = loss_fn.grad(z, yb)  # shape (batch,)
            grad_w = xb.T @ grad_z / len(batch_idx) + regularizer.grad(w)
            grad_b = np.mean(grad_z)

            w, b = optimizer.step(w, b, grad_w, grad_b)
            w = regularizer.prox(w, optimizer.lr)  # no-op unless L1/elastic-net

        # end-of-epoch logging (on full sets, for clean curves)
        z_train_full = X_train @ w + b
        z_val_full = X_val @ w + b
        train_loss = loss_fn.value(z_train_full, y_train) + regularizer.penalty(w)
        val_loss = loss_fn.value(z_val_full, y_val) + regularizer.penalty(w)
        grad_full = X_train.T @ loss_fn.grad(z_train_full, y_train) / n
        grad_norm = float(np.linalg.norm(grad_full))

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["grad_norm"].append(grad_norm)
        history["wall_time"].append(time.time() - t0)

        if verbose_every and epoch % verbose_every == 0:
            print(f"epoch {epoch:4d}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  grad_norm={grad_norm:.4f}")

    return {"w": w, "b": b, "history": history}


def predict_logits(X, w, b):
    return X @ w + b


def hessian_eigs_logreg(X, w, b, lam_l2: float = 0.0, sample_size: int = None,
                         seed: int = 0):
    """Computes the Hessian eigenvalue spectrum of (unregularized or
    L2-regularized) logistic regression at the current w, for the
    'conditioning' plot referenced in the group discussion.

    H = X^T diag(p(1-p)) X / n + lam_l2 * I

    sample_size: optionally subsample rows for speed on large n.
    """
    if sample_size is not None and sample_size < X.shape[0]:
        rng = np.random.RandomState(seed)
        idx = rng.choice(X.shape[0], sample_size, replace=False)
        X = X[idx]

    z = X @ w + b
    p = 1.0 / (1.0 + np.exp(-z))
    weights = p * (1 - p)
    H = (X * weights[:, None]).T @ X / X.shape[0]
    H += lam_l2 * np.eye(X.shape[1])
    eigvals = np.linalg.eigvalsh(H)
    return np.sort(eigvals)
