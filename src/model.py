"""
model.py — Logistic Regression model orchestrating the training loop.

This class combines a loss function, an optimizer, and a regularizer to
train logistic regression via gradient-based methods.

Training loop structure (per run):
  Outer loop: epochs (1 to max_epochs)
    For SGD: inner loop over mini-batches (iter_batches)
    For all others: single full-batch step
    - Record train loss and val AUROC per epoch
    - Level-1 stopping: |loss(t) - loss(t-1)| ≤ tol_obj for patience_inner steps
    - Level-2 stopping: val AUROC no improvement for patience_outer epochs
      (checked at the experiment runner level, not here)

The model does NOT handle hyperparameter sweeping; that is done by runner.py.
"""

import numpy as np
from typing import Callable

from src.utils import sigmoid, iter_batches
from src.metrics import auroc as compute_auroc


class LogisticRegression:
    """
    Logistic Regression with pluggable loss, optimizer, and regularizer.

    Parameters
    ----------
    loss_fn      : loss object with .loss(), .gradient(), optionally .hessian()
    optimizer    : optimizer object with .step() and .reset()
    regularizer  : L1Regularizer | L2Regularizer | None
    reg_type     : 'l1' | 'l2' | 'none'
    max_epochs   : maximum number of training epochs
    tol_obj      : objective tolerance for within-run early stopping
    patience_inner : number of consecutive tol_obj-meeting steps to trigger stop
    batch_size   : mini-batch size for SGD (ignored for GD/NAG/Newton/LBFGS)
    seed         : random seed for mini-batch shuffling
    verbose      : print progress every `verbose` epochs (0 = silent)
    """

    def __init__(
        self,
        loss_fn,
        optimizer,
        regularizer=None,
        reg_type: str = "none",
        max_epochs: int = 1000,
        tol_obj: float = 1e-6,
        patience_inner: int = 5,
        batch_size: int = 32,
        seed: int = 42,
        verbose: int = 0,
    ):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.regularizer = regularizer
        self.reg_type = reg_type
        self.max_epochs = max_epochs
        self.tol_obj = tol_obj
        self.patience_inner = patience_inner
        self.batch_size = batch_size
        self.seed = seed
        self.verbose = verbose

        self.w_: np.ndarray | None = None  # trained weights
        self.train_loss_curve_: list[float] = []
        self.val_loss_curve_:   list[float] = []
        self.val_auroc_curve_:  list[float] = []
        self.epochs_run_: int = 0
        self.stop_reason_: str = "max_epochs"

    # ─── Internal helpers ────────────────────────────────────────────────────

    def _total_loss(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """Loss + regularization penalty (scalar)."""
        l = self.loss_fn.loss(w, X, y)
        if self.regularizer is not None:
            l += self.regularizer.penalty(w)
        return l

    def _total_grad(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Gradient of loss + regularization."""
        g = self.loss_fn.gradient(w, X, y)
        if self.regularizer is not None:
            if self.reg_type == "l2":
                g = g + self.regularizer.gradient(w)
            # L1 gradient is NOT added here; proximal operator handles it in optimizer
        return g

    def _total_hessian(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Hessian of loss + L2 penalty (only for Newton)."""
        H = self.loss_fn.hessian(w, X, y)
        if self.regularizer is not None and self.reg_type == "l2":
            H = H + self.regularizer.lambda_reg * np.eye(H.shape[0])
        return H

    def _is_sgd(self) -> bool:
        return self.optimizer.name == "sgd"

    # ─── Fit ─────────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        w_init: np.ndarray | None = None,
    ) -> "LogisticRegression":
        """
        Train the model.

        Parameters
        ----------
        X_train, y_train : training data
        X_val, y_val     : validation data (for AUROC curve logging)
        w_init           : initial weights (zeros if None)
        """
        n, d = X_train.shape
        self.w_ = np.zeros(d) if w_init is None else w_init.copy()
        self.optimizer.reset()

        self.train_loss_curve_ = []
        self.val_loss_curve_ = []
        self.val_auroc_curve_ = []
        self.stop_reason_ = "max_epochs"

        rng = np.random.default_rng(self.seed)
        tol_counter = 0
        prev_loss = None

        hessian_fn = (self._total_hessian
                      if self.optimizer.name == "newton" else None)

        for epoch in range(1, self.max_epochs + 1):
            self.epochs_run_ = epoch

            if self._is_sgd():
                # ── SGD: iterate over mini-batches ──────────────────────────
                for X_b, y_b in iter_batches(
                    X_train, y_train,
                    batch_size=self.batch_size,
                    shuffle=True,
                    rng=rng,
                ):
                    self.w_ = self.optimizer.step(
                        self.w_,
                        loss_fn=self._total_loss,
                        grad_fn=self._total_grad,
                        X=X_b,
                        y=y_b,
                        X_full=X_train,
                        y_full=y_train,
                    )
            else:
                # ── Full-batch step (GD, NAG, Newton, L-BFGS) ───────────────
                self.w_ = self.optimizer.step(
                    self.w_,
                    loss_fn=self._total_loss,
                    grad_fn=self._total_grad,
                    X=X_train,
                    y=y_train,
                    hessian_fn=hessian_fn,
                )

            # ── Record epoch-level metrics ───────────────────────────────────
            train_loss = self._total_loss(self.w_, X_train, y_train)
            self.train_loss_curve_.append(train_loss)

            if X_val is not None and y_val is not None:
                val_loss = self._total_loss(self.w_, X_val, y_val)
                val_auroc = compute_auroc(y_val, self.predict_proba(X_val))
                self.val_loss_curve_.append(val_loss)
                self.val_auroc_curve_.append(val_auroc)

            if self.verbose > 0 and epoch % self.verbose == 0:
                msg = f"Epoch {epoch:4d} | train_loss={train_loss:.6f}"
                if self.val_auroc_curve_:
                    msg += f" | val_auroc={self.val_auroc_curve_[-1]:.4f}"
                print(msg)

            # ── Level-1 early stopping: tol_obj ──────────────────────────────
            if prev_loss is not None:
                if abs(train_loss - prev_loss) <= self.tol_obj:
                    tol_counter += 1
                    if tol_counter >= self.patience_inner:
                        self.stop_reason_ = "tol_obj"
                        break
                else:
                    tol_counter = 0
            prev_loss = train_loss

        return self

    # ─── Predict ─────────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return predicted probabilities for class 1."""
        if self.w_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        return sigmoid(X @ self.w_)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return hard binary predictions."""
        return (self.predict_proba(X) >= threshold).astype(int)

    # ─── Summary ─────────────────────────────────────────────────────────────

    def training_summary(self) -> dict:
        """Return a compact summary dict for JSON serialization."""
        return {
            "epochs_run":        self.epochs_run_,
            "stop_reason":       self.stop_reason_,
            "train_loss_curve":  [round(v, 8) for v in self.train_loss_curve_],
            "val_loss_curve":    [round(v, 8) for v in self.val_loss_curve_],
            "val_auroc_curve":   [round(v, 6) for v in self.val_auroc_curve_],
            "final_train_loss":  self.train_loss_curve_[-1] if self.train_loss_curve_ else None,
            "final_val_auroc":   self.val_auroc_curve_[-1]  if self.val_auroc_curve_  else None,
        }
