"""
Smoke test for refactored v2 code.
Tests the new logit-based interfaces for losses, optimizers, and training loop.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.step_sizes import FixedLR, ArmijoLineSearch, lipschitz_constant
from src.regularizers import L1Regularizer,L2Regularizer
from src.losses import BCELoss, WeightedBCELoss, SquaredHingeLoss, FocalLoss
from src.optimizers import GradientDescent, SGD, NAG, Newton, LBFGS
from src.metrics import compute_metrics, auprc
from src.utils import sigmoid
from runner import train_logreg, build_step_size, build_regularizer, epochs_to_target


def test_losses():
    """Test new logit-based loss interface."""
    print("Testing losses...")
    n, d = 100, 5
    X = np.random.randn(n, d)
    w = np.random.randn(d)
    b = 0.1
    y = np.random.randint(0, 2, n)
    z = X @ w + b
    
    # BCE
    loss = BCELoss()
    val = loss.value(z, y)
    grad_z = loss.grad(z, y)
    hess_z = loss.hessian(z, y)
    print(f"  BCE: value={val:.4f}, grad shape={grad_z.shape}, hess shape={hess_z.shape}")
    
    # Weighted BCE
    loss_w = WeightedBCELoss(w_pos=2.0, w_neg=1.0)
    val_w = loss_w.value(z, y)
    grad_w = loss_w.grad(z, y)
    print(f"  WeightedBCE: value={val_w:.4f}, grad shape={grad_w.shape}")


def test_regularizers():
    """Test new regularizer interface."""
    print("\nTesting regularizers...")
    d = 5
    w = np.random.randn(d)
    
    # L2
    l2 = L2Regularizer(lam=0.01)
    pen2 = l2.penalty(w)
    grad2 = l2.grad(w)
    print(f"  L2: penalty={pen2:.4f}, grad shape={grad2.shape}")
    
    # L1
    l1 = L1Regularizer(lam=0.01)
    pen1 = l1.penalty(w)
    prox1 = l1.prox(w, lr=0.1)
    print(f"  L1: penalty={pen1:.4f}, prox shape={prox1.shape}")


def test_optimizers():
    """Test new optimizer interface."""
    print("\nTesting optimizers...")
    n, d = 100, 5
    X = np.random.randn(n, d)
    w = np.random.randn(d)
    b = 0.1
    y = np.random.randint(0, 2, n).astype(float)
    z = X @ w + b
    
    loss = BCELoss()
    grad_z = loss.grad(z, y)
    grad_w = X.T @ grad_z / n
    grad_b = np.mean(grad_z)
    
    # Fixed LR
    lr = FixedLR(c=1.0, L=10.0, decay_rate=0.0)
    opt = GradientDescent(step_size=lr, regularizer=None, use_proximal=False)
    w_new, b_new = opt.step(w, b, grad_w, grad_b)
    print(f"  GD: w update norm={np.linalg.norm(w_new - w):.4f}, b update={abs(b_new - b):.4f}")
    
    # NAG with momentum
    lr2 = FixedLR(c=1.0, L=10.0, decay_rate=0.0)
    nag = NAG(step_size=lr2, momentum=0.9, regularizer=None, use_proximal=False)
    w_new2, b_new2 = nag.step(w, b, grad_w, grad_b)
    print(f"  NAG: w update norm={np.linalg.norm(w_new2 - w):.4f}, b update={abs(b_new2 - b):.4f}")


def test_metrics():
    """Test pure numpy metrics."""
    print("\nTesting metrics...")
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0.1, 0.4, 0.35, 0.8, 0.9])
    
    metrics = compute_metrics(y_true, y_pred, threshold=0.5)
    print(f"  AUROC={metrics['auroc']:.4f}, AUPRC={metrics['auprc']:.4f}, F1={metrics['f1']:.4f}")


def test_training_loop():
    """Test inline training loop."""
    print("\nTesting training loop...")
    np.random.seed(42)
    n_train, n_val, d = 200, 50, 5
    
    # Synthetic data
    X_train = np.random.randn(n_train, d)
    w_true = np.random.randn(d)
    b_true = 0.5
    z_train = X_train @ w_true + b_true
    y_train = (z_train + np.random.randn(n_train) * 0.5 > 0).astype(float)
    
    X_val = np.random.randn(n_val, d)
    z_val = X_val @ w_true + b_true
    y_val = (z_val + np.random.randn(n_val) * 0.5 > 0).astype(float)
    
    # Train with BCE + GD + L2
    loss = BCELoss()
    l2 = L2Regularizer(lam=0.01)
    lr = FixedLR(c=1.0, L=10.0, decay_rate=0.0)
    opt = GradientDescent(step_size=lr, regularizer=l2, use_proximal=False)
    
    result = train_logreg(
        X_train, y_train, X_val, y_val,
        loss, opt, l2,
        n_epochs=50,
        batch_size=32,
        seed=42,
        tol_grad=1e-4,
        patience_inner=5,
        patience_outer=10,
        dry_run=False,
    )
    
    n_epochs_run = len(result["history"]["train_loss"])
    final_train_loss = result["history"]["train_loss"][-1]
    final_val_auroc = result["history"]["val_auroc"][-1]
    
    print(f"  Epochs run: {n_epochs_run}/{50}")
    print(f"  Final train loss: {final_train_loss:.4f}")
    print(f"  Final val AUROC: {final_val_auroc:.4f}")
    print(f"  Stop reason: {result['stop_reason']}")


def _synth_data(seed=3, n=400, d=6, n_val=200):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    Xv = rng.randn(n_val, d)
    w = rng.randn(d)
    z = X @ w * 0.8 + 0.3
    y = (z + rng.randn(n) * 0.5 > 0).astype(float)
    yv = (Xv @ w * 0.8 + 0.3 + rng.randn(n_val) * 0.5 > 0).astype(float)
    return X, y, Xv, yv


def _old_delta_criterion_would_fire(train_loss, tol=1e-6, pat=5):
    """Replay of the pre-refactor Level-1 rule (absolute |dLoss| + monotone)."""
    prev = None
    ctr = 0
    for ep, l in enumerate(train_loss, 1):
        if prev is not None and l <= prev and abs(l - prev) < tol:
            ctr += 1
            if ctr >= pat:
                return ep
        else:
            ctr = 0
        prev = l
    return None


def test_level1_stationarity_no_premature_stop():
    """Tiny fixed step: objective delta ~0 (old rule fires) but gradient stays
    large. Stationarity must NOT stop early; the run uses the full budget."""
    print("\nTesting Level-1 stationarity (no premature stop)...")
    X, y, Xv, yv = _synth_data()
    ss = build_step_size('fixed', c=1e-6, L=10.0, decay_rate=0.0, cfg_bt={},
                         opt_name='gd')
    opt = GradientDescent(ss)
    r = train_logreg(X, y, Xv, yv, BCELoss(), opt, build_regularizer('l2', 0.01),
                     n_epochs=60, batch_size=32, seed=3, verbose_every=0,
                     tol_grad=1e-4, patience_inner=5, patience_outer=100)
    old_fire = _old_delta_criterion_would_fire(r["history"]["train_loss"])
    print(f"  old |dLoss| rule would fire @epoch {old_fire}; "
          f"new stop_reason={r['stop_reason']}, epochs={r['epochs_run']}")
    assert old_fire is not None and old_fire < 15, "test setup: old rule must fire early"
    assert r["stop_reason"] != "tol", "must not stop on stationarity when grad is large"
    assert r["epochs_run"] == 60, "should run the full budget, not stop prematurely"


def test_level1_stationarity_triggers_at_convergence():
    """Damped Newton reaches stationarity: stops via 'tol' with small grad."""
    print("\nTesting Level-1 stationarity (fires at convergence)...")
    X, y, Xv, yv = _synth_data()
    bt = {'alpha_init': 1.0, 'beta': 0.5, 'c_armijo': 1e-4}
    ss = build_step_size('backtracking', c=None, L=1.0, decay_rate=0.0,
                         cfg_bt=bt, opt_name='newton')
    r = train_logreg(X, y, Xv, yv, BCELoss(), Newton(ss),
                     build_regularizer('l2', 0.01), n_epochs=200,
                     batch_size=32, seed=3, verbose_every=0,
                     tol_grad=1e-4, patience_inner=5, patience_outer=30)
    print(f"  stop_reason={r['stop_reason']}, epochs={r['epochs_run']}, "
          f"final_grad_norm={r['final_grad_norm']:.3e}")
    assert r["stop_reason"] == "tol"
    assert r["final_grad_norm"] < 1e-3


def test_always_restore_best_val_model():
    """Returned (w,b) is the best-val model; final_epoch model reported separately."""
    print("\nTesting best-val model restore + final_epoch_metrics...")
    X, y, Xv, yv = _synth_data()
    bt = {'alpha_init': 1.0, 'beta': 0.5, 'c_armijo': 1e-4}
    ss = build_step_size('backtracking', c=None, L=1.0, decay_rate=0.0,
                         cfg_bt=bt, opt_name='newton')
    r = train_logreg(X, y, Xv, yv, BCELoss(), Newton(ss),
                     build_regularizer('l2', 0.01), n_epochs=200,
                     batch_size=32, seed=3, verbose_every=0,
                     tol_grad=1e-4, patience_inner=5, patience_outer=30)
    m = compute_metrics(yv, sigmoid(Xv @ r["w"] + r["b"]))
    print(f"  best_epoch={r['best_epoch']}, epochs_run={r['epochs_run']}, "
          f"restored val auroc={m['auroc']:.4f} vs best={r['best_val_auroc']:.4f}")
    assert abs(m["auroc"] - r["best_val_auroc"]) < 1e-9
    assert 0 < r["best_epoch"] <= r["epochs_run"]
    assert r["final_epoch_w"].shape == r["w"].shape


def test_auprc_tie_aware():
    """AUPRC must be tie-aware average precision (not a raw trapezoid)."""
    print("\nTesting tie-aware AUPRC...")
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.9, 0.9, 0.9])
    assert abs(auprc(y, s) - 0.5) < 1e-9, f"tied case: got {auprc(y, s)}"

    # Reference AP by definition: sum (R_k - R_{k-1}) * P_k at positive arrivals.
    y2 = np.array([1, 0, 1, 0, 1])
    s2 = np.array([1.0, 0.99, 0.98, 0.97, 0.96])
    order = np.argsort(-s2, kind='stable')
    ys = y2[order]
    n_pos = np.sum(ys)
    tp = fp = 0
    prev_r = 0.0
    ap_ref = 0.0
    for lbl in ys:
        if lbl == 1:
            tp += 1
            ap_ref += (tp / n_pos - prev_r) * (tp / (tp + fp))
            prev_r = tp / n_pos
        else:
            fp += 1
    print(f"  ap_ref={ap_ref:.4f}, auprc={auprc(y2, s2):.4f}")
    assert abs(auprc(y2, s2) - ap_ref) < 1e-9


def test_epochs_to_target():
    print("\nTesting epochs_to_target...")
    assert epochs_to_target([0.5, 0.7, 0.8], 0.8) == 3
    assert epochs_to_target([0.5, 0.7], 0.9) == ""
    assert epochs_to_target([0.9, 0.7, 0.8], 0.8) == 1


if __name__ == "__main__":
    test_losses()
    test_regularizers()
    test_optimizers()
    test_metrics()
    test_training_loop()
    test_level1_stationarity_no_premature_stop()
    test_level1_stationarity_triggers_at_convergence()
    test_always_restore_best_val_model()
    test_auprc_tie_aware()
    test_epochs_to_target()
    print("\n[PASS] All smoke tests passed!")
