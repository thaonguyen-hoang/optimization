"""
person_template.py
===================
COPY THIS FILE per person (e.g. person1_bce.py, person2_weighted_bce.py,
person3_focal.py) and fill in the marked sections. Everything imported
below is SHARED — do not edit data_utils.py / losses.py / optimizers.py
/ regularizers.py / train.py / metrics.py / plotting.py themselves,
only add to them if the group agrees a new shared piece is needed.

Flow implemented here (matches the group's agreed ablation plan):
    Stage 1: optimizer ablation (own LR sweep per optimizer) -> opt*
    Stage 2b: loss-specific hyperparameter sweep (weight/alpha/gamma)
    Stage 3: regularization ablation (own lambda sweep per regularizer) -> reg*
    Final: report best-setup metrics row + supporting plots
"""

import numpy as np
import pandas as pd

from data_utils import get_feature_columns, to_xy, class_counts, TARGET_COL
from losses import BCELoss, WeightedBCELoss, FocalLoss, SquaredHingeLoss  # noqa
from optimizers import GD, SGD, SGDMomentum, Adam, LR_GRID
from regularizers import NoReg, L2Reg, L1Reg, ElasticNetReg
from train import train_logreg, predict_logits, hessian_eigs_logreg
from metrics import compute_metrics, logits_to_proba
from plotting import (plot_convergence, plot_multi_convergence,
                       plot_confusion_matrix, plot_hessian_spectrum,
                       plot_lr_sensitivity)

# ---------------------------------------------------------------------
# 0. CONFIG — fill in your name / assigned loss
# ---------------------------------------------------------------------
PERSON_NAME = "person1"          # e.g. "person1", "person2", "person3"
ASSIGNED_LOSS_NAME = "bce"       # one of: "bce", "weighted_bce", "focal", "squared_hinge"
SHARED_SPLITS_DIR = "./shared_splits"   # output of make_shared_splits.py
N_EPOCHS = 100
BATCH_SIZE = 512

# ---------------------------------------------------------------------
# 1. LOAD SHARED, FROZEN SPLITS (identical for everyone)
# ---------------------------------------------------------------------
train_df = pd.read_csv(f"{SHARED_SPLITS_DIR}/train.csv")
val_df = pd.read_csv(f"{SHARED_SPLITS_DIR}/val.csv")
test_df = pd.read_csv(f"{SHARED_SPLITS_DIR}/test.csv")

feature_cols = get_feature_columns(train_df)
X_train, y_train = to_xy(train_df, feature_cols)
X_val, y_val = to_xy(val_df, feature_cols)
X_test, y_test = to_xy(test_df, feature_cols)

print("class balance (train):", class_counts(y_train))

# ---------------------------------------------------------------------
# 2. STAGE 2b — loss-specific hyperparameter pre-sweep
#    (skip this block if your assigned loss is plain BCE)
# ---------------------------------------------------------------------
# Example for weighted_bce: sweep class-weight ratio.
# Example for focal: sweep (alpha, gamma).
# Keep this sweep SMALL (a handful of settings) -- use a FIXED cheap
# optimizer (e.g. Adam, default lr) here just to rank candidates; the
# real optimizer tuning happens in Stage 1 below on the winner.

def build_loss(name, **kwargs):
    if name == "bce":
        return BCELoss()
    if name == "weighted_bce":
        return WeightedBCELoss(w_pos=kwargs.get("w_pos", 1.0),
                                w_neg=kwargs.get("w_neg", 1.0))
    if name == "focal":
        return FocalLoss(alpha=kwargs.get("alpha", 0.25),
                          gamma=kwargs.get("gamma", 2.0))
    if name == "squared_hinge":
        return SquaredHingeLoss()
    raise ValueError(name)


# --- fill in candidate hyperparams for YOUR loss, or leave as [{}] for BCE/hinge ---
loss_hparam_candidates = [
    {},  # e.g. {"w_pos": 2.0, "w_neg": 1.0} for weighted_bce
]

best_loss_hparams = None
best_loss_score = -np.inf
loss_sweep_log = []

for hp in loss_hparam_candidates:
    loss_fn = build_loss(ASSIGNED_LOSS_NAME, **hp)
    result = train_logreg(
        X_train, y_train, X_val, y_val,
        loss_fn=loss_fn,
        optimizer=Adam(lr=1e-3),
        regularizer=NoReg(),
        n_epochs=30, batch_size=BATCH_SIZE, verbose_every=0,
    )
    p_val = logits_to_proba(predict_logits(X_val, result["w"], result["b"]), ASSIGNED_LOSS_NAME)
    m = compute_metrics(y_val, p_val)
    loss_sweep_log.append({**hp, "auprc": m["auprc"], "f1_minority": m["f1_minority"]})
    if m["auprc"] > best_loss_score:
        best_loss_score = m["auprc"]
        best_loss_hparams = hp

print("Loss hyperparam sweep:", pd.DataFrame(loss_sweep_log))
print("Best loss hyperparams:", best_loss_hparams)

loss_fn = build_loss(ASSIGNED_LOSS_NAME, **best_loss_hparams)

# ---------------------------------------------------------------------
# 3. STAGE 1 — optimizer ablation (own LR grid per optimizer)
# ---------------------------------------------------------------------
optimizer_candidates = {
    "gd": lambda lr: GD(lr=lr),
    "sgd": lambda lr: SGD(lr=lr),
    "sgd_momentum": lambda lr: SGDMomentum(lr=lr, momentum=0.9),
    "adam": lambda lr: Adam(lr=lr),
}

opt_stage_histories = {}      # best-lr history per optimizer, for the overlay plot
opt_lr_sweep_results = {}     # per-optimizer {lr -> final val loss}, for the LR-sensitivity plot
best_opt_name, best_opt_lr, best_opt_score = None, None, -np.inf

for opt_name, opt_factory in optimizer_candidates.items():
    lr_to_val = {}
    best_for_this_opt = (None, -np.inf, None)  # (lr, auprc, history)

    for lr in LR_GRID:
        optimizer = opt_factory(lr)
        result = train_logreg(
            X_train, y_train, X_val, y_val,
            loss_fn=loss_fn, optimizer=optimizer, regularizer=NoReg(),
            n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, verbose_every=0,
        )
        lr_to_val[lr] = result["history"]["val_loss"][-1]

        p_val = logits_to_proba(predict_logits(X_val, result["w"], result["b"]), ASSIGNED_LOSS_NAME)
        m = compute_metrics(y_val, p_val)
        if m["auprc"] > best_for_this_opt[1]:
            best_for_this_opt = (lr, m["auprc"], result["history"])

    opt_lr_sweep_results[opt_name] = lr_to_val
    opt_stage_histories[opt_name] = best_for_this_opt[2]
    print(f"[{opt_name}] best lr={best_for_this_opt[0]}  val AUPRC={best_for_this_opt[1]:.4f}")

    if best_for_this_opt[1] > best_opt_score:
        best_opt_score = best_for_this_opt[1]
        best_opt_name = opt_name
        best_opt_lr = best_for_this_opt[0]

print(f"\n>>> Best optimizer for {ASSIGNED_LOSS_NAME}: {best_opt_name} (lr={best_opt_lr})")

# Figures for the optimizer stage:
fig_opt_compare = plot_multi_convergence(opt_stage_histories, x_axis="wall_time",
                                          title=f"Optimizer comparison ({ASSIGNED_LOSS_NAME})")
# fig_opt_compare.savefig(f"{PERSON_NAME}_stage1_optimizer_comparison.png", dpi=150)

best_optimizer_factory = optimizer_candidates[best_opt_name]

# ---------------------------------------------------------------------
# 4. STAGE 3 — regularization ablation (own lambda grid per regularizer)
# ---------------------------------------------------------------------
LAMBDA_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]

reg_candidates = {
    "none": lambda lam: NoReg(),
    "l2": lambda lam: L2Reg(lam=lam),
    "l1": lambda lam: L1Reg(lam=lam),
    "elastic_net": lambda lam: ElasticNetReg(lam_l1=lam, lam_l2=lam),
}

best_reg_name, best_reg_lam, best_reg_score, best_reg_result = "none", 0.0, -np.inf, None

for reg_name, reg_factory in reg_candidates.items():
    lam_grid = [0.0] if reg_name == "none" else LAMBDA_GRID
    for lam in lam_grid:
        optimizer = best_optimizer_factory(best_opt_lr)
        result = train_logreg(
            X_train, y_train, X_val, y_val,
            loss_fn=loss_fn, optimizer=optimizer, regularizer=reg_factory(lam),
            n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, verbose_every=0,
        )
        p_val = logits_to_proba(predict_logits(X_val, result["w"], result["b"]), ASSIGNED_LOSS_NAME)
        m = compute_metrics(y_val, p_val)
        if m["auprc"] > best_reg_score:
            best_reg_score = m["auprc"]
            best_reg_name, best_reg_lam, best_reg_result = reg_name, lam, result

print(f"\n>>> Best regularizer for {ASSIGNED_LOSS_NAME}: {best_reg_name} (lambda={best_reg_lam})")

# ---------------------------------------------------------------------
# 5. FINAL EVALUATION on held-out TEST set with the fully-tuned setup
# ---------------------------------------------------------------------
final_optimizer = best_optimizer_factory(best_opt_lr)
final_regularizer = reg_candidates[best_reg_name](best_reg_lam)

final_result = train_logreg(
    X_train, y_train, X_val, y_val,
    loss_fn=loss_fn, optimizer=final_optimizer, regularizer=final_regularizer,
    n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, verbose_every=10,
)

z_test = predict_logits(X_test, final_result["w"], final_result["b"])
p_test = logits_to_proba(z_test, ASSIGNED_LOSS_NAME)
final_metrics = compute_metrics(y_test, p_test)

print(f"\n=== FINAL SETUP: {PERSON_NAME} / {ASSIGNED_LOSS_NAME} ===")
print(f"loss hparams: {best_loss_hparams}")
print(f"optimizer: {best_opt_name} (lr={best_opt_lr})")
print(f"regularizer: {best_reg_name} (lambda={best_reg_lam})")
print("test metrics:", final_metrics)

# This is the row that goes into the group's final comparison table.
final_row = {
    "person": PERSON_NAME,
    "loss": ASSIGNED_LOSS_NAME,
    "loss_hparams": best_loss_hparams,
    "optimizer": best_opt_name,
    "lr": best_opt_lr,
    "regularizer": best_reg_name,
    "lambda": best_reg_lam,
    **final_metrics,
}
pd.DataFrame([final_row]).to_csv(f"{PERSON_NAME}_final_row.csv", index=False)

# ---------------------------------------------------------------------
# 6. SUPPORTING FIGURES
# ---------------------------------------------------------------------
fig_final_convergence = plot_convergence(
    final_result["history"], title=f"{PERSON_NAME} final setup ({ASSIGNED_LOSS_NAME})")
fig_confusion = plot_confusion_matrix(
    {k: final_metrics[k] for k in ["tp", "tn", "fp", "fn"]},
    title=f"{PERSON_NAME} test confusion matrix")

# Hessian conditioning story (most meaningful for L2 vs none)
eigvals_noreg = hessian_eigs_logreg(X_train, final_result["w"], final_result["b"],
                                     lam_l2=0.0, sample_size=20000)
eigvals_l2 = hessian_eigs_logreg(X_train, final_result["w"], final_result["b"],
                                  lam_l2=best_reg_lam if best_reg_name == "l2" else 0.1,
                                  sample_size=20000)
fig_hessian = plot_hessian_spectrum(eigvals_l2, title=f"{PERSON_NAME} Hessian spectrum (with L2)")

# fig_final_convergence.savefig(f"{PERSON_NAME}_final_convergence.png", dpi=150)
# fig_confusion.savefig(f"{PERSON_NAME}_confusion.png", dpi=150)
# fig_hessian.savefig(f"{PERSON_NAME}_hessian.png", dpi=150)

print(f"\nDone. Bring {PERSON_NAME}_final_row.csv to the group merge step.")
