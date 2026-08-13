"""
plot_hessian.py — Hessian eigenvalue spectrum plot.

Computes and plots the eigenvalue spectrum of the Hessian of the BCE loss
at the final trained weights for:
  (a) No regularization
  (b) L2 regularization (ridge)

This illustrates how L2 regularization shifts all eigenvalues up by λ,
improving the condition number (κ = λ_max / λ_min) and making the loss
landscape more isotropic.

Usage:
    python plot_hessian.py --data_dir data/processed --lambda_l2 0.01

The script trains BCE+GD+fixed with and without L2, computes H at final
(w, b), and plots the spectrum.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import load_data, set_seed, get_logger, hessian_joint
from src.losses import BCELoss
from src.regularizers import L2Regularizer
from src.step_sizes import FixedLR, lipschitz_constant
from src.optimizers import GradientDescent
from runner import train_logreg

matplotlib.rcParams.update({"font.family": "sans-serif", "font.size": 12})
sns.set_theme(style="whitegrid")

logger = get_logger("plot_hessian")
PLOTS_DIR = Path("plots")


def train_model(X_train, y_train, X_val, y_val,
                lambda_reg: float = 0.0, max_epochs: int = 500,
                seed: int = 42):
    """Train a BCE logistic regression model with optional L2 reg."""
    loss_fn = BCELoss()
    reg_type = "l2" if lambda_reg > 0 else "none"
    regularizer = L2Regularizer(lambda_reg) if lambda_reg > 0 else None

    L = lipschitz_constant(X_train, reg_type=reg_type, lambda_reg=lambda_reg)
    step_size = FixedLR(c=1.0, L=L)
    optimizer = GradientDescent(step_size, regularizer=regularizer, use_proximal=False)

    result = train_logreg(
        X_train, y_train, X_val, y_val,
        loss_fn, optimizer, regularizer,
        n_epochs=max_epochs,
        batch_size=32,
        seed=seed,
        verbose_every=0,
        tol_grad=1e-8,
        patience_inner=10,
        patience_outer=30,
    )
    return result, loss_fn


def compute_hessian(result, loss_fn, X: np.ndarray, y: np.ndarray,
                    lambda_reg: float = 0.0) -> np.ndarray:
    """Compute the joint Hessian of BCE loss (+ L2 if applicable) at trained (w, b)."""
    w, b = result["w"], result["b"]

    z = X @ w + b
    diag_H = loss_fn.hessian(z, y)

    reg_diag = None
    if lambda_reg > 0:
        reg_diag = lambda_reg * np.eye(w.shape[0])
    return hessian_joint(X, diag_H, reg_diag)


def plot_spectrum(eigenvalues_dict: dict, title: str, filename: str):
    """Plot sorted eigenvalue curves and a histogram of eigenvalues."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    colors = {"No Regularization": "#d62728", "L2 Regularization": "#1f77b4"}

    for label, eigvals in eigenvalues_dict.items():
        sorted_eigs = np.sort(eigvals)[::-1]
        color = colors.get(label, "gray")

        ax1.plot(sorted_eigs, label=label, color=color, linewidth=2)
        ax2.hist(eigvals, bins=50, alpha=0.6, label=label, color=color,
                 edgecolor="none")

    ax1.set_xlabel("Eigenvalue index (sorted descending)")
    ax1.set_ylabel("Eigenvalue")
    ax1.set_title("Sorted Eigenvalue Spectrum")
    ax1.legend()
    ax1.set_yscale("log")

    ax2.set_xlabel("Eigenvalue")
    ax2.set_ylabel("Count")
    ax2.set_title("Eigenvalue Distribution")
    ax2.legend()

    for label, eigvals in eigenvalues_dict.items():
        kappa = max(eigvals) / max(min(eigvals), 1e-12)
        color = colors.get(label, "gray")
        ax1.axhline(max(eigvals), color=color, linestyle=":", alpha=0.5,
                    label=f"max ({label}): {max(eigvals):.3f}")
        logger.info(f"{label}: max={max(eigvals):.4f}, "
                    f"min={min(eigvals):.4f}, kappa={kappa:.1f}")

    plt.tight_layout()
    out_path = PLOTS_DIR / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default="data/processed")
    parser.add_argument("--lambda_l2",   type=float, default=0.01,
                        help="L2 regularization strength for the comparison")
    parser.add_argument("--max_epochs",  type=int,   default=500)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    PLOTS_DIR.mkdir(exist_ok=True)
    data = load_data(args.data_dir)
    X_train, y_train = data["train"]
    X_val,   y_val   = data["val"]

    logger.info("Training model WITHOUT regularization...")
    result_noreg, loss_noreg = train_model(X_train, y_train, X_val, y_val,
                                            lambda_reg=0.0, max_epochs=args.max_epochs,
                                            seed=args.seed)

    logger.info(f"Training model WITH L2 (lambda={args.lambda_l2})...")
    result_l2, loss_l2 = train_model(X_train, y_train, X_val, y_val,
                                      lambda_reg=args.lambda_l2, max_epochs=args.max_epochs,
                                      seed=args.seed)

    logger.info("Computing Hessian eigenvalues...")
    H_noreg = compute_hessian(result_noreg, loss_noreg, X_train, y_train, lambda_reg=0.0)
    H_l2    = compute_hessian(result_l2, loss_l2, X_train, y_train, lambda_reg=args.lambda_l2)

    eigs_noreg = np.linalg.eigvalsh(H_noreg)
    eigs_l2    = np.linalg.eigvalsh(H_l2)

    eigenvalues_dict = {
        "No Regularization": eigs_noreg,
        "L2 Regularization": eigs_l2,
    }

    plot_spectrum(
        eigenvalues_dict,
        title=f"Hessian Eigenvalue Spectrum — BCE Loss (L2 lambda={args.lambda_l2})",
        filename=f"hessian_spectrum_lambda{args.lambda_l2}.png",
    )


if __name__ == "__main__":
    main()
