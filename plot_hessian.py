"""Compare BCE Hessian eigenvalue spectra at GD solutions with and without L2."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data_utils import load_processed
from src.losses import BCELoss
from src.optimizers import build_optimizer
from src.regularizers import build_regularizer
from src.step_sizes import lipschitz_constant
from train import train_logreg


def fit_and_spectrum(data: dict, lam: float, epochs: int, seed: int) -> np.ndarray:
    """
    Train BCE + GD (fixed step) with the given L2 strength,
    then compute and return the Hessian eigenvalues at the converged solution.
    """
    reg_name  = "l2" if lam else "none"
    reg       = build_regularizer(reg_name, lam)
    loss      = BCELoss()
    L         = lipschitz_constant(data["X_train"], reg=reg, lam=lam, loss_name="bce")
    optimizer = build_optimizer("gd", lr=float(1.0 / L))

    result = train_logreg(
        data["X_train"], data["y_train"],
        data["X_val"],   data["y_val"],
        loss, optimizer, reg,
        n_epochs=epochs,
        seed=seed,
        tol=0.0,   # disable early stopping; run for the full budget
    )

    w = result["w"]
    b = result["b"]
    X = data["X_train"]

    p       = 1.0 / (1.0 + np.exp(-(X @ w + b)))
    weights = p * (1.0 - p)
    H       = (X * weights[:, None]).T @ X / len(X) + lam * np.eye(X.shape[1])
    return np.linalg.eigvalsh(H)


def condition_number(eigs: np.ndarray) -> float:
    """κ = λ_max / λ_min (positive eigenvalues only)."""
    positive = eigs[eigs > np.finfo(float).eps]
    if len(positive) == 0:
        return float("inf")
    return float(positive[-1] / positive[0])


def main():
    parser = argparse.ArgumentParser(description="Plot Hessian eigenvalue spectrum.")
    parser.add_argument("--data-dir",   required=True)
    parser.add_argument("--lambda-l2",  type=float, default=1e-2,
                        help="L2 regularization strength for the comparison run")
    parser.add_argument("--epochs",     type=int,   default=500)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--output",     default="hessian_spectrum.png")
    args = parser.parse_args()

    data = load_processed(args.data_dir)

    spectra = {
        "none":                fit_and_spectrum(data, 0.0,          args.epochs, args.seed),
        f"L2 ({args.lambda_l2:g})": fit_and_spectrum(data, args.lambda_l2, args.epochs, args.seed),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for label, eigs in spectra.items():
        kappa      = condition_number(eigs)
        plot_label = f"{label},  κ = {kappa:.2g}"
        axes[0].plot(np.arange(1, len(eigs) + 1), np.maximum(eigs, 1e-16), label=plot_label)
        axes[1].hist(eigs, bins=min(30, max(5, len(eigs))), alpha=0.5, label=label)

    axes[0].set_yscale("log")
    axes[0].set_xlabel("sorted eigenvalue index")
    axes[0].set_ylabel("eigenvalue (log scale)")

    axes[1].set_xlabel("eigenvalue")
    axes[1].set_ylabel("count")

    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()

    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
