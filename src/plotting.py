"""
plotting.py
===========
SHARED module. Common plot styles so every person's figures look
consistent when assembled into the final group report/slides.

All functions return a matplotlib Figure (don't call plt.show()
inside them) so callers can save or embed as needed.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_convergence(history, title="Convergence", x_axis="epoch"):
    """x_axis: 'epoch' or 'wall_time' -- use 'wall_time' when comparing
    optimizers for practical efficiency, 'epoch' for statistical
    efficiency (iterations-to-tolerance).
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = history[x_axis]

    axes[0].plot(x, history["train_loss"], label="train")
    axes[0].plot(x, history["val_loss"], label="val")
    axes[0].set_xlabel(x_axis)
    axes[0].set_ylabel("loss")
    axes[0].set_title(f"{title} — loss")
    axes[0].legend()

    axes[1].plot(x, history["grad_norm"], color="tab:red")
    axes[1].set_xlabel(x_axis)
    axes[1].set_ylabel("||grad||")
    axes[1].set_yscale("log")
    axes[1].set_title(f"{title} — gradient norm")

    fig.tight_layout()
    return fig


def plot_multi_convergence(histories: dict, x_axis="epoch", metric="val_loss",
                            title="Optimizer comparison"):
    """histories: {label -> history_dict}. Overlay multiple runs, e.g.
    all optimizers for one loss, or all losses for one optimizer.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, history in histories.items():
        ax.plot(history[x_axis], history[metric], label=label)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cc: dict, title="Confusion matrix"):
    """cc: dict with tp, tn, fp, fn (from metrics.confusion_counts)."""
    matrix = np.array([[cc["tn"], cc["fp"]],
                        [cc["fn"], cc["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                     color="black", fontsize=12)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_hessian_spectrum(eigvals, title="Hessian eigenvalue spectrum"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.sort(eigvals)[::-1], marker="o", markersize=3)
    ax.set_yscale("log")
    ax.set_xlabel("index (sorted)")
    ax.set_ylabel("eigenvalue (log scale)")
    cond = eigvals.max() / max(eigvals.min(), 1e-12)
    ax.set_title(f"{title}\ncondition number ≈ {cond:.2e}")
    fig.tight_layout()
    return fig


def plot_metrics(history, title="Validation metrics"):
    """Plot val AUPRC / F1-minority / accuracy across epochs.

    Uses epoch-level points (where val_* are non-NaN). Returns a Figure
    with three metric curves so they can be tracked without re-running.
    """
    hist = {k: np.asarray(v) for k, v in history.items()}
    mask = ~np.isnan(hist["val_auprc"])
    epochs = hist["epoch"][mask]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, hist["val_auprc"][mask], label="AUPRC", marker="o", ms=3)
    ax.plot(epochs, hist["val_f1_minority"][mask], label="F1-minority", marker="s", ms=3)
    ax.plot(epochs, hist["val_accuracy"][mask], label="accuracy", marker="^", ms=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_convergence_full(history, title="Convergence", x_axis="epoch"):
    """Three-panel convergence: loss, gradient norm, val metrics.

    x_axis: 'epoch' or 'wall_time'. Loss/grad use iter-level (subsampled)
    points; metrics use epoch-level points.
    """
    hist = {k: np.asarray(v) for k, v in history.items()}
    x_loss = hist[x_axis]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    tmask = ~np.isnan(hist["train_loss"])
    axes[0].plot(x_loss[tmask], hist["train_loss"][tmask], label="train",
                 alpha=0.8, marker="o", ms=3)
    vmask = ~np.isnan(hist["val_loss"])
    axes[0].plot(x_loss[vmask], hist["val_loss"][vmask], label="val",
                 alpha=0.8, marker="s", ms=3)
    axes[0].set_xlabel(x_axis); axes[0].set_ylabel("loss")
    axes[0].set_title(f"{title} — loss"); axes[0].legend()

    gmask = ~np.isnan(hist["grad_norm"])
    axes[1].plot(x_loss[gmask], hist["grad_norm"][gmask], color="tab:red")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(x_axis); axes[1].set_ylabel("||grad||")
    axes[1].set_title(f"{title} — gradient norm")

    mmask = ~np.isnan(hist["val_auprc"])
    xe = hist[x_axis][mmask]
    axes[2].plot(xe, hist["val_auprc"][mmask], label="AUPRC", marker="o", ms=3)
    axes[2].plot(xe, hist["val_f1_minority"][mmask], label="F1-min.", marker="s", ms=3)
    axes[2].plot(xe, hist["val_accuracy"][mmask], label="acc", marker="^", ms=3)
    axes[2].set_xlabel(x_axis); axes[2].set_ylabel("score")
    axes[2].set_title(f"{title} — val metrics"); axes[2].legend()

    fig.tight_layout()
    return fig


def plot_lr_sensitivity(lr_to_final_loss: dict, title="LR sensitivity"):
    """lr_to_final_loss: {lr_value -> final val_loss}. One line per
    optimizer if you overlay several dicts -- otherwise call once per
    optimizer and arrange as a small-multiples grid in your report.
    """
    lrs = sorted(lr_to_final_loss.keys())
    vals = [lr_to_final_loss[lr] for lr in lrs]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(lrs, vals, marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("final val loss")
    ax.set_title(title)
    fig.tight_layout()
    return fig
