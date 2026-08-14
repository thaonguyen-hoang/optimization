"""Generate convergence, validation, timing, iteration, and LR-sensitivity plots."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REGS = ("none", "l2", "l1")


def load_runs(results_dir: str) -> list:
    """Load all per-run JSON artifacts from a results directory."""
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in Path(results_dir).glob("*.json")
    ]


def selected(runs: list) -> list:
    """
    For each (optimizer, regularization, step_type) group,
    keep only the run with the highest peak val AUPRC.
    """
    groups = {}
    for run in runs:
        key = (run["optimizer"], run["regularization"], run["step_size_type"])
        if key not in groups or run["best_val_auprc"] > groups[key]["best_val_auprc"]:
            groups[key] = run
    return list(groups.values())


def panel_plot(runs, xkey, ykey, filename, ylabel, logy=False):
    """
    Draw a 1×3 panel (one subplot per regularizer).
    Each line represents one (optimizer, step_type) combination.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ax, reg in zip(axes, REGS):
        for run in runs:
            if run["regularization"] != reg:
                continue
            y = np.asarray(run["history"][ykey], dtype=float)
            if xkey == "epoch":
                x = np.arange(1, len(y) + 1)
            else:
                x = np.asarray(run["history"][xkey], dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if np.any(mask):
                label = f"{run['optimizer']}-{run['step_size_type']}"
                ax.plot(x[mask], y[mask], label=label)
        ax.set_title(reg)
        ax.set_xlabel(xkey.replace("_", " "))
        ax.grid(alpha=0.25)
        if logy:
            ax.set_yscale("log")

    axes[0].set_ylabel(ylabel)
    axes[-1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.close(fig)


def lr_sensitivity_plot(all_runs, out_dir, loss_name):
    """
    Plot final validation loss vs learning rate for each optimizer (fixed-step only).
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for opt in sorted({r["optimizer"] for r in all_runs}):
        by_lr = {}
        for run in all_runs:
            if run["optimizer"] != opt or run["step_size_type"] != "fixed":
                continue
            lr  = run["hyperparams"]["c"] / run["hyperparams"]["L"]
            val = run["history"]["val_loss"][-1]
            by_lr[lr] = min(by_lr.get(lr, float("inf")), val)
        points = sorted(by_lr.items())
        if points:
            ax.plot(
                [p[0] for p in points],
                [p[1] for p in points],
                marker="o",
                label=opt,
            )

    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("final validation loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{loss_name}_lr_sensitivity.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot sweep results.")
    parser.add_argument("--results-dir", required=True,  help="Directory with per-run JSONs")
    parser.add_argument("--output-dir",  default="plots", help="Where to save plot files")
    args = parser.parse_args()

    all_runs = load_runs(args.results_dir)
    if not all_runs:
        raise ValueError("No result JSON files found in the results directory.")

    runs = selected(all_runs)
    out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    loss = runs[0]["loss_fn"]

    # ── Convergence and validation panels ─────────────────────────────────────
    plots = [
        ("epoch",     "train_loss",    f"{loss}_loss_epoch.png",      "training loss",      False),
        ("epoch",     "val_auprc",     f"{loss}_auprc_epoch.png",     "validation AUPRC",   False),
        ("wall_time", "train_loss",    f"{loss}_loss_time.png",       "training loss",      False),
        ("iter",      "train_loss",    f"{loss}_loss_iteration.png",  "training loss",      False),
        ("epoch",     "grad_norm",     f"{loss}_gradient_epoch.png",  "gradient norm",      True),
    ]
    for xkey, ykey, fname, label, log in plots:
        panel_plot(runs, xkey, ykey, out / fname, label, log)

    # ── LR sensitivity ────────────────────────────────────────────────────────
    lr_sensitivity_plot(all_runs, out, loss)


if __name__ == "__main__":
    main()
