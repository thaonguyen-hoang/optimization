"""
plot_results.py — Generate loss curve and val AUROC curve plots from JSON results.

Usage:
    python plot_results.py --results_dir results --loss bce
    python plot_results.py --results_dir results --loss all

Plots generated per loss function:
  1. Train loss vs epoch — one subplot per regularization type,
     lines colored by optimizer, linestyle by step size type
  2. Val AUROC vs epoch — same structure

Best config per (optimizer, reg_type, step_type) is selected by highest
peak val AUROC to keep plots readable.
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size":   11,
    "axes.grid":   True,
    "grid.alpha":  0.3,
})
sns.set_theme(style="whitegrid", palette="tab10")

PLOTS_DIR = Path("plots")

# Outputs of evaluate_best.py that are not per-run result files.
EXCLUDED_RESULTS = {"final_test_results.json"}

OPTIMIZER_COLORS = {
    "gd":     "#1f77b4",
    "sgd":    "#ff7f0e",
    "nag":    "#2ca02c",
    "newton": "#d62728",
    "lbfgs":  "#9467bd",
}
STEP_LINESTYLES = {
    "fixed":        "-",
    "backtracking": "--",
}
REG_LABELS = {
    "none": "No Regularization",
    "l2":   "L2 (Ridge)",
    "l1":   "L1 (Lasso / Proximal)",
}

LOSS_NAMES = ["bce", "weighted_bce", "squared_hinge", "focal"]


def load_results(results_dir: Path, loss_name: str) -> list[dict]:
    """Load all JSON results for a given loss function."""
    # We now search recursively across subdirectories (results/*/*.json)
    pattern = f"**/{loss_name}_*.json" if loss_name != "all" else "**/*.json"
    results = []
    for jf in results_dir.glob(pattern):
        if jf.name in EXCLUDED_RESULTS:
            continue
        with open(jf) as f:
            r = json.load(f)
        if loss_name == "all" or r.get("loss_fn") == loss_name:
            results.append(r)
    return results


def select_best_per_group(results: list[dict]) -> dict:
    """
    For each (optimizer, regularization, step_size_type) group,
    select the run with the highest peak val AUROC.
    Returns dict keyed by (opt, reg, step).
    """
    groups = defaultdict(list)
    for r in results:
        key = (r.get("optimizer"), r.get("regularization"), r.get("step_size_type"))
        groups[key].append(r)

    best = {}
    for key, runs in groups.items():
        best[key] = max(runs, key=lambda r: max(r.get("val_auroc_curve", [0])))
    return best


def plot_curves_for_loss(
    loss_name: str,
    results: list[dict],
    metric: str = "train_loss_curve",
    x_axis: str = "epoch",   # 'epoch' or 'time'
):
    """
    Generate a 1×3 subplot figure: one panel per regularization type.
    Lines = optimizers, linestyle = step size type.

    Parameters
    ----------
    metric : 'train_loss_curve' | 'val_auroc_curve'
    x_axis : 'epoch' — x = epoch number
             'time'  — x = cumulative wall-clock optimizer time (seconds)
    """
    reg_types = ["none", "l2", "l1"]
    ylabel = "Training Loss" if metric == "train_loss_curve" else "Validation AUROC"
    title_suffix = "Loss Convergence" if metric == "train_loss_curve" else "Val AUROC"
    x_label = "Epoch" if x_axis == "epoch" else "Cumulative Optimizer Time (s)"
    title_suffix += " vs " + ("Epoch" if x_axis == "epoch" else "Wall-Clock Time")

    best_runs = select_best_per_group(results)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle(f"{loss_name.upper().replace('_', ' ')} — {title_suffix}",
                 fontsize=14, fontweight="bold", y=1.02)

    for ax, reg in zip(axes, reg_types):
        ax.set_title(REG_LABELS[reg], fontsize=11)
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)

        plotted = False
        for opt in ["gd", "sgd", "nag", "newton", "lbfgs"]:
            for step in ["fixed", "backtracking"]:
                key = (opt, reg, step)
                if key not in best_runs:
                    continue
                run = best_runs[key]
                curve = run.get(metric, [])
                if not curve:
                    continue

                if x_axis == "epoch":
                    x_vals = list(range(1, len(curve) + 1))
                else:
                    # Use cumulative wall-clock time as x-axis
                    x_vals = run.get("cumulative_time_s") or run.get("wall_time_per_epoch") or []
                    if not x_vals or len(x_vals) != len(curve):
                        # Fall back to epoch if timing data is missing
                        x_vals = list(range(1, len(curve) + 1))

                label = f"{opt.upper()} ({step})"
                ax.plot(
                    x_vals, curve,
                    color=OPTIMIZER_COLORS.get(opt, "gray"),
                    linestyle=STEP_LINESTYLES[step],
                    linewidth=1.8,
                    label=label,
                    alpha=0.85,
                )
                plotted = True

        if plotted:
            ax.legend(fontsize=8, loc="best")
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color="gray")

    plt.tight_layout()
    metric_tag = "loss" if metric == "train_loss_curve" else "auroc"
    axis_tag   = "epoch" if x_axis == "epoch" else "time"
    out_path = PLOTS_DIR / f"{loss_name}_{metric_tag}_vs_{axis_tag}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--loss", default="all",
                        help="Loss name or 'all'")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    PLOTS_DIR.mkdir(exist_ok=True)
    losses = LOSS_NAMES if args.loss == "all" else [args.loss]

    for loss_name in losses:
        results = load_results(results_dir, loss_name)
        if not results:
            print(f"No results found for {loss_name}, skipping.")
            continue
        print(f"Plotting {loss_name} ({len(results)} runs)...")
        # Loss vs epoch
        plot_curves_for_loss(loss_name, results, metric="train_loss_curve", x_axis="epoch")
        plot_curves_for_loss(loss_name, results, metric="val_auroc_curve",  x_axis="epoch")
        # Loss vs wall-clock time (key for optimizer efficiency comparison)
        plot_curves_for_loss(loss_name, results, metric="train_loss_curve", x_axis="time")
        plot_curves_for_loss(loss_name, results, metric="val_auroc_curve",  x_axis="time")


if __name__ == "__main__":
    main()
