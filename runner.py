"""YAML-driven experiment sweep engine."""

import argparse
import csv
import itertools
import json
import logging
from pathlib import Path

import numpy as np
import yaml

from src.data_utils import load_processed
from src.losses import build_loss
from src.metrics import compute_metrics, logits_to_proba
from src.optimizers import build_optimizer
from src.regularizers import build_regularizer
from src.step_sizes import build_step_size, lipschitz_constant
from src.utils import get_logger, sanitize_json, set_seed
from train import train_logreg

LOGGER = get_logger("runner")


# ─── Run configuration generation ────────────────────────────────────────────

def valid_step_types(optimizer: str) -> tuple:
    """Return the valid step-size modes for a given optimizer name."""
    if optimizer == "sgd":
        return ("fixed", "diminishing")
    return ("fixed", "backtracking")


def iter_run_configs(cfg: dict, data: dict):
    """
    Yield one spec dict per (optimizer, regularizer, step_type, lam, c, batch_size)
    combination, skipping invalid combos with a logged warning.
    """
    loss_name  = cfg["loss"]
    optimizers = cfg.get("optimizers", ["gd", "sgd", "nag", "newton"])
    regs       = cfg.get("regularizations", ["none", "l2", "l1"])
    lambdas    = cfg.get("lambda", [1e-3])
    scales     = cfg.get("c", [1.0])
    batches    = cfg.get("batch_sizes", [256])

    counts = data["class_counts"]
    w_pos  = cfg.get("w_pos", counts["n_neg"] / counts["n_pos"] if counts["n_pos"] else 1.0)
    w_neg  = cfg.get("w_neg", 1.0)

    # Precompute σ_max once (avoids repeated SVD across the nested loop).
    sigma = float(np.linalg.svd(data["X_train"], compute_uv=False)[0])

    for opt, reg in itertools.product(optimizers, regs):
        if opt == "newton" and reg == "l1":
            LOGGER.warning("Skipping invalid combination: Newton + L1")
            continue

        reg_lams = [0.0] if reg == "none" else lambdas

        for step_type in valid_step_types(opt):
            # Newton doesn't need a c sweep — step size is determined by BT or t=1.
            step_scales = scales if step_type in {"fixed", "diminishing"} and opt != "newton" else [1.0]

            for lam, c in itertools.product(reg_lams, step_scales):
                batch_grid = batches if opt == "sgd" else [len(data["y_train"])]

                for batch_size in batch_grid:
                    L = lipschitz_constant(
                        data["X_train"],
                        reg_type=reg,
                        lam=lam,
                        loss_name=loss_name,
                        w_pos=w_pos,
                        w_neg=w_neg,
                        sigma_max=sigma,
                    )
                    yield {
                        "loss_name":      loss_name,
                        "optimizer":      opt,
                        "regularization": reg,
                        "step_type":      step_type,
                        "lam":            float(lam),
                        "c":              float(c),
                        "L":              L,
                        "batch_size":     int(batch_size),
                        "w_pos":          float(w_pos),
                        "w_neg":          float(w_neg),
                    }


def run_id(spec: dict) -> str:
    """Generate a human-readable run identifier from a spec dict."""
    return (
        f"{spec['loss_name']}_{spec['optimizer']}_{spec['regularization']}_"
        f"{spec['step_type']}_lam{spec['lam']:g}_c{spec['c']:g}_bs{spec['batch_size']}"
    )


# ─── Single-run execution ────────────────────────────────────────────────────

def execute_run(spec: dict, cfg: dict, data: dict) -> dict:
    """Build components from spec, run train_logreg, and return a result dict."""
    loss      = build_loss(spec["loss_name"], w_pos=spec["w_pos"], w_neg=spec["w_neg"])
    reg       = build_regularizer(spec["regularization"], spec["lam"])
    step      = build_step_size(spec["step_type"], c=spec["c"], L=spec["L"])

    backtracking = spec["step_type"] == "backtracking"
    schedule     = "diminishing" if spec["step_type"] == "diminishing" else "fixed"

    optimizer = build_optimizer(
        spec["optimizer"],
        lr=step or spec["c"],
        backtracking=backtracking,
        schedule=schedule,
        **cfg.get("backtracking", {}),
        **cfg.get("newton", {}),
    )

    result = train_logreg(
        data["X_train"], data["y_train"],
        data["X_val"],   data["y_val"],
        loss, optimizer, reg,
        n_epochs=cfg.get("max_epochs", 500),
        batch_size=spec["batch_size"],
        seed=cfg.get("seed", 42),
        verbose_every=cfg.get("verbose_every", 0),
        tol=cfg.get("tol", 1e-4),
        patience_inner=cfg.get("patience_inner", 5),
        patience_outer=cfg.get("patience_outer", 20),
    )

    w = result["w"]
    b = result["b"]

    final_metrics = {}
    for split in ("train", "val"):
        X = data[f"X_{split}"]
        y = data[f"y_{split}"]
        final_metrics[split] = compute_metrics(y, logits_to_proba(X @ w + b, loss.name))

    hyperparams = {k: v for k, v in spec.items()
                   if k not in {"loss_name", "optimizer", "regularization", "step_type"}}

    return {
        "run_id":           run_id(spec),
        "loss_fn":          spec["loss_name"],
        "optimizer":        spec["optimizer"],
        "regularization":   spec["regularization"],
        "step_size_type":   spec["step_type"],
        "hyperparams":      hyperparams,
        "history":          result["history"],
        "best_w":           w,
        "best_b":           b,
        "final_metrics":    final_metrics,
        "epochs_run":       result["epochs_run"],
        "stop_reason":      result["stop_reason"],
        "best_epoch":       result["best_epoch"],
        "best_val_auprc":   result["best_val_auprc"],
        "final_grad_norm":  result["final_grad_norm"],
        "total_wall_time":  result["history"]["wall_time"][-1],
    }


# ─── Main sweep loop ─────────────────────────────────────────────────────────

def _build_summary_row(artifact: dict, path: Path) -> dict:
    return {
        "run_id":           artifact["run_id"],
        "result_path":      str(path.resolve()),
        "loss_fn":          artifact["loss_fn"],
        "optimizer":        artifact["optimizer"],
        "regularization":   artifact["regularization"],
        "step_type":        artifact["step_size_type"],
        "best_val_auprc":   artifact["best_val_auprc"],
        "best_epoch":       artifact["best_epoch"],
        "epochs_run":       artifact["epochs_run"],
        "stop_reason":      artifact["stop_reason"],
        "total_wall_time":  artifact["total_wall_time"],
    }


def main():
    parser = argparse.ArgumentParser(description="Run optimization experiments.")
    parser.add_argument("--config",      required=True,     help="Path to YAML config")
    parser.add_argument("--data-dir",    required=True,     help="Path to data/processed/")
    parser.add_argument("--results-dir", default="results", help="Output directory")
    parser.add_argument("--overwrite",   action="store_true")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Run at most 1 configuration (sanity check)")
    args = parser.parse_args()

    cfg  = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(cfg.get("seed", 42))
    data = load_processed(args.data_dir)

    out_dir = Path(args.results_dir) / cfg["loss"]
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, spec in enumerate(iter_run_configs(cfg, data)):
        if args.dry_run and i >= 1:
            break

        path = out_dir / f"{run_id(spec)}.json"

        if path.exists() and not args.overwrite:
            LOGGER.info("Skipping existing %s", path.name)
            continue

        max_epochs = 1 if args.dry_run else cfg.get("max_epochs", 500)

        try:
            artifact = execute_run(spec, {**cfg, "max_epochs": max_epochs}, data)
        except Exception:
            LOGGER.exception("Run failed: %s", run_id(spec))
            continue

        path.write_text(json.dumps(sanitize_json(artifact), indent=2), encoding="utf-8")

    # ── Rebuild complete summary CSV from all JSONs on disk ──────────────────
    # (picks up any previously completed runs in resumed sweeps)
    rows = []
    for path in sorted(out_dir.glob("*.json")):
        artifact = json.loads(path.read_text(encoding="utf-8"))
        rows.append(_build_summary_row(artifact, path))

    if rows:
        summary_path = out_dir / "summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        LOGGER.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()
