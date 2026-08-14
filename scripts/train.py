"""Run one selected experiment from a loss configuration YAML."""

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runner import execute_run            # noqa: E402
from src.data_utils import load_processed  # noqa: E402
from src.step_sizes import lipschitz_constant  # noqa: E402
from src.utils import sanitize_json      # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Run a single training experiment from a loss config."
    )
    parser.add_argument("--config",         required=True,
                        help="Path to a loss YAML config file (e.g. configs/bce.yaml)")
    parser.add_argument("--data-dir",       required=True,
                        help="Path to data/processed/")
    parser.add_argument("--optimizer",      required=True,
                        choices=["gd", "sgd", "nag", "newton"])
    parser.add_argument("--regularization", default="none",
                        choices=["none", "l1", "l2"])
    parser.add_argument("--step-type",      default="fixed",
                        choices=["fixed", "backtracking", "diminishing"])
    parser.add_argument("--lam",            type=float, default=0.0,
                        help="Regularization strength λ")
    parser.add_argument("--c",              type=float, default=1.0,
                        help="Step size scale: η = c / L")
    parser.add_argument("--batch-size",     type=int,   default=256,
                        help="Mini-batch size (SGD only)")
    parser.add_argument("--output",         default=None,
                        help="Write result JSON to this path (default: print to stdout)")
    args = parser.parse_args()

    # ── Guard invalid combinations ─────────────────────────────────────────────
    if args.optimizer == "newton" and args.regularization == "l1":
        parser.error("Newton + L1 is invalid (Hessian undefined for L1)")
    if args.optimizer == "sgd" and args.step_type == "backtracking":
        parser.error("SGD + backtracking is invalid; use --step-type diminishing instead")
    if args.optimizer != "sgd" and args.step_type == "diminishing":
        parser.error("Diminishing steps are only for SGD")

    # ── Load config and data ──────────────────────────────────────────────────
    cfg    = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data   = load_processed(args.data_dir)
    counts = data["class_counts"]
    w_pos  = cfg.get("w_pos", counts["n_neg"] / max(counts["n_pos"], 1))
    w_neg  = cfg.get("w_neg", 1.0)

    L = lipschitz_constant(
        data["X_train"],
        reg_type=args.regularization,
        lam=args.lam,
        loss_name=cfg["loss"],
        w_pos=w_pos,
        w_neg=w_neg,
    )

    spec = {
        "loss_name":      cfg["loss"],
        "optimizer":      args.optimizer,
        "regularization": args.regularization,
        "step_type":      args.step_type,
        "lam":            args.lam,
        "c":              args.c,
        "L":              L,
        "batch_size":     args.batch_size,
        "w_pos":          w_pos,
        "w_neg":          w_neg,
    }

    result = sanitize_json(execute_run(spec, cfg, data))
    text   = json.dumps(result, indent=2)

    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
