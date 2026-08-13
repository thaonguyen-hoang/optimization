#!/usr/bin/env python
"""
scripts/expand_tune.py
======================
Expand a tuning YAML grid into a manifest TSV on stdout. One row per
valid (hyper)parameter combination. No training, no metrics, no JSON —
pure grid expansion so bash ``run_tune.sh`` can loop over the trials.

Usage:
    python -m scripts.expand_tune configs/tune_bce.yaml > manifest.tsv

TSV columns (16):
    loss  w_pos  reg  lam  opt  lr  schedule  backtracking  initial_lr
    batch_size  armijo_alpha  armijo_beta  epochs  loss_epsilon  patience  seed

Conventions:
    backtracking            = 1 / 0
    batch_size              = "." for full-batch optimizers (gd/nag/newton)
    lr / initial_lr / armijo_alpha / armijo_beta = "." when not applicable

Combination rules (per optimizer/reg), see TUNING_PLAN.md section 4.1:
    gd      none/l2  : fixed lr x lr_grid ; backtracking (alpha x beta)
    gd      l1 (ISTA): fixed lr x lr_grid ; backtracking beta only (Parabol)
    nag     none/l2  : fixed lr x lr_grid ; backtracking beta only
    nag     l1 (FISTA): fixed lr x lr_grid ; backtracking beta only
    newton  none/l2  : fixed lr (=1.0) ; backtracking (alpha x beta)
    newton  l1       : dropped (mathematically invalid)
    sgd     none/l2/l1 : fixed/diminishing x batch_size; no backtracking
"""

import argparse
import itertools
import sys

import yaml

COLUMNS = [
    "loss", "w_pos", "reg", "lam", "opt", "lr", "schedule",
    "backtracking", "initial_lr", "batch_size", "armijo_alpha",
    "armijo_beta", "epochs", "loss_epsilon", "patience", "seed",
]

# Backtracking uses (alpha x beta) on F for these; others (nag) Parabol beta-only.
ALPHA_BETA_BT = {"gd", "newton"}
# Optimizers that operate on the full batch.
FULL_BATCH = {"gd", "nag", "newton"}
# Optimizers that cannot handle non-smooth L1.
NO_L1 = {"newton"}


def fmt(v):
    """Compact canonical number string; '.' for None. Ints stay ints."""
    if v is None:
        return "."
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    return format(float(v), ".12g")


def emit(loss, w_pos, reg, lam, opt, lr, schedule, bt, initial_lr,
         batch_size, alpha, beta, epochs, epsilon, patience, seed):
    vals = [loss, fmt(w_pos), reg, fmt(lam), opt, fmt(lr), schedule,
            "1" if bt else "0", fmt(initial_lr), fmt(batch_size),
            fmt(alpha), fmt(beta), str(epochs), fmt(epsilon),
            str(patience), str(seed)]
    assert len(vals) == len(COLUMNS), f"bad row: {vals}"
    return "\t".join(vals)


def gen(yaml_data):
    globals_conf = yaml_data.get("globals", {})
    loss = yaml_data.get("loss", "bce")
    epochs = int(globals_conf.get("tune_epochs", 100))
    epsilon = float(globals_conf.get("loss_epsilon", 1e-4))
    patience = int(globals_conf.get("patience", 3))
    seed = int(globals_conf.get("seed", 42))
    # w_pos swept only for weighted_bce; fixed at 1.0 otherwise.
    w_pos_grid = [float(w) for w in globals_conf.get("w_pos_grid", [1.0])]

    for w_pos in w_pos_grid:
        for obj in yaml_data.get("objectives", []):
            reg = obj["reg"]
            lam_grid = obj.get("lam_grid", [0.0])

            for lam in lam_grid:
                for oc in obj.get("optimizers", []):
                    opt = oc["name"]
                    lr_grid = oc.get("lr_grid", [0.01])
                    schedules = oc.get("schedule", ["fixed"])
                    batch_sizes = oc.get("batch_size", [256])
                    bt_conf = oc.get("backtracking", {})
                    enabled = bt_conf.get("enabled", [False])
                    alphas = [a for a in bt_conf.get("armijo_alpha", [None])
                              if a is not None] or [None]
                    betas = [b for b in bt_conf.get("armijo_beta", [None])
                             if b is not None] or [None]

                    if opt in NO_L1 and reg == "l1":
                        continue  # newton + l1 does not exist

                    # ---- fixed-step combos ----
                    if False in enabled:
                        if opt == "sgd":
                            for lr, sched, bs in itertools.product(
                                    lr_grid, schedules, batch_sizes):
                                yield emit(loss, w_pos, reg, lam, opt, lr,
                                           sched, bt=False, initial_lr=None,
                                           batch_size=bs, alpha=None,
                                           beta=None, epochs=epochs,
                                           epsilon=epsilon, patience=patience,
                                           seed=seed)
                        else:
                            for lr in lr_grid:
                                yield emit(loss, w_pos, reg, lam, opt, lr,
                                           "fixed", bt=False, initial_lr=None,
                                           batch_size=None, alpha=None,
                                           beta=None, epochs=epochs,
                                           epsilon=epsilon, patience=patience,
                                           seed=seed)

                    # ---- backtracking combos ----
                    if True in enabled and opt != "sgd":
                        if opt in ALPHA_BETA_BT:
                            combos = itertools.product(alphas, betas)
                        else:  # nag: Parabol majorization, beta only
                            combos = ((None, b) for b in betas)
                        for alpha, beta in combos:
                            yield emit(loss, w_pos, reg, lam, opt,
                                       lr=None, schedule="fixed", bt=True,
                                       initial_lr=1.0, batch_size=None,
                                       alpha=alpha, beta=beta, epochs=epochs,
                                       epsilon=epsilon, patience=patience,
                                       seed=seed)


def main():
    p = argparse.ArgumentParser(
        description="Expand a tuning YAML into a manifest TSV on stdout.")
    p.add_argument("config", help="path to configs/tune_{loss}.yaml")
    args = p.parse_args()

    with open(args.config, "r") as f:
        yaml_data = yaml.safe_load(f)

    print("#" + "\t".join(COLUMNS))
    n = 0
    for line in gen(yaml_data):
        print(line)
        n += 1
    print(f"# total_trials\t{n}", file=sys.stderr)


if __name__ == "__main__":
    main()
