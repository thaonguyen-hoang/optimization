# Diabetes BRFSS — Optimization for Logistic Regression

Optimization course project: train **logistic regression** `z = Xw + b`
to classify diabetes (BRFSS, imbalanced ~14% positive) and study how
different **optimizers**, **losses**, and **regularizers** behave. Everything
is implemented from scratch with NumPy (no sklearn / scipy).

## Repository layout

```
optimization/
├── data/                 # 3 BRFSS CSVs (gitignored — large)
│   ├── diabetes_binary_health_indicators_BRFSS2015.csv   # train
│   ├── diabetes_2021_val.csv                             # val
│   └── diabetes_2021_test.csv                            # test
├── src/                  # shared library
│   ├── data_utils.py     # load 3 splits, standardize (fit on train only)
│   ├── losses.py         # BCE, WeightedBCE, SquaredHinge (+Focal, unused)
│   ├── optimizers.py     # GD, NAG, Newton, SGD + Backtracking & Proximal logic
│   ├── regularizers.py   # None, L2 (gradient), L1 (proximal operator)
│   ├── metrics.py        # AUPRC, F1-minority, accuracy, AUROC (pure numpy)
│   ├── train.py          # generic training loop, per-iter logging, checkpoint
│   └── plotting.py       # convergence + metrics + Hessian-spectrum figures
├── scripts/
│   ├── train.py          # CLI: one run -> runs/<id>/{config,history,checkpoint,metrics}
│   └── expand_tune.py    # CLI: YAML grid -> manifest TSV (1 trial / line)
├── runs/                 # run artifacts (gitignored)
├── figures/              # generated figures (gitignored)
├── run_train.sh          # example single run
├── run_tune.sh           # YAML-driven tuning loop -> summary + best per objective
├── requirements.txt
└── README.md
```

## Setup

```bash
conda activate optim
pip install -r requirements.txt   # numpy, pandas, matplotlib
```

The three CSVs live under `data/` (gitignored). They are loaded directly —
there is no re-splitting step.

## Data note

The split is fixed by the files on disk:

- **train** = BRFSS 2015 (`diabetes_binary_health_indicators_BRFSS2015.csv`, ~229k rows after dedup)
- **val / test** = BRFSS 2021 (`diabetes_2021_val.csv`, `diabetes_2021_test.csv`)

Train (2015) does not overlap val/test (2021) — the year split prevents
leakage. Val and test are different row sets but share ~6.5k duplicate rows
(~5.5%); this is noted but not re-split, per the project decision.
Standardization (mean/std) is fit on **train only**.

## Workflow

### 1. Single run

```bash
python -m scripts.train --loss bce --reg l2 --lam 1e-2 \
  --optimizer newton --backtracking --epochs 50 --eval-test --save-figures
```

Or via the helper script:

```bash
./run_train.sh
```

Outputs land in `runs/<run_id>/`:

| File | Contents |
|---|---|
| `config.json` | full hyperparameters |
| `history.npz` | per-iter train loss / grad norm / wall time + per-epoch val loss & metrics (AUPRC, F1, accuracy, AUROC) |
| `checkpoints/best.npz` | `(w, b)` at best val AUPRC |
| `checkpoints/last.npz` | `(w, b)` at the final epoch |
| `metrics.json` | best val metrics (+ test metrics if `--eval-test`) |
| `best.json` | pointer to best checkpoint + epoch |
| `convergence.png` | loss / grad-norm / val-metrics panels (if `--save-figures`) |
| `hessian_eigs.npy` | Hessian spectrum (if `--hessian-spectrum`) |

### 2. Hyperparameter tuning

YAML-driven grid search. Each loss has a grid file `configs/tune_<loss>.yaml`
(see below for the current grids).

```bash
./run_tune.sh --config configs/tune_bce.yaml            # full grid, epochs from YAML
./run_tune.sh --config configs/tune_bce.yaml --epochs 50  # override epochs
./run_tune.sh --config configs/tune_bce.yaml --dry-run    # print trials, run nothing
```

Pipeline — each layer does exactly one thing:

1. `python -m scripts.expand_tune <yaml>` parses the grid and prints a
   **manifest TSV** (1 row = 1 valid trial). No training, no metrics.
2. `run_tune.sh` loops the manifest and launches
   `python -m scripts.train ...` per trial with an explicit
   `--run-id trial_<NNN>_...`, so every trial is a full run dir
   (`metrics.json`, `history.npz`, `config.json`, `convergence.png`, ...).
   Failed trials are recorded in `failed_trials.txt`; the loop continues.
3. Aggregation (jq + awk) produces `tune_<loss>_summary.tsv` (every trial +
   convergence stats) and `best_per_objective.tsv`: per objective `loss|reg`,
   the **fastest** (min `wall_time_to_best`) and **fewest-iters**
   (min `iters_to_best`) runs.

Output layout:

```
runs/tune_<loss>_<timestamp>/
├── manifest.tsv                # trial grid used
├── failed_trials.txt           # trial run_ids that errored (empty if none)
├── trials/trial_<NNN>_*/       # one full run dir per trial
├── tune_<loss>_summary.tsv
└── best_per_objective.tsv
```

Combination rules enforced by `expand_tune.py` (per optimizer / reg):

- **Smooth (none, L2):** GD, NAG, SGD, Newton with fixed `lr`, **and**
  backtracking variants (Armijo alpha×beta for GD/Newton, Parabol beta-only
  for NAG; `initial_lr=1`).
- **Non-smooth (L1):** GD (ISTA), NAG (FISTA), SGD (Proximal) — fixed `lr`
  and Proximal Backtracking (Parabol, beta-only). Newton is dropped for L1.
- SGD has no backtracking; mini-batch `batch_size` swept; `diminishing`
  schedule variant.

Selection criterion = **validation AUPRC** (imbalanced, minority-focused).
Convergence-efficiency comparisons use `wall_time_to_best` (fastest) and
`iters_to_best` (fewest iterations; note SGD counts batch-updates, so compare
cross-method with `epochs_to_best`).

## Math summary

- **Objective:** `F(w) = f(w) + r(w)`, with `f` = data loss on logits,
  `r` = regularizer.
- **Losses (convex, smooth):** BCE, Weighted BCE, Squared Hinge.
- **Regularizers:** L2 smooth gradient; **L1 via Proximal Operator** (Soft-Thresholding). Subgradient method is deliberately NOT used in favor of ISTA/FISTA.
- **Optimizers:**
  - Fixed step size: GD, SGD (mini-batch), NAG (Nesterov Accelerated Gradient), Newton (damped) — `lr` tuned.
  - Backtracking: GD (Armijo), NAG (Parabol), Newton (Armijo), ISTA/FISTA (Proximal Backtracking with energy-preserving $s_k$).
  - SGD features Diminishing step size schedules ($1/\sqrt{k}$).
  - Newton is never combined with L1 (non-smooth Hessian math invalidity).

## Reproducibility / fairness

- Data splits are fixed files; no per-run re-splitting.
- Standardization fits on train only.
- Same optimizer/loss/regularizer classes and same grids for every loss.
- Best model chosen by val AUPRC; test is evaluated only with the final
  best config (`--eval-test`), never used for selection.
