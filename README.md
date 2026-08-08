# Diabetes BRFSS — Loss/Optimizer/Regularizer Ablation Scaffold

Shared codebase for the 3-person split: each person owns one loss
function end-to-end (optimizer tuning → regularizer tuning → final
test-set evaluation), using identical data, splits, model, optimizer
set, and metrics — so the three results merge into one fair comparison
table at the end.

## File map

| File | Owner | Purpose |
|---|---|---|
| `data_utils.py` | shared, frozen | loading, merging train years, cleaning, continuous-only standardization |
| `make_shared_splits.py` | run once, together | merges the 3 train-year files + takes the pre-split val/test files, writes frozen `train.csv`/`val.csv`/`test.csv` |
| `losses.py` | shared, frozen | BCE, weighted BCE, focal, squared hinge (value + gradient + Hessian diagonal) |
| `optimizers.py` | shared, frozen | GD, SGD, SGD+Momentum (incl. true Nesterov), Adam, Newton, GD+backtracking + shared `LR_GRID` |
| `regularizers.py` | shared, frozen | None, L2, L1 (proximal), Elastic Net |
| `train.py` | shared, frozen | training loop wiring loss+optimizer+regularizer together; Hessian eigenvalue helper |
| `metrics.py` | shared, frozen | accuracy/precision/recall/F1/AUROC/AUPRC, no sklearn dependency |
| `plotting.py` | shared, frozen | convergence curves, multi-run overlays, confusion matrix, Hessian spectrum, LR-sensitivity |
| `person_template.py` | **copy per person** | the actual ablation pipeline (Stage 2b → 1 → 3 → final) |
| `merge_results.py` | run once, together, at the end | combines the 3 `*_final_row.csv` into `final_comparison_table.csv` |

## Workflow

**Step 0 (group, together, before anyone forks off):**

Split scheme: **train** = first 3 survey years merged into one file; **val**/**test** = the last survey year, already split in half into two separate files (given as-is, not re-split here).

```bash
python make_shared_splits.py \
    --train_years brfss_year1.csv brfss_year2.csv brfss_year3.csv \
    --val brfss_lastyear_val.csv \
    --test brfss_lastyear_test.csv \
    --out_dir ./data/processed/
```
This writes `data/processed/{train,val,test}.csv`, identical for everyone.
`person_template.py` already points `SHARED_SPLITS_DIR` there — commit/share
this folder, do not regenerate it per-person.

Only **continuous** features (`BMI`, `MentHlth`, `PhysHlth` by default —
edit `CONTINUOUS_COLS` in `data_utils.py` once, as a group, if your
schema differs) get standardized; binary 0/1 and ordinal survey-scale
columns are left on their original scale.

**Step 1 (each person, in parallel):**
Copy `person_template.py` → e.g. `person1_bce.py`. Edit only the
`CONFIG` section at the top (`PERSON_NAME`, `ASSIGNED_LOSS_NAME`) and,
if your loss has extra hyperparameters (weighted BCE's class weights,
focal's alpha/gamma), fill in `loss_hparam_candidates`. Run it. It
walks:

1. loss hyperparameter pre-sweep (skip if plain BCE/hinge)
2. optimizer ablation with its own LR sweep per optimizer → picks `opt*`
3. regularizer ablation with its own λ sweep per regularizer → picks `reg*`
4. final held-out test evaluation with the fully tuned setup
5. writes `{person}_final_row.csv` + convergence/confusion/Hessian figures

**Step 2 (group, together, at the end):**
```bash
python merge_results.py person1_final_row.csv person2_final_row.csv person3_final_row.csv
```
Produces `final_comparison_table.csv` — this is the "best setup"
table for your report.

## Suggested loss assignment

- **Person 1 — BCE**: baseline, convex, smooth.
- **Person 2 — Weighted BCE**: convex, smooth, cost-sensitive framing (tune the class-weight ratio in the pre-sweep).
- **Person 3 — Focal loss** (non-convex contrast) *or* **Squared hinge** (stays fully convex) — pick based on whether your course wants the convex-vs-non-convex story or a fully-convex comparison.

## Rules to keep the comparison fair

1. Never re-run `split_data` yourself — always load from `data/processed/`.
2. Never hand-pick a shared learning rate across optimizers — always sweep `LR_GRID` per optimizer and take each optimizer's own best.
3. Same rule for regularizer λ.
4. Compare across losses using **task metrics** (AUPRC, F1-minority), not raw loss values — they aren't on the same scale across different loss functions.
5. If you touch a "shared, frozen" file because you found a real bug, tell the group immediately — everyone's already-run numbers may need a re-run.

## Notes on the implementation

- Losses/optimizers/regularizers work on **logits** `z = Xw + b`, not
  probabilities, for numerical stability — see the docstring at the
  top of `losses.py`.
- All loss gradients were verified against numerical differentiation
  during scaffold testing.
- `train.py::hessian_eigs_logreg` gives you the eigenvalue spectrum of
  the (optionally L2-regularized) Hessian at a given `w` — use this
  for the "L2 improves conditioning" plot from your report discussion.
- L1's non-smoothness is handled via **proximal gradient (ISTA)**, not
  plain subgradient descent — see `regularizers.py::L1Reg.prox`.
- **Newton** is included only for the convex smooth losses (BCE, weighted
  BCE, squared hinge) — `losses.py` exposes an exact Hessian diagonal for
  those; it is skipped automatically for focal (non-convex). Newton runs
  full-batch with a damping `lr` (default 1.0) and an adaptive ridge.
- `sgd_momentum` with `nesterov=True` is **true Nesterov accelerated
  gradient**: `train.py` evaluates the gradient at the extrapolated point
  `θ + μv` (via `optimizer.lookahead`). With `nesterov=False` it is plain
  heavy-ball momentum.
- `gd_backtracking` is gradient descent with a backtracking (Armijo)
  line search and warm-started step size; it also handles the proximal
  (L1) step internally.
- `optimizers.LR_GRID` and the λ grid in `person_template.py` are
  deliberately small for a first pass; widen them if you have compute
  budget left, but keep the same grid across all three people.

> **Shared-file note (rule 5):** `train.py`, `optimizers.py`,
> `regularizers.py`, `losses.py`, and `metrics.py` were recently changed
> as a group (true NAG, Newton, backtracking, diagonal Hessians, AUROC
> tie-midranks). Everyone should **re-run their pipeline** before merging —
> any numbers produced with the previous scaffold are superseded.
