# Optimization Project — Experiment Setup & Execution Plan

## Overview

Binary classification (diabetes prediction) using custom logistic regression implemented in **pure NumPy**, benchmarked against scikit-learn. Experiments sweep all combinations of loss function, optimizer, regularization, and step size strategy. Results are saved to JSON files and visualized with Matplotlib.

---

## Design Decisions Summary

| Dimension | Decision |
|---|---|
| Framework | Pure NumPy (+ scikit-learn as reference) |
| Project layout | Python scripts + YAML configs |
| Epoch definition | 1 epoch = 1 full pass over training data |
| SGD variant | Both mini-batch (configurable size) and single-sample (batch=1) |
| Backtracking | Armijo line search (reduce by factor β until sufficient decrease) |
| L-BFGS | `scipy.optimize.minimize(method='L-BFGS-B')` with manual gradient |
| Newton | Custom (explicit Hessian + damping εI for stability) |
| L1 optimizer | Proximal gradient (GD/SGD/NAG variants only) |
| Learning rate grid | Derived from Lipschitz constant L; sweep `c/L` where `c` from YAML |
| Hyperparam search | Grid search defined in YAML config |
| Within-run stopping | Fixed `max_epochs` + tol_obj early stop (Δloss ≤ 1e-6) |
| Between-config stopping | Patience-based on **validation AUROC** |
| Test evaluation | Best config per loss (by val AUROC) → one final test set pass |
| Results format | Per-run JSON: `{loss}_{optimizer}_{reg}_{step_type}_{params}.json` |
| Plotting | Matplotlib (seaborn style) |
| Hessian spectrum | At final trained weights: L2 vs no-reg, eigenvalue histogram + sorted curve |

---

## Proposed Project Structure

```
code-v2/
├── data/
│   └── processed/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── configs/
│   ├── bce.yaml
│   ├── weighted_bce.yaml
│   ├── squared_hinge.yaml
│   └── focal.yaml
├── src/
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── bce.py               # BCE: loss, grad, hessian
│   │   ├── weighted_bce.py      # Weighted BCE: loss, grad (w_pos, w_neg)
│   │   ├── squared_hinge.py     # Squared hinge: loss, grad
│   │   └── focal.py             # Focal loss: loss, grad (γ, α)
│   ├── optimizers/
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract Optimizer base class
│   │   ├── gd.py                # Gradient Descent (standard + proximal for L1)
│   │   ├── sgd.py               # SGD (mini-batch + single-sample; proximal for L1)
│   │   ├── nag.py               # Nesterov Accelerated Gradient (+ proximal for L1)
│   │   ├── newton.py            # Newton (explicit Hessian + εI damping)
│   │   └── lbfgs.py             # L-BFGS via scipy (internal line search for backtrack)
│   ├── regularizers/
│   │   ├── __init__.py
│   │   ├── l1.py                # L1: penalty value + proximal operator
│   │   └── l2.py                # L2: penalty value + gradient term
│   ├── step_size/
│   │   ├── __init__.py
│   │   ├── fixed.py             # Fixed LR (value = c / L, computed at runtime)
│   │   ├── armijo.py            # Armijo backtracking line search
│   │   └── lipschitz.py         # Compute L = λ_max(X^T X)/4 [+ λ for L2]
│   ├── model.py                 # LogisticRegression class (fit/predict/proba)
│   ├── metrics.py               # recall, precision, f1, auroc, auprc, etc.
│   └── utils.py                 # Data loading, seeding, imbalance ratio, logging
├── results/                     # Auto-created; per-run JSON files
├── plots/                       # Auto-created; saved .png figures
├── runner.py                    # Main experiment runner (reads config, sweeps grid)
├── evaluate_best.py             # Final test-set evaluation of best-per-loss configs
├── plot_results.py              # Generates all plots from results/ JSON files
├── plot_hessian.py              # Hessian eigenvalue spectrum plot
└── requirements.txt
```

---

## Experiment Matrix

### Valid optimizer × regularization combinations

| Optimizer | None | L2 (Ridge) | L1 (Lasso) |
|---|---|---|---|
| GD | ✅ fixed + backtrack | ✅ fixed + backtrack | ✅ **proximal GD**, fixed + backtrack |
| SGD | ✅ fixed + backtrack† | ✅ fixed + backtrack† | ✅ **proximal SGD**, fixed + backtrack† |
| NAG | ✅ fixed + backtrack | ✅ fixed + backtrack | ✅ **proximal NAG**, fixed + backtrack |
| Newton | ✅ fixed + backtrack | ✅ fixed + backtrack | ❌ skip (non-smooth) |
| L-BFGS | ✅ fixed + backtrack | ✅ fixed + backtrack | ❌ skip (non-smooth) |

> **† SGD + backtracking** — run it to demonstrate the problem, explain in write-up why backtracking is ill-defined for SGD (stochastic gradients make the Armijo condition non-deterministic; line search evaluates the loss on the full batch but the gradient was computed on a mini-batch, breaking the theoretical guarantee).

> **L1 = non-smooth** — standard gradient steps are invalid at w=0. GD/SGD/NAG all switch to their **proximal** variants: gradient step on the smooth part (loss + L2 if combined), then apply the soft-thresholding proximal operator for L1.

---

## Learning Rate Strategy (Lipschitz-Guided)

The Lipschitz constant **L** of the gradient bounds the curvature of the loss and gives a principled upper bound on the step size. It is computed **once per experiment** from the training data (and λ if L2 is used) before the grid sweep.

### Computation

```
L = λ_max(X^T X) / 4       # for BCE / squared hinge / focal (logistic gradient)
L = λ_max(X^T X) / 4 + λ   # add regularization strength λ when L2 is used
```

> For Weighted BCE, multiply by the dominant class weight: `L *= max(w_pos, w_neg)`.
> λ_max is computed via power iteration or `np.linalg.eigvalsh` on X^T X.

### Fixed step size grid (c/L)

| Optimizer | c sweep in YAML | Effective LR |
|---|---|---|
| GD | `c ∈ [0.1, 0.5, 1.0, 1.5, 1.9]` | η = c / L |
| NAG | `c ∈ [0.1, 0.5, 1.0, 1.5, 1.9]` | η = c / L |
| SGD | `c ∈ [1e-4, 1e-3, 0.01, 0.05, 0.1]` | η = c / L (+ decay) |

> **SGD decay schedule**: η_t = η_0 / (1 + decay_rate × t) where `decay_rate` is a YAML param.
> Smaller c range for SGD because stochastic noise already destabilizes large steps.

### Backtracking (Armijo)

Armijo ignores L entirely. Initial step is always `α_0 = 1.0`, reduced by factor `β` until:
```
f(w - α∇f) ≤ f(w) - c_armijo · α · ||∇f||²
```

| Optimizer | Backtracking |
|---|---|
| GD / NAG | Standard Armijo on full-batch gradient |
| SGD | Run Armijo (for demonstration), document why it is theoretically unsound |
| Newton | **Internal**: uses the Hessian to determine step; Armijo confirms sufficient decrease |
| L-BFGS | **Internal** to `scipy`: scipy handles its own line search; no external Armijo loop |

---

## Stopping Criteria (Two-Level)

### Level 1 — Within a single run
- Run for `max_epochs` (default: 1000)
- Stop early if `|loss(t) - loss(t-1)| ≤ tol_obj` (default: `1e-6`) for `patience_inner` consecutive steps

### Level 2 — Across hyperparameter configs (per loss)
- After each epoch, compute **validation AUROC**
- Stop a config early (patience_outer = 20 epochs with no improvement)
- Best config for a loss = highest peak val AUROC over the entire grid

---

## Hyperparameter Config (YAML schema — `configs/focal.yaml` as example)

```yaml
loss: focal
max_epochs: 1000
tol_obj: 1e-6
patience_inner: 5
patience_outer: 20
seed: 42

optimizers:
  - gd
  - sgd
  - nag
  - newton
  - lbfgs

regularizations:
  - none
  - l2
  # L1 not applicable for focal (Newton/LBFGS skip; GD/SGD/NAG use proximal)

# ── Step size ──────────────────────────────────────────────────────────────
step_size:
  fixed:
    # c values for GD/NAG; SGD uses its own range below
    c_gd_nag: [0.1, 0.5, 1.0, 1.5, 1.9]
    c_sgd:    [1e-4, 1e-3, 0.01, 0.05, 0.1]
    sgd_decay_rate: [0.0, 0.001, 0.01]   # 0.0 = no decay
  backtracking:
    alpha_init: 1.0
    beta: 0.5
    c_armijo: 1e-4   # Armijo sufficient decrease constant

# ── Regularization ─────────────────────────────────────────────────────────
regularization:
  lambda: [1e-4, 1e-3, 0.01, 0.1]

# ── SGD batch sizes ────────────────────────────────────────────────────────
sgd:
  batch_sizes: [1, 32, 64]   # 1 = true stochastic SGD

# ── Newton damping ─────────────────────────────────────────────────────────
newton:
  epsilon_damp: 1e-6   # Hessian ← Hessian + ε * I to ensure invertibility

# ── Focal loss hyperparams ─────────────────────────────────────────────────
# α and γ are anti-correlated: as γ↑ (more dynamic balancing), α→0.5 (less static balancing)
# Sweep is defined as explicit (γ, α) pairs, not a full cross-product
focal:
  gamma_alpha_pairs:
    - [0.5, 0.75]   # low focusing → more static balancing needed
    - [1.0, 0.65]
    - [2.0, 0.50]   # paper default (Lin et al. 2017)
    - [3.0, 0.40]
    - [5.0, 0.25]   # high focusing → minimal static balancing needed
```

> For `configs/bce.yaml`: omit `focal` section, keep all other sections.
> For `configs/weighted_bce.yaml`: replace focal section with the weighted-BCE section below.

### Weighted BCE hyperparam block

```yaml
# Class imbalance ratio R = N_neg / N_pos is computed dynamically from train.csv
# YAML defines a scaling multiplier c; runtime sets w_pos = c * R
weighted_bce:
  c_scale: [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]  # w_pos = c_scale * R
  # w_neg is always fixed at 1.0
```

---

## Results JSON Schema (per run)

```json
{
  "run_id": "focal_gd_l2_fixed_c1.0_lambda0.001_g2.0_a0.5",
  "loss_fn": "focal",
  "optimizer": "gd",
  "regularization": "l2",
  "step_size_type": "fixed",
  "hyperparams": {
    "c": 1.0, "L": 42.7, "lr": 0.0234,
    "lambda": 0.001,
    "gamma": 2.0, "alpha": 0.5
  },
  "epochs_run": 512,
  "stopped_early": true,
  "stop_reason": "tol_obj",
  "train_loss_curve": [0.693, 0.551, "..."],
  "val_loss_curve":   [0.690, 0.548, "..."],
  "val_auroc_curve":  [0.501, 0.734, "..."],
  "final_metrics": {
    "train": { "loss": 0.298, "auroc": 0.892 },
    "val":   { "loss": 0.305, "auroc": 0.886 }
  }
}
```

---

## Plots to Generate

| Plot | Script | Description |
|---|---|---|
| Loss curve per optimizer | `plot_results.py` | Per loss fn: subplots by reg type, lines = optimizers (fixed vs backtrack) |
| Val AUROC curve per optimizer | `plot_results.py` | Same structure; shows convergence quality across optimizers |
| Final test metrics (best configs) | `evaluate_best.py` | Bar chart: recall, precision, F1, AUROC, AUPRC — one bar per loss |
| Hessian eigenvalue spectrum | `plot_hessian.py` | Sorted eigenvalue curve + histogram at final weights: L2 vs no-reg |

---

## Verification Plan

1. **Gradient check**: finite-difference verification for all 4 losses and all regularizers on a toy dataset
2. **Proximal operator check**: verify soft-thresholding drives small weights to exactly zero
3. **Smoke test**: 1 epoch of all valid combinations — confirm no crashes
4. **Sanity check**: custom logistic (BCE + GD, no reg) vs `sklearn.linear_model.LogisticRegression` — weights should match closely
5. **Convergence check**: loss curves monotonically non-increasing for GD + fixed step (c < 2) on convex losses


---

# Revised Experiment Design — Two-Phase Structured Ablation

## Guiding Principle

> **Change one thing at a time.** Every experiment group has a fixed baseline and exactly one varying axis. Every run must answer a specific question or demonstrate a specific effect.

---

## Phase 1 — Mechanics Ablation (anchor: BCE only)

### Shared anchor for Phase 1
| Setting | Value |
|---|---|
| Loss | BCE |
| Seed | 42 |
| Max epochs | 500 |
| Eval metric | Val AUROC |

---

### Group A — Optimizer Comparison
**Question**: Which optimizer converges fastest, in epochs and wall-clock time?

**Fixed**: BCE, L2 (λ=0.01), step size strategy per optimizer (see hyperparam section)

| ID | Optimizer | Batch | Step |
|---|---|---|---|
| A1 | GD | full | fixed c=1.0/L |
| A2 | NAG (μ=0.9) | full | fixed c=1.0/L |
| A3 | SGD | 64 | fixed c=0.01/L |
| A4 | Newton (H+εI) | full | backtracking (damped) |
| A5 | L-BFGS | full | scipy Wolfe (1 step/epoch) |

**Plots**: loss vs epoch AND loss vs wall-clock time (side by side — key comparison)
**Story**: Newton/L-BFGS win per epoch; GD/NAG win per second for large n

---

### Group B — Step Size Strategy
**Question**: When does backtracking help? How does the choice interact with each optimizer family? Why is backtracking meaningless for SGD?

**Fixed**: BCE, L2 (λ=0.01)

| ID | Optimizer | Step | Key insight demonstrated |
|---|---|---|---|
| B1 | GD | fixed c=0.5/L | Conservative — slow but stable |
| B2 | GD | fixed c=1.0/L | Theoretically optimal (η = 1/L) |
| B3 | GD | fixed c=1.9/L | Aggressive — near divergence boundary (c < 2 guarantees convergence) |
| B4 | GD | backtracking | Adaptive — auto-finds safe step |
| B5 | NAG | fixed c=1.0/L | Acceleration with correct step |
| B6 | NAG | backtracking | Adaptive acceleration |
| B7 | Newton | fixed c=1.0 | Pure Newton step (works near solution, may diverge far) |
| B8 | Newton | backtracking | **Damped Newton** — globally convergent, classic algorithm |
| B9 | L-BFGS | 1 step/epoch | Quasi-Newton, one update per "epoch" |
| B10 | L-BFGS | full convergence | L-BFGS run to convergence (scipy Wolfe) |
| B11 | SGD (bs=64) | fixed | Correct use of SGD |
| B12 | SGD (bs=64) | backtracking | **Demonstration**: Armijo on mini-batch — erratic behavior expected |

**Key comparisons**:
- B1/B2/B3/B4: shows the stability range `c ∈ (0, 2)` for GD and how backtracking auto-adapts
- B7 vs B8: pure Newton (unstable far from solution) vs damped Newton (globally convergent)
- B9 vs B10: one L-BFGS step per epoch vs full convergence — tradeoff between fairness of epoch comparison and practical use
- B11 vs B12: the SGD+backtracking failure mode

**Note on Newton + fixed step (B7)**: When initializing from w=0, the Newton step at early iterations can be large (Hessian near-flat → large inverse). A fixed step of 1.0 may cause divergence, which is itself an interesting observation — motivating the need for damping (B8).

---

### Group C — Regularization Effect
**Question**: What does regularization do to convergence, generalization, and the loss landscape?

**Fixed**: BCE, GD, fixed step (c=1.0/L)

| ID | Reg | λ | Notes |
|---|---|---|---|
| C1 | None | — | Baseline — may overfit |
| C2 | L2 | 1e-4 | Weak ridge |
| C3 | L2 | 1e-2 | Moderate ridge |
| C4 | L2 | 0.1 | Strong ridge |
| C5 | L1 (proximal GD) | 1e-4 | Weak lasso |
| C6 | L1 (proximal GD) | 1e-2 | Moderate lasso |
| C7 | L1 (proximal GD) | 0.1 | Strong lasso → sparsity |

**Additional output from C**: Hessian eigenvalue spectrum for C1 vs C3
**Plots**: train/val loss gap (generalization), sparsity (# zero weights vs λ), Hessian spectrum
**Story**: L2 shifts λ_min up → better conditioning → faster convergence; L1 drives weights to zero (sparsity); no-reg overfits

> **Phase 1 output**: Best (optimizer, λ, step) from A+C = `best_bce_setup` → used in Phase 2a as the controlled baseline

---

## Phase 2 — Per-Loss Targeted Sweep (all 4 losses)

### Sub-goal 2a: Controlled Comparison (isolate loss function effect)

**Fix**: GD + L2 (λ=0.01) + fixed step for ALL losses. Only loss function and its specific HPs vary.

| ID | Loss | Loss-specific HP |
|---|---|---|
| D_ctrl_1 | BCE | — |
| D_ctrl_2 | Weighted BCE | c_scale ∈ {0.75, 1.0, 1.5} → best by val AUROC |
| D_ctrl_3 | Squared Hinge | — |
| D_ctrl_4 | Focal | (γ,α) ∈ {(0.5,0.75),(2.0,0.5),(5.0,0.25)} → best by val AUROC |

**Story**: Differences in val AUROC / AUPRC / recall are attributable to the loss function alone (optimizer is controlled)

---

### Sub-goal 2b: Best-Effort Per Loss (for final test comparison)

For each loss, run a compact optimizer+reg+loss-HP sweep to find its best setup.

**For each of {weighted BCE, squared hinge, focal}**:

| Slot | Candidates | Fixed |
|---|---|---|
| Step 1: Optimizer | GD, NAG, L-BFGS | L2 λ=0.01, fixed step |
| Step 2: Regularization | none, L2 {0.01, 0.1} | Best optimizer from Step 1 |
| Step 3: Loss HPs | loss-specific grid | Best optimizer + reg |

Results feed `evaluate_best.py` → final test table.

---

## Hyperparameter Justification

Every value in these experiments has a principled reason. This section documents them for use in the write-up.

---

### Learning rate c (fixed step: η = c/L)

**Theoretical basis**: For gradient descent on an L-smooth function, the convergence guarantee requires:
```
η ≤ 1/L   →   c ≤ 1
```
The exact bound is `η < 2/L` for convergence (oscillation possible at η = 2/L). So:

| c value | Meaning | In write-up |
|---|---|---|
| 0.5 | Conservative (half the safe maximum) | Safe, slow |
| 1.0 | η = 1/L — theoretically optimal for GD | Best fixed rate per theory |
| 1.9 | Near the divergence boundary | Demonstrates instability risk |

**For SGD**: The stochastic gradient has variance σ² that adds an irreducible noise floor. To ensure that `E[loss]` decreases, the effective step must be much smaller. Standard result: SGD needs `η ∝ 1/(L + σ/√t)` — we use a smaller c range {1e-4, 1e-3, 0.01, 0.05, 0.1} to reflect this, with optional decay schedule.

---

### Armijo parameters (β, c_armijo)

**Source**: Nocedal & Wright, "Numerical Optimization" (2nd ed., 2006), Chapter 3.

| Parameter | Value | Justification |
|---|---|---|
| α₀ (initial step) | 1.0 | Standard: for Newton-like methods, α=1 is the natural full step |
| β (reduction factor) | 0.5 | Halving per iteration is standard; geometric with base 2 |
| c_armijo (sufficient decrease) | 1e-4 | Small enough to accept most reasonable steps, per Nocedal §3.1 |
| max_iter | 50 | 50 halvings → α_min ≈ 10⁻¹⁵, well below machine precision |

---

### Regularization λ range {1e-4, 1e-2, 0.1}

**Basis**: Log-scale search is standard practice (equivalent to uniform search in log space). The useful range for logistic regression is typically where the regularization is neither negligible nor dominant.

- **Too small (λ < 1e-5)**: indistinguishable from no regularization
- **Too large (λ > 1)**: solution collapses toward w=0; val AUROC ≈ 0.5 (random classifier)
- **Target range**: λ ∈ {1e-4, 1e-2, 0.1} covers 3 orders of magnitude, sufficient to show the effect

**L2 specifically**: The Hessian eigenvalues are shifted up by λ (H_damp = H_loss + λI). At λ=0.01, the minimum eigenvalue is bounded below by 0.01, improving the condition number κ = λ_max/λ_min significantly.

---

### NAG momentum μ = 0.9

**Source**: Sutskever et al. (2013), "On the importance of initialization and momentum in deep learning." μ=0.9 is the standard default for Nesterov momentum and matches the classical accelerated gradient theory's sequence μ_t = (t-1)/(t+2) → 1 as t → ∞.

---

### Newton damping ε = 1e-6

**Basis**: ε should be small enough not to distort the curvature information but large enough to guarantee H_damp = H + εI is positive definite. Since H is a sum of outer products scaled by σ(1-σ) ∈ (0, 0.25), its smallest eigenvalue can be near zero early in training (when σ ≈ 0 or 1 for all samples). ε=1e-6 is a standard choice (Tikhonov regularization in the numerical linear algebra literature).

---

### Weighted BCE: c_scale ∈ {0.5, 0.75, 1.0, 1.25, 1.5, 2.0}

**Basis**: The theoretically optimal weight for balanced cross-entropy is `w_pos = R = N_neg/N_pos`. This perfectly compensates frequency imbalance. We sweep around R:
- c_scale < 1: under-weighting positives (closer to standard BCE)
- c_scale = 1: exact frequency compensation
- c_scale > 1: over-weighting positives (improves recall, hurts precision)

The range [0.5, 2.0] covers ½R to 2R — sufficient to show the recall/precision tradeoff.

---

### Focal loss: anti-correlated (γ, α) pairs

**Basis**: Lin et al. (2017) "Focal Loss for Dense Object Detection."

- **γ** (focusing): down-weights easy examples. γ=0 → standard BCE. As γ increases, the loss increasingly focuses on hard/misclassified examples.
- **α** (static balancing): should be reduced as γ increases, because γ already provides dynamic balancing. Using a full γ×α grid would include dominated combinations (e.g., γ=5, α=0.9 — double-counting imbalance correction).

| (γ, α) | Rationale |
|---|---|
| (0.5, 0.75) | Low focusing → strong static weight needed |
| (1.0, 0.65) | — |
| (2.0, 0.50) | Paper default (Lin et al.) |
| (3.0, 0.40) | — |
| (5.0, 0.25) | High focusing → minimal static weight needed |

---

### SGD batch sizes {1, 16, 64, full}

**Basis**: Powers of 2 are standard in the literature (cache alignment, GPU efficiency). The range covers:
- bs=1: true stochastic — maximum variance, minimum per-step cost
- bs=64: mini-batch — practical tradeoff, widely used
- full: deterministic GD (limit case — included in Group A as A1)

bs=16 added in Group E to better show the transition between stochastic and mini-batch regimes.

---

## Total Run Count

| Phase / Group | Runs |
|---|---|
| A (optimizer) | 5 |
| B (step size) | 12 |
| C (regularization) | 7 |
| Phase 2a (controlled, 4 losses) | ~10 |
| Phase 2b (best-effort, 3 non-BCE losses) | ~30 |
| Group E (batch size, BCE only) | 5 |
| **Total** | **~70** |

## After running

Now that the experiment pipeline is fully designed and implemented with the 8 explicit configs, here is exactly what comes next once you run those configurations on your machine (or cluster):

### 1. Final Evaluation on the Held-Out Test Set
After all groups finish, you'll have a `results/` folder filled with subdirectories (`A`, `B`, `C`, etc.). We will run the evaluation script to find the absolute best setup for each loss function and evaluate it on the untouched test set:
```bash
python evaluate_best.py --data_dir data/processed --results_dir results
```
*Note: I just proactively updated `evaluate_best.py` and `plot_results.py` to recursively search the new `results/<group>/` directory structure, so they are ready to go.*

This script will output a final comparison table showing how `BCE`, `Weighted BCE`, `Squared Hinge`, and `Focal` perform against each other in terms of F1, Recall, Precision, AUROC, and AUPRC on the test set.

### 2. Generate the Visualizations
We will generate the specific plots needed for the thesis write-up. We will likely write small, targeted plotting scripts for each group (or update `plot_results.py`) to generate:
- **Group A (Optimizer)**: `Loss vs Epoch` and `Loss vs Wall-Clock Time` side-by-side to show that while Newton/L-BFGS win per epoch, GD/NAG win per second.
- **Group B (Step Size)**: Overlaid loss curves showing the GD stability range ($c \in 0.5, 1.0, 1.9$) and a plot contrasting correct SGD with the erratic SGD+backtracking failure mode.
- **Group C (Regularization)**: We need to write/run the `plot_hessian.py` script to generate the eigenvalue spectrum of the Hessian (showing how L2 shifts the minimum eigenvalue up) and a plot showing weight sparsity vs L1 $\lambda$.
- **Group D (Loss Functions)**: A single controlled comparison plot showing the loss curves of the 4 different loss functions when using the exact same optimizer.

### 3. Thesis Write-Up (The "Results and Discussion" Section)
With the plots and the final test-set table in hand, the next step is transferring the findings into your thesis. The structure of the `implementation_plan.md` maps directly to your chapter layout:
1. **Hyperparameter Selection**: You can essentially copy the "Hyperparameter Justification" section from the plan into your methodology.
2. **Mechanics of Optimization**: Sections discussing Optimizer choice, Step Size stability, and Regularization effects (with their respective plots).
3. **Loss Function Comparison**: A section discussing the controlled vs. best-effort performance of the loss functions on the highly imbalanced dataset.