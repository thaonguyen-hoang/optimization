#!/usr/bin/env bash
# run_train.sh — example single training run.
# Usage: ./run_train.sh
#
# Edit the flags below to try a different loss / optimizer / regularizer.
# Activate the env first:  conda activate optim
set -euo pipefail

DATA_DIR="data"
OUT_DIR="runs"

python -m scripts.train \
  --loss bce \
  --reg l2 \
  --lam 1e-2 \
  --optimizer newton \
  --backtracking \
  --epochs 50 \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT_DIR" \
  --eval-test \
  --save-figures \
  --hessian-spectrum

# A few more one-off examples (uncomment as needed):
# python -m scripts.train --loss weighted_bce --w-pos 6.0 --reg none --optimizer sgd --lr 1e-2 --epochs 50 --batch-size 256 --eval-test
# python -m scripts.train --loss squared_hinge --reg l1 --lam 1e-2 --optimizer accelerated_gd --lr 1e-2 --epochs 50
# python -m scripts.train --loss bce --reg none --optimizer gd --backtracking --epochs 50 --eval-test --save-figures
