#!/usr/bin/env bash
# run_tune.sh — run hyperparameter tuning for all three losses.
# Usage: ./run_tune.sh
#
# Each loss produces runs/tune_<loss>_summary.csv and runs/best_<loss>.json.
# Activate the env first:  conda activate optim
set -euo pipefail

DATA_DIR="data"
OUT_DIR="runs"
TUNE_EPOCHS=${TUNE_EPOCHS:-30}
BATCH=${BATCH:-256}

for LOSS in bce weighted_bce squared_hinge; do
  echo "================ TUNING: $LOSS ================"
  python -m scripts.tune \
    --loss "$LOSS" \
    --tune-epochs "$TUNE_EPOCHS" \
    --batch-size "$BATCH" \
    --data-dir "$DATA_DIR" \
    --out-dir "$OUT_DIR"
done

echo
echo "Done. Summaries:"
ls -1 "$OUT_DIR"/tune_*_summary.csv "$OUT_DIR"/best_*.json 2>/dev/null

# Optional: merge the three best configs into a final comparison table.
python - <<PY
import json, glob, os
import pandas as pd
rows=[]
for f in sorted(glob.glob("runs/best_*.json")):
    cfg=json.load(open(f)); cfg["source"]=os.path.basename(f); rows.append(cfg)
if rows:
    pd.DataFrame(rows).to_csv("runs/final_comparison_table.csv", index=False)
    print("wrote runs/final_comparison_table.csv")
PY
