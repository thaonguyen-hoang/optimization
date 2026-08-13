#!/usr/bin/env bash
# run_tune.sh — YAML-driven grid-search tuning loop.
#
# Pipeline (mỗi lớp làm 1 việc, xem TUNING_PLAN.md):
#   1. scripts/expand_tune.py: YAML -> manifest TSV (1 dòng / 1 tổ hợp hợp lệ).
#   2. Loop manifest, mỗi trial chạy `python -m scripts.train ...` độc lập
#      -> runs/tune_<loss>_<ts>/trials/trial_<NNN>_*/ (artifacts đầy đủ).
#   3. Tổng hợp: tune_<loss>_summary.tsv + best_per_objective.tsv
#      (per objective loss|reg: fastest theo wall_time_to_best,
#       fewest iters theo iters_to_best).
#
# Usage:
#   ./run_tune.sh --config configs/tune_bce.yaml
#   ./run_tune.sh --config configs/tune_bce.yaml --dry-run
#   ./run_tune.sh --config configs/tune_bce.yaml --epochs 50 --out-dir runs
#
# Requires: python (pyyaml), jq. Env: conda activate optim
set -uo pipefail

DATA_DIR="data"
OUT_DIR="runs"
CONFIG=""
EPOCHS=""
DRY_RUN=0

usage() {
    echo "usage: $0 --config <yaml> [--dry-run] [--epochs N] [--out-dir DIR] [--data-dir DIR]" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)   CONFIG="${2:?missing value}"; shift 2 ;;
        --dry-run)  DRY_RUN=1; shift ;;
        --epochs)   EPOCHS="${2:?missing value}"; shift 2 ;;
        --out-dir)  OUT_DIR="${2:?missing value}"; shift 2 ;;
        --data-dir) DATA_DIR="${2:?missing value}"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; usage ;;
    esac
done
[[ -n "$CONFIG" ]] || usage

# ---------------------------------------------------------------- manifest
MANIFEST=$(mktemp)
trap 'rm -f "$MANIFEST"' EXIT

echo "== expand_tune: $CONFIG"
python -m scripts.expand_tune "$CONFIG" > "$MANIFEST" || {
    echo "expand_tune failed for $CONFIG" >&2
    exit 1
}
mapfile -t TRIAL_ROWS < <(grep -v '^#' "$MANIFEST")
N_TOTAL=${#TRIAL_ROWS[@]}
LOSS="${TRIAL_ROWS[0]%%$'\t'*}"
echo "== loss=$LOSS  trials=$N_TOTAL"

# parse_row: read 1 manifest line into indexed array F, require exactly 16
# tab-separated fields (see expand_tune.py COLUMNS). Avoids misalignment
# turning a malformed row into garbage CLI args.
parse_row() {
    IFS=$'\t' read -ra F <<< "$1"
    [[ ${#F[@]} -eq 16 ]]
}

row_fields() {
    loss="${F[0]}"; w_pos="${F[1]}"; reg="${F[2]}"; lam="${F[3]}"; opt="${F[4]}"
    lr="${F[5]}"; schedule="${F[6]}"; bt="${F[7]}"; initial_lr="${F[8]}"; bs="${F[9]}"
    alpha="${F[10]}"; beta="${F[11]}"; epochs="${F[12]}"; loss_eps="${F[13]}"
    patience="${F[14]}"; seed="${F[15]}"
}

# --------------------------------------------------------------- dry-run
if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "== dry-run: in từng lệnh trial (không chạy thật)"
    for ((i = 0; i < N_TOTAL; i++)); do
        if ! parse_row "${TRIAL_ROWS[$i]}"; then
            echo "  SKIP: malformed manifest line $((i + 1)): ${TRIAL_ROWS[$i]}" >&2
            continue
        fi
        row_fields
        n=$((i + 1))
        run_id=$(printf "trial_%03d" "$n")
        if [[ "$bt" == "1" ]]; then
            run_id+="_${reg}_${opt}_bt"
            [[ "$alpha" != "." ]] && run_id+="_a${alpha}"
            [[ "$beta" != "." ]] && run_id+="_b${beta}"
        else
            run_id+="_${reg}_${opt}_lr${lr}"
            [[ "$schedule" == "diminishing" ]] && run_id+="_dim"
        fi
        run_id+="_lam${lam}"
        [[ "$opt" == "sgd" ]] && run_id+="_bs${bs}"

        printf 'python -m scripts.train --loss %s --w-pos %s --reg %s --lam %s --optimizer %s ' \
            "$loss" "$w_pos" "$reg" "$lam" "$opt"
        if [[ "$bt" == "1" ]]; then
            printf -- '--backtracking --initial_lr %s' "$initial_lr"
            [[ "$alpha" != "." ]] && printf ' --armijo-alpha %s' "$alpha"
            [[ "$beta" != "." ]] && printf ' --armijo-beta %s' "$beta"
        else
            printf -- '--lr %s' "$lr"
        fi
        if [[ "$opt" == "sgd" ]]; then
            printf -- ' --batch-size %s' "$bs"
            [[ "$schedule" == "diminishing" ]] && printf ' --lr-schedule diminishing'
        fi
        printf ' --epochs %s --loss-epsilon %s --patience %s --seed %s --data-dir %s --out-dir <SESSION>/trials --save-figures --run-id %s\n' \
            "${EPOCHS:-$epochs}" "$loss_eps" "$patience" "$seed" "$DATA_DIR" "$run_id"
    done
    exit 0
fi

# ---------------------------------------------------------------- session
SESSION="$OUT_DIR/tune_${LOSS}_$(date +%Y%m%d_%H%M%S)"
TRIALS_DIR="$SESSION/trials"
mkdir -p "$TRIALS_DIR"
cp "$MANIFEST" "$SESSION/manifest.tsv"
: > "$SESSION/failed_trials.txt"

SUMMARY="$SESSION/tune_${LOSS}_summary.tsv"
printf '#%s\n' "trial_id	run_id	loss	w_pos	reg	lam	optimizer	lr	schedule	backtracking	batch_size	armijo_alpha	armijo_beta	best_val_auprc	best_epoch	iters_to_best	epochs_to_best	wall_time_to_best	total_iters	train_time_sec" > "$SUMMARY"

echo "== session: $SESSION"
OK=0
FAIL=0

# ------------------------------------------------------------------ loop
for ((i = 0; i < N_TOTAL; i++)); do
    if ! parse_row "${TRIAL_ROWS[$i]}"; then
        echo "  SKIP: malformed manifest line $((i + 1)): ${TRIAL_ROWS[$i]}" >&2
        echo "trial_$(printf '%03d' "$((i + 1))")" >> "$SESSION/failed_trials.txt"
        FAIL=$((FAIL + 1))
        continue
    fi
    row_fields
    n=$((i + 1))
    run_id=$(printf "trial_%03d" "$n")
    if [[ "$bt" == "1" ]]; then
        run_id+="_${reg}_${opt}_bt"
        [[ "$alpha" != "." ]] && run_id+="_a${alpha}"
        [[ "$beta" != "." ]] && run_id+="_b${beta}"
    else
        run_id+="_${reg}_${opt}_lr${lr}"
        [[ "$schedule" == "diminishing" ]] && run_id+="_dim"
    fi
    run_id+="_lam${lam}"
    [[ "$opt" == "sgd" ]] && run_id+="_bs${bs}"

    cmd=(python -m scripts.train --loss "$loss" --w-pos "$w_pos" --reg "$reg" --lam "$lam" --optimizer "$opt")
    if [[ "$bt" == "1" ]]; then
        cmd+=(--backtracking --initial_lr "$initial_lr")
        [[ "$alpha" != "." ]] && cmd+=(--armijo-alpha "$alpha")
        [[ "$beta" != "." ]] && cmd+=(--armijo-beta "$beta")
    else
        cmd+=(--lr "$lr")
    fi
    if [[ "$opt" == "sgd" ]]; then
        cmd+=(--batch-size "$bs")
        [[ "$schedule" == "diminishing" ]] && cmd+=(--lr-schedule diminishing)
    fi
    cmd+=(--epochs "${EPOCHS:-$epochs}" --loss-epsilon "$loss_eps" --patience "$patience"
          --seed "$seed" --data-dir "$DATA_DIR" --out-dir "$TRIALS_DIR" --save-figures)
    cmd+=(--run-id "$run_id")

    echo "[$n/$N_TOTAL] $run_id"
    if "${cmd[@]}"; then
        OK=$((OK + 1))
        if [[ -f "$TRIALS_DIR/$run_id/metrics.json" ]]; then
            metrics_row=$(jq -r '[.best_val_auprc, .best_epoch, .iters_to_best,
                                  .epochs_to_best, .wall_time_to_best, .total_iters,
                                  .train_time_sec] | @tsv' "$TRIALS_DIR/$run_id/metrics.json")
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$n" "$run_id" "$loss" "$w_pos" "$reg" "$lam" "$opt" "$lr" "$schedule" \
                "$bt" "$bs" "$alpha" "$beta" "$metrics_row" >> "$SUMMARY"
        else
            echo "  WARN: metrics.json missing for $run_id" >&2
        fi
    else
        FAIL=$((FAIL + 1))
        echo "$run_id" >> "$SESSION/failed_trials.txt"
        echo "  FAILED -> $TRIALS_DIR/$run_id/train.log" >&2
    fi
done

echo
echo "== done: ok=$OK  fail=$FAIL  total=$N_TOTAL"
[[ "$FAIL" -gt 0 ]] && echo "failed trials: $SESSION/failed_trials.txt" >&2

# ----------------------------------------------------------- aggregation
echo "== summary -> $SUMMARY"
if [[ "$OK" -gt 0 ]]; then
    BEST="$SESSION/best_per_objective.tsv"
    printf '#%s\n' "loss	reg	rank_type	run_id	primary	secondary" > "$BEST"
    awk -F'\t' '
        NR == 1 { next }
        {
            key = $3 "|" $5
            wt = $18; it = $16
            if (!(key in min_wt) || wt < min_wt[key]) { min_wt[key] = wt; r_wt[key] = $2; tt[key] = $20 }
            if (!(key in min_it) || it < min_it[key]) { min_it[key] = it; r_it[key] = $2; et[key] = $17 }
        }
        END {
            for (k in min_wt) {
                split(k, a, "|")
                printf "%s\t%s\tfastest\t%s\t%s\t%s\n",      a[1], a[2], r_wt[k], min_wt[k], tt[k]
                printf "%s\t%s\tfewest_iters\t%s\t%s\t%s\n", a[1], a[2], r_it[k], min_it[k], et[k]
            }
        }
    ' "$SUMMARY" >> "$BEST"
    echo "== best per objective -> $BEST"
fi
