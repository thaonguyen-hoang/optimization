#!/usr/bin/env bash
# run_train.sh — Khởi chạy một cấu hình huấn luyện duy nhất (Single Run).
# Môi trường yêu cầu: conda activate optim
#
# Dưới đây là danh sách toàn diện các tham số hỗ trợ bởi scripts/train.py.
# Bạn có thể bật/tắt (bằng dấu \) hoặc thay đổi giá trị để thử nghiệm.

set -euo pipefail

DATA_DIR="data"
OUT_DIR="runs"

python -m scripts.train \
  --loss weighted_bce \
  --w-pos 5.15 \
  --reg l2 \
  --lam 1e-2 \
  --optimizer newton \
  --lr 1e-2 \
  --backtracking 1 \
  --alpha0 1.0 \
  --lr-schedule fixed \
  --epochs 100 \
  --batch-size 256 \
  --seed 42 \
  --log-every-iters 50 \
  --verbose-every 10 \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT_DIR" \
  --eval-test \
  --save-figures \
  --hessian-spectrum

# ==============================================================================
# VÍ DỤ CÁC TRƯỜNG HỢP SỬ DỤNG PHỔ BIẾN (Mở comment để chạy thử):
# ==============================================================================

# 1. SGD với Diminishing Step Size (Không hỗ trợ backtracking)
# python -m scripts.train --loss bce --reg none --optimizer sgd --lr 0.1 --lr-schedule diminishing --epochs 50

# 2. ISTA (GD + L1) với Backtracking trên hàm trơn (FISTA thì thay optimizer thành nag)
# python -m scripts.train --loss bce --reg l1 --lam 1e-3 --optimizer gd --backtracking --alpha0 1.0 --epochs 50

# 3. Weighted BCE xử lý mất cân bằng lớp (Custom w-pos)
# python -m scripts.train --loss weighted_bce --w-pos 6.0 --reg l2 --lam 1e-3 --optimizer nag --backtracking --epochs 50
