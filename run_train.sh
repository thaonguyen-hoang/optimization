#!/usr/bin/env bash
# run_train.sh — Khởi chạy một cấu hình huấn luyện duy nhất (Single Run).
# Môi trường yêu cầu: conda activate optim
#
# Dưới đây là danh sách toàn diện các mẫu lệnh chạy cho TẤT CẢ các 
# tổ hợp thuật toán và hàm mục tiêu. Bạn chỉ cần uncomment lệnh bạn muốn thử.

set -euo pipefail

DATA_DIR="data"
OUT_DIR="runs"

# ------------------------------------------------------------------------------
# NHÓM 1: HÀM MỤC TIÊU TRƠN (SMOOTH OBJECTIVE)
# Sử dụng regularizer: --reg none hoặc --reg l2 (Kèm --lam 1e-2)
# Các thuật toán hỗ trợ: gd, nag, newton, sgd
# ------------------------------------------------------------------------------

echo "Chạy ví dụ (vui lòng sửa script để chọn lệnh khác)..."

# 1.1. GRADIENT DESCENT (GD)
# Fixed step size (Dùng --lr, không dùng --backtracking)
# python -m scripts.train --loss bce --reg l2 --lam 1e-2 --optimizer gd --lr 0.1 --epochs 500 --loss-epsilon 1e-4 --data-dir "$DATA_DIR" --out-dir "$OUT_DIR" --save-figures
# Backtracking Armijo (Dùng --backtracking và --initial_lr. Bỏ qua --lr)
# python -m scripts.train --loss bce --reg l2 --lam 1e-2 --optimizer gd --backtracking --initial_lr 1.0 --epochs 500 --loss-epsilon 1e-4 --data-dir "$DATA_DIR" --out-dir "$OUT_DIR" --save-figures

# 1.2. NESTEROV ACCELERATED GRADIENT (NAG)
# Fixed step size
# python -m scripts.train --loss bce --reg l2 --optimizer nag --lr 0.1 --epochs 500 --loss-epsilon 1e-4 --data-dir "$DATA_DIR" --out-dir "$OUT_DIR" --save-figures
# Backtracking Parabol Majorization (Dùng --backtracking và --initial_lr)
# python -m scripts.train --loss bce --reg l2 --optimizer nag --backtracking --initial_lr 1.0 --epochs 500 --loss-epsilon 1e-4 --data-dir "$DATA_DIR" --out-dir "$OUT_DIR" --save-figures

# 1.3. NEWTON METHOD
# Fixed step size (Damped Newton với learning rate)
# python -m scripts.train --loss bce --reg l2 --lam 1e-2 --optimizer newton --lr 1.0 --epochs 500 --loss-epsilon 1e-4 --data-dir "$DATA_DIR" --out-dir "$OUT_DIR" --save-figures
# Backtracking Armijo (Newton LUÔN ép initial_lr=1.0 để giữ hội tụ bậc 2)
# python -m scripts.train --loss bce --reg l2 --lam 1e-2 --optimizer newton --backtracking --epochs 500 --loss-epsilon 1e-4 --data-dir "$DATA_DIR" --out-dir "$OUT_DIR" --save-figures

# 1.4. STOCHASTIC GRADIENT DESCENT (SGD) (Mini-batch, không có backtracking)
# Fixed step size (Dùng --lr và --batch-size)
# python -m scripts.train --loss bce --reg none --optimizer sgd --lr 0.1 --batch-size 256 --epochs 500 --loss-epsilon 1e-4 --data-dir "$DATA_DIR" --out-dir "$OUT_DIR" --save-figures
# Diminishing step size (Bước nhảy giảm dần 1/sqrt(k))
# python -m scripts.train --loss bce --reg none --optimizer sgd --lr 0.1 --lr-schedule diminishing --batch-size 256 --epochs 500 --loss-epsilon 1e-4 --data-dir "$DATA_DIR" --out-dir "$OUT_DIR" --save-figures


# ------------------------------------------------------------------------------
# NHÓM 2: HÀM MỤC TIÊU PHỨC HỢP (LASSO / L1) - TỰ ĐỘNG DÙNG TOÁN TỬ PROXIMAL
# Sử dụng regularizer: --reg l1 (Kèm --lam 1e-2)
# Các thuật toán hỗ trợ: gd (ISTA), nag (FISTA), sgd (Proximal SGD).
# CẤM SỬ DỤNG: newton
# ------------------------------------------------------------------------------

# 2.1. ISTA (GD + Proximal)
# Fixed step size 
# python -m scripts.train --loss bce --reg l1 --lam 1e-2 --optimizer gd --lr 0.1 --epochs 500 --loss-epsilon 1e-4 --data-dir "$DATA_DIR" --out-dir "$OUT_DIR" --save-figures
# Proximal Backtracking (Lipschitz Upper Bound trên phần trơn f)
# python -m scripts.train --loss bce --reg l1 --lam 1e-2 --optimizer gd --backtracking --initial_lr 1.0 --epochs 500 --loss-epsilon 1e-4 --data-dir "$DATA_DIR" --out-dir "$OUT_DIR" --save-figures

# 2.2. FISTA (NAG + Proximal)
# Fixed step size
# python -m scripts.train --loss bce --reg l1 --lam 1e-2 --optimizer nag --lr 0.1 --epochs 500 --loss-epsilon 1e-4 --data-dir "$DATA_DIR" --out-dir "$OUT_DIR" --save-figures
# Proximal Backtracking (FISTA-BT bảo toàn năng lượng s_k)
# python -m scripts.train --loss bce --reg l1 --lam 1e-2 --optimizer nag --backtracking --initial_lr 1.0 --epochs 500 --loss-epsilon 1e-4 --data-dir "$DATA_DIR" --out-dir "$OUT_DIR" --save-figures

# 2.3. PROXIMAL SGD
# Fixed step size
# python -m scripts.train --loss bce --reg l1 --lam 1e-2 --optimizer sgd --lr 0.1 --batch-size 256 --epochs 500 --loss-epsilon 1e-4 --data-dir "$DATA_DIR" --out-dir "$OUT_DIR" --save-figures
# Diminishing step size
python -m scripts.train --loss bce --reg l1 --lam 1e-2 --optimizer sgd --lr 0.1 --lr-schedule diminishing --batch-size 256 --epochs 500 --loss-epsilon 1e-4 --data-dir "$DATA_DIR" --out-dir "$OUT_DIR" --save-figures
