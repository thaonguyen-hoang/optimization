# Tuning Plan — YAML-Driven Grid Search via Bash

## 1. Mục tiêu

Thay thế hệ thống tuning đang lỗi (`scripts/tune.py`) bằng pipeline đơn giản:

- **YAML** giữ làm nguồn grid (3 file: `tune_bce.yaml`, `tune_weighted_bce.yaml`, `tune_squared_hinge.yaml`).
- 1 script Python **nhỏ** đọc YAML → in ra **manifest TSV** (mỗi dòng = 1 tổ hợp hợp lệ). Không train, không metrics, không JSON.
- **Bash** (`run_tune.sh`) đọc manifest, loop chạy `python -m scripts.train ...` cho từng trial, y như `run_train.sh`.
- Mỗi trial = 1 run dir đầy đủ artifacts (config.json, history.npz, metrics.json, checkpoints/, convergence.png).
- Sau khi chạy: gộp summary + so sánh **tốc độ (wall time)** và **số vòng lặp tới hội tụ (iterations-to-best)** theo từng objective `(loss + reg)`.

## 2. Vì sao bỏ `scripts/tune.py`

- Tự loop tích đề-các 7 chiều + dedup → sinh trial sai/lặp (vd `gd_bt` đi kèm `lr=0.01`).
- Tự lưu JSON với history → lỗi serialization (`ndarray`, `NaN`, `int64`).
- Gộp cả logic sinh trial + train + lưu + tổng hợp vào 1 script → khó gỡ, dễ hỏng toàn bộ.

Nguyên tắc mới: **mỗi lớp làm đúng 1 việc, không gộp.**

## 3. Kiến trúc

```
configs/tune_{loss}.yaml  ──►  scripts/expand_tune.py  ──►  manifest TSV (stdout)
                                                                  │
                                                     run_tune.sh loop
                                                                  │
                                              python -m scripts.train <trial>
                                                                  │
                                runs/tune_{loss}_{timestamp}/trials/trial_{NNN}_.../
                                                                  │
                        jq + awk ──► tune_{loss}_summary.tsv + best_per_objective.tsv
```

### Chú thích kỹ thuật
- Bash không đọc được YAML thuần, `yq` không có. Có sẵn `jq` và `pyyaml` → dùng Python expander nhỏ cho việc parse duy nhất.
- `run_id` của từng trial đặt tường minh → không ghi đè lẫn nhau, dễ trace.

## 4. Thay đổi từng file

### 4.1 NEW `scripts/expand_tune.py` (~60 dòng)
- CLI: `python -m scripts.expand_tune <path/to/config.yaml>`
- In TSV ra stdout, 1 dòng/trial, header dòng đầu (prefix `#` để bash bỏ qua).

**Cột TSV (18 cột):**
```
loss  w_pos  reg  lam  opt  lr  schedule  backtracking  initial_lr  batch_size  armijo_alpha  armijo_beta  epochs  loss_epsilon  patience  seed
```
(`backtracking` = `1`/`0`; `batch_size` để `.` nếu full-batch; `armijo_alpha/beta` để `.` nếu không áp dụng.)

**Luật sinh tổ hợp hợp lệ (tường minh theo từng optimizer, KHÔNG dùng `itertools.product` nhiều chiều):**

| Optimizer | reg | fixed-step | backtracking | schedule | batch_size |
|---|---|---|---|---|---|
| `gd` | none/l2 | lr từ `lr_grid` | `(alpha × beta)` từ grid | `fixed` | `.` (full-batch) |
| `gd` | l1 (ISTA) | lr từ `lr_grid` | chỉ `beta` (Parabol) | `fixed` | `.` |
| `nag` | none/l2 | lr từ `lr_grid` | chỉ `beta` (Parabol, không alpha) | `fixed` | `.` |
| `nag` | l1 (FISTA) | lr từ `lr_grid` | chỉ `beta` | `fixed` | `.` |
| `newton` | none/l2 | lr = 1.0 | `(alpha × beta)` | `fixed` | `.` |
| `newton` | l1 | **không tồn tại** (bỏ qua) | — | — | — |
| `sgd` | none/l2/l1 | lr từ `lr_grid` | **không có** | `fixed` + `diminishing` | từ `batch_size` grid |

- `backtracking=true` → `lr` không dùng (mặc định `initial_lr=1.0`).
- Đọc `globals` từ YAML: `tune_epochs`, `loss_epsilon`, `patience`, `seed`, `w_pos_grid` (chỉ `weighted_bce`).
- `w_pos` từ `w_pos_grid` (mặc định `[1.0]` cho bce / squared_hinge).
- Tổng trial ước lượng cho `tune_bce.yaml` hiện tại: `none`=37, `l2`=148, `l1`=120 → **~305 trials** (500 epochs/trial sẽ rất lâu → nên cắt grid hoặc dùng `--epochs` ngắn).

### 4.2 REWRITE `run_tune.sh` (đè file cũ — file cũ đang gọi `scripts.tune`)
```
usage: ./run_tune.sh --config <yaml> [--dry-run] [--epochs N] [--out-dir runs]
```
Các bước:
1. Parse args.
2. Sinh manifest: `python -m scripts.expand_tune "$CONFIG" > manifest.tsv`.
3. `--dry-run`: in từng lệnh `python -m scripts.train ...` rồi thoát (không chạy) → cho phép kiểm tra tổ hợp trước.
4. Tạo session dir: `runs/tune_{loss}_{timestamp}/`; copy `manifest.tsv` vào.
5. Loop đọc TSV, build `cmd=(...)` mảng bash:
   - nếu `backtracking=1` → `--backtracking --initial_lr 1.0` + `--armijo-alpha`/`--armijo-beta` (nếu khác `.`)
   - ngược lại → `--lr "$lr"` (+ `--lr-schedule diminishing` nếu có)
   - `sgd` → `--batch-size "$bs"`
   - luôn có: `--loss --w-pos --reg --lam --optimizer --epochs --loss-epsilon --patience --seed`
   - `--run-id trial_{NNN:03d}_{reg}_{opt}_lr{lr}_lam{lam}[_dim][_bs{bs}][_a{alpha}_b{beta}]`
   - `--out-dir "$SESSION/trials" --save-figures`
   - chạy `"${cmd[@]}"`; nếu fail, ghi tên trial vào `failed_trials.txt`, tiếp tục (không `set -e` trong loop).
6. Tổng hợp:
   - `tune_{loss}_summary.tsv`: mỗi trial 1 dòng, đọc bằng `jq` từ `metrics.json` (jq đã có trên máy).
   - `best_per_objective.tsv`: dùng `awk` group theo cột `loss|reg`, mỗi nhóm in 2 hạng:
     - **fastest**: min `wall_time_to_best` (kèm `train_time_sec`)
     - **fewest iters**: min `iters_to_best` (kèm `epochs_to_best`)
   - Xóa `scripts/tune.py` khỏi `run_tune.sh` cũ.

### 4.3 EDIT `scripts/train.py`
1. **Thêm CLI args:**
   - `--armijo-alpha` (float, default `1e-4`) — hệ số điều kiện Armijo.
   - `--armijo-beta` (float, default `0.5`) — hệ số giảm bước backtracking.
   - Forward cả 2 vào `build_optimizer(...)` (nếu `backtracking`).
2. **`make_run_id`:** mode backtracking thêm hậu tố phân biệt:
   - `bt_init1.0_a0.1_b0.5` (khi alpha/beta khác default).
3. **`build_config`:** thêm `armijo_alpha`, `armijo_beta` vào `config.json`.
4. **Convergence stats** — tính sau `train_logreg`, ghi vào `metrics.json`:
   ```python
   hist = result["history"]; be = result["best_epoch"]
   out["iters_to_best"]     = int(hist["iter"][be])        # full-batch: best_epoch+1; sgd: số batch-update cộng dồn
   out["epochs_to_best"]    = int(hist["epoch"][be]) + 1   # chuẩn chung, so được cross-method
   out["wall_time_to_best"] = float(hist["wall_time"][be]) # giây thực tế tới best
   out["total_iters"]       = int(hist["iter"][-1])        # tổng vòng lặp cả run (cả early stop)
   ```
   → `history.npz` giữ nguyên để plot; mọi chỉ số so sánh nằm gọn trong `metrics.json`.

**Ý nghĩa các chỉ số so sánh:**

| Metric | Nghĩa | Dùng để |
|---|---|---|
| `wall_time_to_best` | giây thực tế tới checkpoint tốt nhất | chọn "nhanh nhất" (hội tụ) |
| `train_time_sec` | tổng giây cả run | so tổng chi phí |
| `iters_to_best` | số vòng lặp tới best (đơn vị riêng từng method) | chọn "ít bước lặp nhất" |
| `epochs_to_best` | số epoch tới best | so chéo giữa method |
| `total_iters` | tổng vòng lặp cả run | so chi phí lặp |

### 4.4 Configs YAML
- Giữ nguyên 3 file hiện có (đã có `lr_grid`, `lam_grid`, `batch_size`, `armijo_alpha/beta`).
- Thêm `patience` vào `globals` mỗi file.
- (Tùy chọn) cắt bớt grid khi tổng trial quá lớn.

### 4.5 DELETE `scripts/tune.py`
- Xóa hẳn. Git history giữ bản cũ.

## 5. Thứ tự triển khai

1. `scripts/expand_tune.py` (mới).
2. `scripts/train.py` (armijo args, run_id, convergence stats).
3. `run_tune.sh` (rewrite).
4. `configs/*.yaml` (thêm `patience`).
5. Xóa `scripts/tune.py`.
6. Test.

## 6. Kiểm tra (verification)

1. `python -m scripts.expand_tune configs/tune_bce.yaml | wc -l` → đếm dòng, xem header.
2. `./run_tune.sh --config configs/tune_bce.yaml --dry-run` → kiểm tra lệnh từng trial:
   - `gd_bt` không còn `--lr` rác.
   - `nag_bt` không có `--armijo-alpha`.
   - `newton` + `reg=l1` không xuất hiện.
   - `sgd` có `--batch-size`, `--lr-schedule diminishing` khi cần.
3. `./run_tune.sh --config configs/tune_bce.yaml --epochs 3` (chạy thật, vài phút):
   - `runs/tune_bce_{ts}/trials/trial_*/` đủ: `metrics.json`, `history.npz`, `config.json`, `convergence.png`.
   - 1 `metrics.json` bất kỳ có `iters_to_best`, `wall_time_to_best`, `epochs_to_best`, `total_iters`.
   - `tune_bce_summary.tsv` và `best_per_objective.tsv` đúng số nhóm `none/l2/l1`.
4. Không còn lỗi JSON serialization (không còn dòng `ERROR:` trong log).

## 7. Ghi chú mở

- ~305 trials cho `tune_bce.yaml` đầy đủ: chạy `--epochs 50-100` trước, hoặc cắt `lr_grid`/`lam_grid` trong YAML.
- `wall_time_to_best` là chỉ số chính cho "nhanh nhất" (bỏ qua thời gian early-stop dư).
- `iters_to_best` không so được chéo giữa full-batch và SGD vì đơn vị khác nhau (epoch vs batch-update) — đúng như định nghĩa user; `epochs_to_best` là chuẩn so chéo.
- Output tổng hợp dùng TSV (không cần pandas); nếu cần CSV, đổi separator trong bước tổng hợp.
