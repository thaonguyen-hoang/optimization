# Hướng dẫn chạy thực nghiệm (toàn bộ kịch bản)

Kịch bản gồm 2 phần:

- **Phần A — tập trung bài toán tối ưu hóa** (~60-70% khối lượng):
  - **A.1**: với mỗi loss (BCE | weighted-BCE | squared hinge), sweep mọi tổ hợp
    optimizer × step strategy × regularization × λ, huấn luyện đến hội tụ, rồi so sánh
    tốc độ hội tụ (theo **thời gian** và theo **số vòng lặp**), tìm bộ
    (optimizer + step strategy + λ + c) tốt nhất cho từng reg, và chọn **x***.
  - **A.2**: dùng x* để tune **w_pos / w_neg** trên tập val (chỉ cho weighted-BCE).
- **Phần B — liên hệ bài toán học máy**: đánh giá 3 x* (mỗi loss 1 cái) trên
  tập test held-out, so sánh AUPRC/F1-minority/AUROC/accuracy, kết luận về lựa
  chọn loss với dữ liệu imbalance.

Toàn bộ code nằm trong thư mục `final/`.

---

## 1. Chuẩn bị

### 1.1 Môi trường
```bash
cd final
pip install -r requirements.txt      # numpy, pandas, matplotlib, pyyaml
```

### 1.2 Dữ liệu
Thư mục dữ liệu (`<DATA_DIR>`) phải chứa 3 file CSV đã tách sẵn:
```
<DATA_DIR>/
  train.csv
  val.csv
  test.csv
```
(Trong repo này có sẵn ở `../v2/data/processed`.)

### 1.3 Kiểm tra nhanh (1 run)
```bash
python runner.py --config configs/a1_bce.yaml \
    --data-dir <DATA_DIR> --results-dir results --dry-run
```
Chạy thử 1 config, sinh `results/<loss>/summary.csv` để kiểm tra pipeline.

---

## 2. Phần A.1 — sweep theo loss, chia cho 3 người

Mỗi người phụ trách **1 loss** (không đụng nhau, có thể chạy song song):

| Người | Loss | Config | Số run |
|---|---|---|---|
| Người 1 | BCE | `configs/a1_bce.yaml` | 190 |
| Người 2 | Weighted-BCE | `configs/a1_weighted_bce.yaml` | 190 |
| Người 3 | Squared Hinge | `configs/a1_squared_hinge.yaml` | 190 |

### 2.1 Lệnh chạy
```bash
# Người 1
python runner.py --config configs/a1_bce.yaml \
    --data-dir <DATA_DIR> --results-dir results

# Người 2
python runner.py --config configs/a1_weighted_bce.yaml \
    --data-dir <DATA_DIR> --results-dir results

# Người 3
python runner.py --config configs/a1_squared_hinge.yaml \
    --data-dir <DATA_DIR> --results-dir results
```

Kết quả mỗi người (tự động sinh):
```
results/<loss>/
  summary.csv                 # 1 dòng / run: optimizer, step_type, lam, c, best_val_auprc,
                              #   best_epoch, epochs_run, stop_reason, converged, total_wall_time
  <run_id>.json               # artifact đầy đủ (history, best_w/best_b, final_grad_norm,
                              #   final_objective, final_metrics, ...)
```

### 2.2 Những phân tích phải làm (mỗi người, cho loss của mình)

1. **Tốc độ hội tụ — nhận xét riêng theo 2 hướng:**
   - Theo **thời gian**: cột `total_wall_time` (chỉ so giữa các run có `converged = True`).
   - Theo **số vòng lặp**: cột `epochs_run`.
   - Run có `converged = False` (thường `stop_reason = max_epochs`) là *không hội tụ trong
     budget* — tách riêng, không tính vào bảng so tốc độ.
2. **Với mỗi reg (none / l2 / l1):** tìm bộ tốt nhất gồm
   `(optimizer, step_type, lam, c)` theo `best_val_auprc` trên val
   (step_type = fixed | backtracking | diminishing).
3. **So sánh L2 vs None:** quan sát `epochs_run` / `final_grad_norm` của cùng một
   optimizer để nhận xét L2 có giúp hội tụ nhanh hơn không (kiểm chứng, không khẳng định trước).
4. **So thời gian chạy với thư viện:** dùng `sklearn.linear_model.LogisticRegression`
   (đặt `solver` tương ứng, `max_iter` đủ lớn, `tol` theo grad-norm) làm mốc tham chiếu —
   chú ý so trên **cùng một tiêu chí hội tụ** mới công bằng.
5. **Chọn x\* cho loss của mình:** tổ hợp `(loss, reg, optimizer, step_type, lam, c)`
   có `best_val_auprc` cao nhất **trên val** (mặc định x* có thể là weighted-BCE khi
   so chung, nhưng mỗi người giữ 1 x* riêng cho loss mình để dùng ở Phần B).

Ví dụ lọc bằng pandas:
```python
import pandas as pd
df = pd.read_csv("results/<loss>/summary.csv")
converged = df[df["converged"] == True]                        # chỉ tính run hội tụ
best_per_reg = df.loc[df.groupby("regularization")["best_val_auprc"].idxmax()]
print(best_per_reg[["run_id", "optimizer", "step_type",
                    "best_val_auprc", "epochs_run", "total_wall_time"]])
```
> Lưu ý: `summary.csv` **không** có cột `lam`/`c`. Hai giá trị này nằm trong `run_id`
> (định dạng `..._lam<λ>_c<c>_bs<batch>`) và trong mỗi artifact JSON
> (`<run_id>.json` → `hyperparams.lam`, `hyperparams.c`).

### 2.3 Vẽ biểu đồ

Mỗi người chạy `plot_results.py` cho loss của mình **sau khi sweep xong** (plot đọc từ
artifact JSON trong `results/<loss>/`, **không phải** từ `summary.csv` — do đó đừng xóa
các file `<run_id>.json`):

```bash
python plot_results.py --results-dir results/<loss> --output-dir plots
```

Sinh 6 PNG trong `plots/`, mỗi plot tổng hợp **nhiều thuật toán trên cùng 1 đồ thị**
(một đường = một tổ hợp `optimizer-step_type`, tự chọn run có val AUPRC cao nhất của
tổ hợp đó), lưới 1×3 chia theo reg none/l2/l1:

| File | Trục x | Trục y |
|---|---|---|
| `<loss>_loss_epoch.png` | epoch | training loss |
| `<loss>_loss_time.png` | wall_time | training loss |
| `<loss>_loss_iteration.png` | iter | training loss |
| `<loss>_auprc_epoch.png` | epoch | validation AUPRC |
| `<loss>_gradient_epoch.png` | epoch | gradient norm (log) |
| `<loss>_lr_sensitivity.png` | learning rate (c/L) | final validation loss |

> Ghi chú khi nhận xét: plot hiển thị mọi run đã lưu, kể cả run `converged = False`
> (không hội tụ trong budget) — khi đọc biểu đồ hãy đối chiếu cột `converged` trong
> `summary.csv` để tránh kết luận sai về tốc độ hội tụ.

---

## 3. Phần A.2 — tune w_pos / w_neg (chỉ weighted-BCE, 8 run)

Sau khi Người 2 có x* của weighted-BCE:

1. Điền x* vào **4 file template** (mỗi file có TODO hướng dẫn):
   ```bash
   configs/a2_weighted_bce_wpos_2.0.yaml
   configs/a2_weighted_bce_wpos_5.0.yaml
   configs/a2_weighted_bce_wpos_10.0.yaml
   configs/a2_weighted_bce_wpos_20.0.yaml
   ```
   Thay các trường `optimizers`, `regularizations`, `lambda`, `c`, `batch_sizes`
   bằng đúng giá trị của x* (bỏ qua trường không áp dụng: `lambda` nếu reg=none,
   `c` nếu optimizer=newton, `batch_sizes` nếu không phải SGD).
   Mỗi file chạy **2 run** (cả 2 step type của optimizer — giữ đúng tinh thần so
   backtracking vs fixed của Phần A).
2. Chạy cả 4:
   ```bash
   python runner.py --config configs/a2_weighted_bce_wpos_2.0.yaml \
       --data-dir <DATA_DIR> --results-dir results
   # ... lặp lại cho 5.0, 10.0, 20.0
   ```
3. Chọn `w_pos` tốt nhất theo `best_val_auprc` (kết quả nằm ở `results/weighted_bce/summary.csv`).

---

## 4. Phần B — đánh giá test held-out (3 lần)

Với mỗi loss, dùng x* của loss đó (file `results/<loss>/summary.csv` đã chứa sẵn
thông tin run được chọn bởi `evaluate_best.py` theo `best_val_auprc`):

```bash
python evaluate_best.py --summary results/bce/summary.csv \
    --data-dir <DATA_DIR> --output results/bce/final_test.json

python evaluate_best.py --summary results/weighted_bce/summary.csv \
    --data-dir <DATA_DIR> --output results/weighted_bce/final_test.json

python evaluate_best.py --summary results/squared_hinge/summary.csv \
    --data-dir <DATA_DIR> --output results/squared_hinge/final_test.json
```

Kết quả mỗi file `final_test.json` gồm `test_metrics` (accuracy, f1_minority, auroc,
auprc, confusion matrix) và `cost_weighted_score`.

> Lưu ý: `evaluate_best.py` tự chọn run có `best_val_auprc` cao nhất trong `summary.csv`.
> Với weighted-BCE, sau khi chạy xong A.2, summary đã gộp cả run tuned `w_pos` → nó tự
> chọn đúng x* cuối cùng.

So sánh 3 x* với nhau, **metric chính là AUPRC / F1-minority / AUROC** (accuracy chỉ
tham khảo vì dữ liệu mất cân bằng). Kết luận: loss nào phù hợp với imbalance, ưu nhược
của từng hàm loss, và việc chọn loss theo đặc điểm bài toán.

> Khuyến nghị: chạy test nhiều seed (nếu cần) để báo mean ± std, tránh kết luận từ 1 lần chạy.

---

## 5. Tổng số lần chạy

| Giai đoạn | Số lần | Ghi chú |
|---|---|---|
| A.1 (3 loss × 190) | 570 | chia 3 người |
| A.2 (4 w_pos × 2 step) | 8 | 1 người |
| B (3 x* trên test) | 3 | 1 người |
| **Tổng** | **581** | |

---

## 6. Ghi chú kỹ thuật

- **Resume / chạy lại**: `runner.py` bỏ qua run đã có file JSON (không ghi đè). Muốn chạy
  lại run cũ thì thêm `--overwrite`. `summary.csv` luôn được dựng lại từ toàn bộ JSON trên đĩa
  → có thể chạy nhiều lần/ngắt tiếp tục thoải mái.
- **Dry-run**: `--dry-run` chỉ chạy 1 config để kiểm tra.
- **Seed**: mặc định `seed: 42` (sửa trong config). Với SGD nên cân nhắc thêm nhiều seed
  để giảm phương sai.
- **Tiêu chí dừng** (xem chi tiết trong `train.py`): trigger là EMA tương đối của loss
  (`|delta| < tol·|smoothed| + atol` trong `patience` epochs); trước khi công nhận hội tụ
  phải qua acceptance grad-norm (`tol_stat` / `atol_stat`); run không qua acceptance sẽ
  grace chạy tiếp tới `max_epochs` và được gắn `converged = False`.
- **Không có nghĩa khi chạy**: newton+L1 (tự bỏ), SGD+backtracking (thay bằng diminishing),
  c sweep cho newton/backtracking (tự ép c=1) — runner đã xử lý sẵn.
- **Cấu trúc tham số dừng**: `tol=1e-4, atol=1e-8, tol_stat=1e-2, atol_stat=1e-4, patience=3`,
  `max_epochs=500`, backtracking `{initial_lr:1.0, beta:0.5, alpha:1e-4}`, newton
  `{epsilon_damp:1e-8}` — chỉnh trong từng file config.