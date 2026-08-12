# Hướng Dẫn Sử Dụng Toàn Diện: Logistic Regression Optimization Framework

Tài liệu này là hướng dẫn chính thức và toàn diện cho việc chạy thực nghiệm mô hình Logistic Regression sử dụng mã nguồn trong dự án. Tài liệu giải thích chi tiết mọi tham số CLI, các tổ hợp hợp lệ, giá trị mặc định, và những lưu ý cốt lõi về thuật toán.

---

## 1. Môi Trường Chạy
Bạn bắt buộc phải kích hoạt môi trường Conda đã cài đặt trước khi chạy bất kỳ script nào.
```bash
conda activate optim
```

---

## 2. Kịch Bản Khởi Chạy (Scripts)

Dự án cung cấp 2 kịch bản chính:
1.  **`run_train.sh` (Single Run):** Gọi `scripts/train.py`. Dùng để chạy thử nghiệm một tổ hợp thuật toán - tham số cụ thể. Rất hữu ích để debug hoặc theo dõi chi tiết một quá trình hội tụ.
2.  **`run_tune.sh` (Grid Search):** Gọi `scripts/tune.py`. Dùng để tự động chạy quét hàng loạt các tham số (Learning rate, Lambda) qua các thuật toán khác nhau, sau đó xuất ra bảng tổng hợp kết quả (summary csv) và file cấu hình tốt nhất.

Bạn có thể chỉnh sửa trực tiếp nội dung các file bash `.sh` này, hoặc gọi thẳng file python trên terminal.

---

## 3. Từ Điển Tham Số CLI (dành cho `scripts/train.py`)

Dưới đây là toàn bộ các arguments bạn có thể truyền vào `scripts/train.py`.

### 3.1. Cấu hình Hàm Mục Tiêu (Objective)
*   `--loss`: Hàm suy hao. Cấu hình: `bce` (mặc định), `weighted_bce`, `squared_hinge`, `focal`.
    *   *Lưu ý:* `focal` là hàm không lồi (non-convex).
*   `--w-pos`, `--w-neg` (Kiểu float): Dùng riêng cho `weighted_bce` để xử lý dữ liệu mất cân bằng (imbalanced). Mặc định đều là `1.0`.
*   `--reg`: Hàm điều chuẩn (Regularizer). Cấu hình: `none` (mặc định), `l2` (Ridge), `l1` (Lasso).
*   `--lam` (Kiểu float): Hệ số điều chuẩn $\lambda$. Mặc định `0.0`. Chỉ có tác dụng khi `--reg` là `l1` hoặc `l2`.

### 3.2. Cấu hình Thuật Toán (Optimizer)
*   `--optimizer`: Thuật toán tối ưu. Cấu hình: `gd` (mặc định), `nag` (Nesterov), `newton`, `sgd`.

### 3.3. Cấu hình Bước Nhảy (Step Size / Line Search)
ĐÂY LÀ PHẦN QUAN TRỌNG NHẤT. Có 3 tham số chi phối bước nhảy: `--lr`, `--backtracking`, và `--initial_lr`.

*   `--lr` (Kiểu float, mặc định `1e-2`): Chiều dài bước nhảy (Learning Rate) khi chạy ở chế độ **Fixed Step Size**. 
*   `--backtracking` (Cờ/Flag, không cần giá trị): Kích hoạt tính năng tìm kiếm bước nhảy tự động (Armijo / Parabol / Lipschitz). 
    *   **⚠️ LƯU Ý ĐỎ:** Nếu bạn thêm cờ `--backtracking`, tham số `--lr` **SẼ BỊ BỎ QUA HOÀN TOÀN** (ngay cả khi bạn chỉ định `--lr 0.05`, thuật toán cũng không quan tâm).
*   `--initial_lr` (Kiểu float, mặc định `1.0`): Bước nhảy khởi tạo (Initial step) dành **riêng cho chế độ Backtracking**. Mỗi iteration, thuật toán sẽ bắt đầu thử với bước $t$ bằng giá trị của `--initial_lr`, sau đó giảm dần nếu chưa thỏa mãn điều kiện.
*   `--lr-schedule`: Chế độ giảm bước nhảy. Cấu hình: `fixed` (mặc định), `diminishing`.
    *   *Chỉ có tác dụng khi `--optimizer sgd`.* Nếu `diminishing`, bước nhảy sẽ giảm theo $t_k = \frac{\text{lr}}{\sqrt{k}}$.

### 3.4. Cấu hình Quá Trình Huấn Luyện
*   `--epochs` (Kiểu int, mặc định `50`): Số vòng lặp qua toàn bộ dữ liệu.
*   `--batch-size` (Kiểu int, mặc định `256`): Kích thước mini-batch. 
    *   *Lưu ý:* Chỉ có tác dụng với `--optimizer sgd`. Các thuật toán còn lại (GD, NAG, Newton) luôn là Full-Batch (dùng toàn bộ tập train), tham số này tự động bị bỏ qua.
*   `--seed` (Kiểu int, mặc định `42`): Hạt giống ngẫu nhiên để chia batch cho SGD và lấy mẫu Hessian.

### 3.5. Logging và Output
*   `--data-dir`, `--out-dir`: Thư mục chứa data và nơi lưu output. Mặc định `data` và `runs`.
*   `--log-every-iters` (Kiểu int, mặc định `50`): Tần suất (tính theo bước cập nhật/iteration) ghi log Train Loss và Gradient Norm để vẽ biểu đồ mịn.
*   `--verbose-every` (Kiểu int, mặc định `10`): Tần suất (tính theo Epoch) in kết quả ra Terminal.
*   `--eval-test` (Cờ/Flag): Chạy đánh giá trên tập Test sau khi train xong bằng Checkpoint có Validation AUPRC cao nhất.
*   `--save-figures` (Cờ/Flag): Tự động tạo và lưu ảnh biểu đồ hội tụ (Convergence plot).
*   `--hessian-spectrum` (Cờ/Flag): Tính toán phổ giá trị riêng (Eigenvalues) của ma trận Hessian ở epoch cuối. Lưu thành file `.npy`. (Tốn thời gian với data lớn).
*   `--no-standardize` (Cờ/Flag): Tắt tính năng tự động chuẩn hóa Z-score (Mean/Std) dữ liệu. (Không khuyến cáo).

---

## 4. Ma Trận Tương Thích (Khắc phục lỗi chạy)

Không phải tổ hợp tham số nào cũng hợp lệ về mặt toán học. Dưới đây là bảng quy tắc sống còn mà code đã thiết lập:

| Đặc tính / Mục tiêu | Cấu hình cho phép chạy | Tổ hợp sẽ BÁO LỖI (Bị Cấm) | Giải thích ngắn gọn |
| :--- | :--- | :--- | :--- |
| **Bậc 2 (Newton)** | `--reg none`, `--reg l2` | CẤM DÙNG VỚI `--reg l1` | L1 không khả vi (Non-smooth). Việc đưa L1 vào ma trận Hessian là sai toán học. L1 không thể dùng Newton. |
| **Backtracking Line Search** | `gd`, `nag`, `newton` | CẤM DÙNG VỚI `--optimizer sgd` | SGD dùng Mini-batch, hàm mục tiêu bị nhiễu từng bước. Backtracking đòi hỏi đánh giá sự suy giảm của Full-batch objective, không thể áp dụng cho SGD. |
| **Bước nhảy Diminishing** | `--optimizer sgd` | CẤM (Vô tác dụng) với `gd`, `nag`, `newton` | GD/NAG/Newton luôn dùng Fixed Step hoặc Backtracking. Scheduling $1/\sqrt{k}$ chỉ dành cho SGD. |
| **Soft-Thresholding (ISTA/FISTA)** | Kích hoạt tự động khi chọn `--reg l1` với `gd`, `nag`, `sgd`. | N/A | Code tự động chuyển sang dùng toán tử kế cận (Proximal Operator) thay vì đạo hàm cho L1. Backtracking của L1 cũng tự động chuyển sang mô hình Parabol Majorization. |

---

## 5. Hiện Vật Đầu Ra (Artifacts)
Sau khi chạy `scripts/train.py`, hệ thống sinh ra một folder độc nhất tại `runs/<run_id>/` chứa:
1.  `config.json`: File lưu toàn bộ tham số đã dùng (cực kỳ quan trọng để tái lập kết quả).
2.  `history.npz`: File numpy chứa các mảng lịch sử (Train loss, Val metrics, Thời gian, Iteration).
3.  `metrics.json`: Thống kê kết quả AUPRC, F1, Accuracy... của mô hình tốt nhất.
4.  `checkpoints/best.npz` và `checkpoints/last.npz`: Trọng số $w$ và bias $b$ đã học.
5.  `convergence.png`: (Nếu có cờ `--save-figures`). Ảnh vẽ 3 panel đánh giá quá trình hội tụ.
