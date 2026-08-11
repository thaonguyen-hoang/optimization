# Hướng Dẫn Sử Dụng: Huấn Luyện & Thực Nghiệm Mô Hình

Tài liệu này hướng dẫn cách chạy thực nghiệm mô hình Logistic Regression sử dụng các công cụ đã được cung cấp trong repository. 

## 1. Môi trường chạy
Môi trường bắt buộc là `optim`.
Kích hoạt bằng lệnh:
```bash
conda activate optim
```

## 2. File `run_train.sh`

`run_train.sh` là script bash mẫu dùng để chạy **một cấu hình thực nghiệm duy nhất**. 
Bên trong script, công việc thực tế được chuyển giao cho lệnh `python -m scripts.train`.

### Chỉnh sửa tham số
Bạn có thể tự do chỉnh sửa các cờ (flags) truyền cho `scripts.train` bên trong file `run_train.sh` bằng bất cứ text editor nào.

Ví dụ về một lệnh gọi chuẩn trong script:
```bash
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
  --save-figures
```

### Giải thích các tham số quan trọng

*   `--loss {bce, weighted_bce, squared_hinge}`: Chọn hàm Loss.
*   `--reg {none, l2, l1}`: Chọn loại hàm điều chuẩn (Regularizer).
    *   *Lưu ý:* `l1` không hỗ trợ thuật toán bậc 2 (Newton).
*   `--lam FLOAT`: Độ lớn (strength) của tham số điều chuẩn ($\lambda$).
*   `--optimizer {gd, nag, newton, sgd}`: Chọn thuật toán tối ưu.
*   `--epochs INT`: Số lượng vòng lặp (Epochs) tối đa.
*   `--eval-test`: Nếu có cờ này, sau khi train xong, mô hình (tại epoch có AUPRC Validation tốt nhất) sẽ được dùng để đánh giá trên tập Test.
*   `--save-figures`: Nếu có cờ này, sẽ vẽ biểu đồ Loss và Gradient Norm rồi lưu vào ảnh `convergence.png`.

## 3. Lưu ý sống còn về cơ chế Step Size (Learning Rate vs. Backtracking)

Cách hệ thống xử lý bước nhảy phụ thuộc vào việc bạn **có bật cờ `--backtracking` hay không**.

### Trường hợp 1: Chế độ Fixed Step Size (KHÔNG dùng `--backtracking`)
*   Nếu bạn KHÔNG truyền cờ `--backtracking`, thuật toán sẽ sử dụng **Fixed step size**.
*   Khi đó, độ lớn của bước nhảy được quy định hoàn toàn bởi tham số `--lr`.
*   Ví dụ: `--optimizer gd --lr 0.05` => Chạy Gradient Descent với bước nhảy cố định $t = 0.05$ suốt toàn bộ quá trình.

### Trường hợp 2: Chế độ Backtracking Line Search (CÓ dùng `--backtracking`)
*   Nếu bạn truyền cờ `--backtracking`, hệ thống sẽ kích hoạt tìm kiếm bước nhảy tự động.
*   **QUAN TRỌNG:** Ở chế độ này, giá trị của `--lr` sẽ bị **bỏ qua hoàn toàn** (dù bạn có truyền vào hay không).
*   Thay vào đó, Backtracking sẽ luôn luôn khởi tạo bước thử nghiệm đầu tiên (initial step) bằng một tham số riêng gọi là `--alpha0`. 
*   Giá trị mặc định của `--alpha0` là `1.0`. Nếu bạn không truyền `--alpha0`, thuật toán luôn bắt đầu thử bước $t = 1.0$ rồi chia đôi dần (nhân với 0.5) cho đến khi thỏa mãn điều kiện Armijo/Parabol.
*   Ví dụ: `--optimizer gd --backtracking --lr 0.05` => Chạy GD với Backtracking. Khởi tạo thử nghiệm luôn là $t=1.0$ (bỏ qua giá trị 0.05), lặp lại việc giảm $t$ để tìm được bước đi tối ưu cho iter hiện tại.

### Cơ chế Scheduling cho SGD
*   Thuật toán `sgd` (Stochastic Gradient Descent) làm việc với mini-batch, do đó **không hỗ trợ Backtracking**.
*   Tuy nhiên, SGD có cơ chế giảm bước nhảy dần đều (Diminishing step size) theo lý thuyết Robbins-Monro.
*   Bạn điều khiển bằng cờ `--lr-schedule {fixed, diminishing}`.
    *   Nếu `--lr-schedule fixed`: SGD chạy với bước nhảy không đổi bằng `--lr`.
    *   Nếu `--lr-schedule diminishing`: SGD khởi tạo bước nhảy là `--lr`, và ở bước thứ $k$, bước nhảy thực tế sẽ là $t_k = \frac{\text{--lr}}{\sqrt{k}}$.

---
*Tóm lại, nếu muốn tuning cố định, hãy quét (sweep) tham số `--lr`. Nếu muốn thuật toán tự đi tìm, hãy bật `--backtracking` và không cần bận tâm đến `--lr`.*
