# Tài Liệu Toán Học: Các Thuật Toán Tối Ưu (Optimization Algorithms)

Tài liệu này ghi chép lại chính xác cơ sở toán học đã được lập trình cho các thuật toán trong repository, bao gồm cả hai trường hợp: Hàm mục tiêu hoàn toàn trơn (Smooth) và Hàm mục tiêu phức hợp (Composite/Lasso).

Quy ước ký hiệu:
*   $F(x)$: Hàm mục tiêu tổng thể $F(x) = f(x) + r(x)$.
*   $f(x)$: Phần hàm mục tiêu khả vi, trơn (Ví dụ: BCE Loss).
*   $r(x)$: Phần hàm điều chuẩn (Regularizer).
*   $t$ hoặc $t_k$: Chiều dài bước nhảy (Step size / Learning rate).

---

## PHẦN A. NHÓM HÀM MỤC TIÊU TRƠN (SMOOTH)
Áp dụng khi $r(x) = 0$ (None) hoặc $r(x) = \frac{\lambda}{2} \Vert x \Vert_2^2$ (Ridge/L2). Hàm $F(x)$ khả vi tại mọi điểm.

### 1. Gradient Descent (GD)
Thuật toán leo đồi bậc 1 cơ bản.
*   **Fixed Step Size:** 
    $x_{k+1} = x_k - t \nabla F(x_k)$
*   **Backtracking Line Search:** 
    Khởi tạo $t = \alpha_0$ (mặc định $\alpha_0 = 1.0$). Lặp lại việc giảm $t \leftarrow \beta t$ (với hệ số co $\beta = 0.5$) cho đến khi thỏa mãn bất đẳng thức giảm đủ (Armijo condition):
    $$F(x_k - t \nabla F(x_k)) \le F(x_k) - \alpha t \Vert\nabla F(x_k)\Vert_2^2$$
    *(Với hằng số nới lỏng $\alpha = 10^{-4}$)*

### 2. Nesterov Accelerated Gradient (NAG)
Thuật toán tăng tốc bậc 1 dựa trên quán tính. Duy trì một dãy gia tốc $s_k$.
*   Dãy gia tốc cơ bản: $s_0 = 1$, $s_{k+1} = \frac{1 + \sqrt{1 + 4s_k^2}}{2}$
*   **Fixed Step Size:**
    Tính tâm quán tính dự phóng: $y_k = x_k + \frac{s_{k-1} - 1}{s_k} (x_k - x_{k-1})$
    Cập nhật gradient tại $y_k$: $x_{k+1} = y_k - t \nabla F(y_k)$
*   **Backtracking Line Search:**
    Vẫn tính $y_k$ như trên. Khởi tạo $t = \alpha_0$. Thử $x_{new} = y_k - t \nabla F(y_k)$ và lặp giảm $t \leftarrow \beta t$ cho đến khi thỏa mãn bất đẳng thức chặn trên (Parabol Majorization), **không dùng** hệ số nới lỏng $\alpha$:
    $$F(x_{new}) \le F(y_k) - \frac{t}{2} \Vert\nabla F(y_k)\Vert_2^2$$

### 3. Newton Method
Thuật toán bậc 2, sử dụng ma trận Hessian $H = \nabla^2 F(x_k)$. Hướng dịch chuyển $\Delta x = -H^{-1} \nabla F(x_k)$.
*   **Fixed Step Size (Pure Newton):**
    $x_{k+1} = x_k + t \Delta x$. Trong chế độ Pure Newton, $t$ luôn bằng $1.0$.
*   **Backtracking Line Search (Damped Newton):**
    Khởi tạo $t = 1.0$. Lặp giảm $t \leftarrow \beta t$ cho đến khi thỏa mãn Armijo dọc theo hướng $\Delta x$:
    $$F(x_k + t \Delta x) \le F(x_k) + \alpha t \nabla F(x_k)^T \Delta x$$

### 4. Stochastic Gradient Descent (SGD)
Tối ưu hóa ngẫu nhiên trên các Mini-batch. Gọi gradient ước lượng trên batch là $\nabla F_{batch}(x_k)$. (Không sử dụng Backtracking).
*   **Fixed Step Size:** $x_{k+1} = x_k - t \nabla F_{batch}(x_k)$
*   **Diminishing Step Size:** Để đảm bảo hội tụ (Robbins-Monro), bước nhảy giảm dần theo thời gian:
    $t_k = \frac{t_0}{\sqrt{k}}$ (với $t_0$ là learning rate cấu hình ban đầu, $k$ là số đếm iteration).

---

## PHẦN B. NHÓM HÀM MỤC TIÊU PHỨC HỢP (COMPOSITE / LASSO)
Áp dụng khi $r(x) = \lambda \Vert x \Vert_1$. Hàm không khả vi tại gốc tọa độ. Các thuật toán không sử dụng subgradient mà sử dụng Toán tử kế cận (Proximal Operator) của chuẩn L1, hay còn gọi là Soft-Thresholding:
$$ \text{prox}_{t\lambda}(x) = \text{sign}(x) \max(|x| - t\lambda, 0) $$
*(Lưu ý: Không áp dụng thuật toán bậc 2 - Newton cho bài toán này vì sai lệch cơ sở toán học khi xấp xỉ chuỗi Taylor bậc 2).*

### 1. ISTA (Iterative Shrinkage-Thresholding Algorithm - GD + Proximal)
*   **Proximal Fixed Step Size:**
    $x_{k+1} = \text{prox}_{t\lambda} (x_k - t \nabla f(x_k))$
*   **Proximal Backtracking (ISTA-BT):**
    Khởi tạo $t$. Thử $x_{new} = \text{prox}_{t\lambda} (x_k - t \nabla f(x_k))$ và lặp giảm $t \leftarrow \beta t$ cho đến khi phần hàm trơn $f(x)$ thỏa mãn bất đẳng thức chặn trên (Lipschitz):
    $$f(x_{new}) \le f(x_k) + \nabla f(x_k)^T (x_{new} - x_k) + \frac{1}{2t} \Vert x_{new} - x_k \Vert_2^2$$

### 2. FISTA (Fast ISTA - NAG + Proximal)
Tích hợp động lượng Nesterov vào quá trình Proximal.
*   **Proximal Fixed Step Size:**
    $y_k = x_k + \frac{s_{k-1} - 1}{s_k} (x_k - x_{k-1})$
    $x_{k+1} = \text{prox}_{t\lambda} (y_k - t \nabla f(y_k))$
*   **Proximal Backtracking (FISTA-BT):**
    Thuật toán FISTA-BT phức tạp hơn vì dãy gia tốc $s_k$ phải bảo toàn năng lượng khi $t_k$ thay đổi giữa các bước lặp.
    - Tâm quán tính: $y_k = x_k + \frac{s_{k-1} - 1}{s_k} (x_k - x_{k-1})$
    - Lặp giảm $t_k$ để tìm $x_{new} = \text{prox}_{t_k\lambda} (y_k - t_k \nabla f(y_k))$ thỏa mãn:
      $$f(x_{new}) \le f(y_k) + \nabla f(y_k)^T (x_{new} - y_k) + \frac{1}{2t_k} \Vert x_{new} - y_k \Vert_2^2$$
    - Sau khi chốt được $t_k$, cập nhật dãy gia tốc cho bước kế tiếp:
      $$s_{k+1} = \frac{1 + \sqrt{1 + 4s_k^2 \frac{t_k}{t_{k-1}}}}{2}$$

### 3. Proximal SGD (SGD với L1)
Áp dụng Soft-Thresholding trên từng bước cập nhật Mini-batch.
*   **Proximal Fixed Step Size:**
    $x_{k+1} = \text{prox}_{t\lambda} (x_k - t \nabla f_{batch}(x_k))$
*   **Proximal Diminishing Step Size:**
    Sử dụng dãy giảm $t_k = \frac{t_0}{\sqrt{k}}$. Lưu ý ngưỡng của hàm prox lúc này cũng thu hẹp dần:
    $x_{k+1} = \text{prox}_{t_k\lambda} (x_k - t_k \nabla f_{batch}(x_k))$
