## Outline

**Bài toán:** phân loại nhị phân — dự đoán một người có mắc bệnh tiểu đường hay không.

**Dữ liệu:** [Diabetes Binary Health Indicators BRFSS](https://www.kaggle.com/datasets/giangle1003/diabetes-prediction-brfss/data) (2015 – 2017 – 2019 – 2021).

**Đặc điểm của dữ liệu cần khai thác**

- **Mất cân bằng (class imbalance)**: tỉ lệ dương tính (mắc bệnh) thường chỉ ~14-15% -> lý do chính để đưa weighted BCE / focal loss vào so sánh, ưu tiên AUPRC, F1 (lớp thiểu số), AUROC.
- **Cost-sensitive scenario**: bỏ sót một người mắc bệnh (false negative) nguy hiểm hơn cảnh báo nhầm (false positive) -> cơ sở để chọn trọng số lớp trong weighted BCE, giả thiết một "cost-weighted score" (vd: chi phí = 5×FN + 1×FP).
- **Tương quan mạnh**: nhiều biến như HighBP, HighChol, HeartDiseaseorAttack, BMI... có tương quan với nhau -> ma trận Hessian của bài toán logistic regression có điều kiện yếu (ill-conditioned) -> ảnh hưởng trực tiếp đến tốc độ hội tụ của GD/SGD -> phân tích phổ trị riêng (eigenvalue spectrum) của Hessian trước/sau khi thêm L2.
- **Gộp nhiều năm khảo sát**: cách hỏi/đặc trưng có thể lệch giữa các năm → nên kiểm tra lại phân phối theo năm, cân nhắc việc chia train/val/test có stratify theo năm hay không.

---

## 1. Loss functions

| | công thức (dạng xác suất, $y\in\{0,1\}$) | công thức (dạng margin, $m=y'z$) | lồi? | trơn? |
| :---- | :---- | :---- | :---- | :---- |
| **BCE** | $L=-[y\log\hat y+(1-y)\log(1-\hat y)]$ | $\phi(m)=\log(1+e^{-m})$ | ✅ | ✅ |
| **Weighted BCE** | $L=-[w_1\, y\log\hat y+w_0(1-y)\log(1-\hat y)]$ | $\phi(m)=w_y\log(1+e^{-m})$ | ✅ | ✅ |
| **Focal** | $L=-\alpha_t(1-p_t)^\gamma\log(p_t)$, với $p_t = y\hat y+(1-y)(1-\hat y)$, $\alpha_t=y\alpha+(1-y)(1-\alpha)$ |  | ❌ | ✅ |
| **Hinge** | | $\phi(m)=\max(0,\,1-m)$ | ✅ | ❌ (tại $m=1$) |
| **Squared hinge** |  | $\phi(m)=\max(0,\,1-m)^2$ | ✅ | ✅ |

Trong đó $w_1, w_0$ là trọng số lớp (weighted BCE), $\gamma\ge 0$ là hệ số focusing và $\alpha\in(0,1)$ là hệ số cân bằng lớp (focal).

**Lưu ý**: hàm loss (công thức) không phụ thuộc model, nhưng **tính lồi thì có** — vì nó là tính chất của loss ∘ model (hợp hàm). Với logistic regression (model tuyến tính, $z = \theta^\top x$ là affine theo $\theta$), loss lồi theo $z$ ⇒ vẫn lồi theo $\theta$. Nếu sau này có mở rộng sang MLP thì hợp với các lớp phi tuyến sẽ phá vỡ tính lồi dù công thức loss không đổi.

Chọn hàm loss:
- nếu tập trung vào tối ưu lồi: **BCE / weighted BCE / squared hinge** (có thể support lý thuyết đầy đủ - Lipschitz constant, lồi mạnh...). 
- nếu muốn đa dạng hơn: thêm hoặc thay bằng **focal loss** (không lồi) -> so sánh hành vi hội tụ của optimizer trên landscape không lồi (nhạy với LR, dễ mắc kẹt hơn...).

---

## 2. Regularization (hiệu chỉnh)

- **L2 (ridge)**: $\frac{\lambda}{2}\|\theta\|_2^2$ — lồi, trơn. -> bài toán trở thành lồi mạnh, cải thiện điều kiện (conditioning) của Hessian, đảm bảo hội tụ tuyến tính cho GD.
- **L1 (lasso)**: $\lambda\|\theta\|_1$ — lồi nhưng không trơn (tại 0) -> Cần proximal gradient (ISTA) hoặc subgradient method thay vì GD thường.

---

## 3. Optimizers (thuật toán tối ưu hóa)

- **GD (gradient descent)** — full-batch.
- **SGD (stochastic gradient descent)** — mini-batch.
- **Accelerated GD** (Nesterov / momentum).
- **Newton** — dùng Hessian, chỉ áp dụng tốt cho loss lồi + trơn (BCE, weighted BCE, squared hinge).
- **Độ dài bước (step size / learning rate)**: so sánh **backtracking line search** vs **cố định**

*(Mở rộng thêm: có thể thêm Adam - adaptive step size, L-BFGS)*

---

## 4. Kịch bản thực nghiệm

Mỗi người cố định 1 loss function:

### Stage 0 — Baseline
BCE + SGD (LR mặc định hợp lý) + không regularization.

### Stage 1 — Optimizer ablation (loss cố định)
- Chạy lần lượt: GD, SGD, Accelerated GD, Newton,...
- mỗi optimizer được **tự tune learning rate riêng** (grid nhỏ, ví dụ {1e-3, 1e-2, 1e-1}) hoặc dùng backtracking — không so sánh các optimizer ở cùng 1 LR cố định chung.
- **Metrics**: đường cong hội tụ (loss vs epoch, loss vs thời gian thực), giá trị hàm mục tiêu cuối cùng, số vòng lặp đến ngưỡng hội tụ (iterations-to-tolerance), độ chuẩn gradient (||∇L||) theo epoch.
- **Output**: chọn ra **opt\*** (thuật toán + lr tốt nhất) cho loss đang xét.

### Stage 2 — Loss-specific hyperparameter tuning
(chỉ áp dụng nếu loss có tham số riêng)
- Weighted BCE: tune tỉ lệ trọng số $w_{pos}:w_{neg}$ (vd 1:1, 2:1, 5:1, ngược lại,...).
- Focal: tune $\gamma$ (vd 0, 1, 2) và $\alpha$.
- Dùng optimizer mặc định nhẹ (vd Adam/SGD LR cố định) chỉ để **xếp hạng** các lựa chọn hyperparameter, chưa cần optimizer đã tune ở Stage 1.
- **Output**: bộ hyperparameter tốt nhất cho loss đang xét (không cần trình bày hết các lần thử, chỉ 1 bảng/biểu đồ nhỏ tóm tắt sweep).

### Stage 3 — Regularization ablation (loss + opt\* cố định)
- So sánh: none vs L1 vs L2, mỗi loại tự tune $\lambda$ riêng (vd {1e-4, 1e-3, 1e-2, 1e-1, 1}).
- **Metrics**: hiệu năng trên tập validation (AUPRC, F1 lớp thiểu số), phổ trị riêng (eigenvalue spectrum) của Hessian để minh họa hiệu ứng conditioning của L2, độ sparse của hệ số với L1.
- **Output**: chọn ra **reg\*** (loại + λ tốt nhất) cho loss đang xét.

### Stage 4 — Đánh giá cuối cùng trên tập test
- Sử dụng bộ (loss cố định, opt\*, hyperparameter loss tốt nhất, reg\*) huấn luyện lại và đánh giá trên tập **test** (chưa từng dùng để tune).
- Ghi lại: accuracy, precision, recall, F1 (lớp thiểu số), AUROC, AUPRC, confusion matrix -> so sánh giữa các loss với setup tốt nhất thu được

## 5. Quy ước chung

Thống nhất:
1. Tiền xử lý dữ liệu + chia train/val/test (cùng seed) — làm 1 lần, dùng chung.
2. Model dùng chung: vd logistic regression (cùng cách khởi tạo).
3. Bộ metric + định dạng báo cáo kết quả.
4. Danh sách optimizer cần thử + khoảng learning rate cần sweep — dùng chung 1 lưới.
5. Danh sách regularizer cần thử + khoảng λ cần sweep — dùng chung 1 lưới.
6. Chỉ báo cáo kết quả tốt nhất mỗi giai đoạn trong bài chính; các lần sweep trung gian để ở phụ lục (appendix) nếu cần.

## 6. Một số câu hỏi thảo luận khi so sánh

- Optimizer nào thắng ở loss nào? Có giống nhau giữa các loss không, hay khác nhau tùy loss?
- Trọng số lớp / weighted BCE và focal có thực sự cải thiện F1 lớp thiểu số so với BCE thường không?
- L2 có cải thiện conditioning rõ rệt không (so phổ trị riêng trước/sau)? L1 có cho nghiệm sparse hữu ích để diễn giải không?
- (nếu có focal) landscape không lồi có khiến optimizer nhạy với LR hơn, hội tụ kém ổn định hơn so với 3 loss lồi còn lại không?