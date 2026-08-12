"""
optimizers.py
=============
Optimization algorithms for smooth and composite (L1) objectives.

Algorithms implemented strictly following the mathematical definitions:
1. GD:
   - Smooth: Fixed step, or Backtracking (Armijo on F).
   - L1: ISTA Fixed step, or ISTA-BT (Parabol majorization on f).
2. NAG (Nesterov Accelerated Gradient):
   - Smooth: Fixed step, or Backtracking (Parabol on F without alpha).
   - L1: FISTA Fixed step, or FISTA-BT (Parabol on f, energy-preserving s_k).
3. Newton:
   - Smooth only. Fixed step (t=1), or Damped Newton (Armijo on F).
4. SGD:
   - Smooth: Fixed step, or Diminishing (t_0 / sqrt(k)).
   - L1: Proximal SGD (Fixed or Diminishing).

Interface:
    w_new, b_new = opt.step(ctx)
where ctx contains:
    - w, b: current parameters
    - smooth: boolean, whether objective is fully smooth
    - lam: regularization strength (for prox threshold)
    - f_val(w,b): smooth part value
    - F_val(w,b): full objective value (f + r)
    - grad_w(w,b), grad_b(w,b): full-batch gradient of smooth part f(w)
    - grad_w_batch, grad_b_batch: mini-batch gradient of f(w)
    - hessian_wb(w,b): returns (H, g_w, g_b) of full F(w) (includes L2)
    - reg_grad(w): gradient of smooth regularizer r(w)
    - prox(w, threshold): proximal operator of r(w)
"""

import numpy as np
import math

LR_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
LAMBDA_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]


class GD:
    name = "gd"
    def __init__(self, lr: float = 1e-2, backtracking: bool = False, beta: float = 0.5, alpha: float = 1e-4):
        self.lr = lr
        self.backtracking = backtracking
        self.beta = beta
        self.alpha = alpha

    def reset(self, w_shape, b_shape):
        pass

    def step(self, ctx):
        w, b = ctx["w"], ctx["b"]
        
        if ctx["smooth"]:
            # --- Smooth: Gradient Descent ---
            # F(x) = f(x) + r(x)
            F0 = ctx["F_val"](w, b)
            gw = ctx["grad_w"](w, b) + ctx["reg_grad"](w)
            gb = ctx["grad_b"](w, b)
            grad_sq_norm = float(np.sum(gw**2) + gb**2)

            if not self.backtracking:
                t = self.lr
                w_new = w - t * gw
                b_new = b - t * gb
            else:
                t = 1.0  # initialize backtracking
                while True:
                    w_new = w - t * gw
                    b_new = b - t * gb
                    F_new = ctx["F_val"](w_new, b_new)
                    # Armijo: F(x - t*g) <= F(x) - alpha * t * ||g||^2
                    if F_new <= F0 - self.alpha * t * grad_sq_norm + 1e-12:
                        break
                    t *= self.beta
                    if t < 1e-12: break
                self.lr = t
            return w_new, b_new
            
        else:
            # --- Non-Smooth (L1): ISTA ---
            f0 = ctx["f_val"](w, b)
            gw = ctx["grad_w"](w, b)
            gb = ctx["grad_b"](w, b)
            
            if not self.backtracking:
                t = self.lr
                w_new = ctx["prox"](w - t * gw, t * ctx["lam"])
                b_new = b - t * gb  # bias is not regularized
            else:
                t = 1.0
                while True:
                    w_new = ctx["prox"](w - t * gw, t * ctx["lam"])
                    b_new = b - t * gb
                    f_new = ctx["f_val"](w_new, b_new)
                    
                    # NOTE: Không thể rút gọn như Smooth GD, bắt buộc tính đủ
                    diff_w = w_new - w
                    diff_b = b_new - b
                    diff_sq = float(np.sum(diff_w**2) + diff_b**2)
                    dot_grad = float(np.dot(gw, diff_w) + gb * diff_b)
                    
                    # Parabol majorization on f: f(x_new) <= f(x) + <g, x_new - x> + (1/2t)||x_new - x||^2
                    if f_new <= f0 + dot_grad + (1.0 / (2 * t)) * diff_sq + 1e-12:
                        break
                    t *= self.beta
                    if t < 1e-12: break
                self.lr = t
            return w_new, b_new


class NAG:
    name = "nag"
    def __init__(self, lr: float = 1e-2, backtracking: bool = False, beta: float = 0.5):
        self.lr = lr
        self.backtracking = backtracking
        self.beta = beta
        self.s = 1.0
        self.x_prev_w = None
        self.x_prev_b = None

    def reset(self, w_shape, b_shape):
        self.s = 1.0
        self.x_prev_w = np.zeros(w_shape)
        self.x_prev_b = 0.0

    def step(self, ctx):
        x_w, x_b = ctx["w"], ctx["b"]
        
        # Momentum point y_k
        momentum_term = (self.s - 1.0) / self.next_s(self.s, 1.0) # approx for next s if not backtracking
        y_w = x_w + momentum_term * (x_w - self.x_prev_w)
        y_b = x_b + momentum_term * (x_b - self.x_prev_b)

        if ctx["smooth"]:
            # --- Smooth: Nesterov Accelerated Gradient ---
            gw = ctx["grad_w"](y_w, y_b) + ctx["reg_grad"](y_w)
            gb = ctx["grad_b"](y_w, y_b)
            
            if not self.backtracking:
                t = self.lr
                s_next = (1.0 + math.sqrt(1.0 + 4 * self.s**2)) / 2.0
                y_w_actual = x_w + ((self.s - 1.0) / s_next) * (x_w - self.x_prev_w)
                y_b_actual = x_b + ((self.s - 1.0) / s_next) * (x_b - self.x_prev_b)
                gw_act = ctx["grad_w"](y_w_actual, y_b_actual) + ctx["reg_grad"](y_w_actual)
                gb_act = ctx["grad_b"](y_w_actual, y_b_actual)
                x_new_w = y_w_actual - t * gw_act
                x_new_b = y_b_actual - t * gb_act
                self.s = s_next
            else:
                t = 1.0
                F_y = ctx["F_val"](y_w, y_b)
                
                # OPTIMIZATION NOTE (Bản chất Toán học & Hiệu năng):
                # Bản chất điều kiện Backtracking ở đây chính là Descent Lemma (Lipschitz Upper Bound):
                # F(x_new) <= F(y) + <grad_F(y), x_new - y> + (1/2t)||x_new - y||^2
                # Tuy nhiên, đối với hàm hoàn toàn trơn, ta có x_new - y = -t * grad_F(y).
                # Bằng cách thế trực tiếp vào bất đẳng thức, ta rút gọn được thành:
                # F(x_new) <= F(y) - (t/2)||grad_F(y)||^2
                # Lợi ích khổng lồ về mặt tính toán: ||grad_F(y)||^2 không phụ thuộc vào t.
                # Ta tính nó ĐÚNG 1 LẦN duy nhất ở ngoài vòng lặp while (tốn O(d)), 
                # giúp bên trong vòng lặp thử t (while) chỉ cần làm phép tính vô hướng O(1).
                grad_sq_norm = float(np.sum(gw**2) + gb**2)
                
                while True:
                    x_new_w = y_w - t * gw
                    x_new_b = y_b - t * gb
                    F_new = ctx["F_val"](x_new_w, x_new_b)
                    
                    # Parabol Majorization (Rút gọn từ Descent Lemma)
                    if F_new <= F_y - (t / 2.0) * grad_sq_norm + 1e-12:
                        break
                    t *= self.beta
                    if t < 1e-12: break
                self.lr = t
                self.s = (1.0 + math.sqrt(1.0 + 4 * self.s**2)) / 2.0
        else:
            # --- Non-Smooth (L1): FISTA ---
            if not self.backtracking:
                t = self.lr
                s_next = (1.0 + math.sqrt(1.0 + 4 * self.s**2)) / 2.0
                y_w_actual = x_w + ((self.s - 1.0) / s_next) * (x_w - self.x_prev_w)
                y_b_actual = x_b + ((self.s - 1.0) / s_next) * (x_b - self.x_prev_b)
                gw_act = ctx["grad_w"](y_w_actual, y_b_actual)
                gb_act = ctx["grad_b"](y_w_actual, y_b_actual)
                
                x_new_w = ctx["prox"](y_w_actual - t * gw_act, t * ctx["lam"])
                x_new_b = y_b_actual - t * gb_act
                self.s = s_next
            else:
                # FISTA-BT with energy-preserving s_k
                t = 1.0
                t_prev = self.lr
                
                while True:
                    # s_next depends on t / t_prev
                    s_next = (1.0 + math.sqrt(1.0 + 4 * self.s**2 * (t / t_prev))) / 2.0
                    y_w_bt = x_w + ((self.s - 1.0) / s_next) * (x_w - self.x_prev_w)
                    y_b_bt = x_b + ((self.s - 1.0) / s_next) * (x_b - self.x_prev_b)
                    
                    f_y = ctx["f_val"](y_w_bt, y_b_bt)
                    gw_bt = ctx["grad_w"](y_w_bt, y_b_bt)
                    gb_bt = ctx["grad_b"](y_w_bt, y_b_bt)
                    
                    x_new_w = ctx["prox"](y_w_bt - t * gw_bt, t * ctx["lam"])
                    x_new_b = y_b_bt - t * gb_bt
                    
                    # OPTIMIZATION NOTE:
                    # Vì x_new được sinh ra từ toán tử Proximal chứ không phải trực tiếp là
                    # x_new = y - t*grad, ta KHÔNG THỂ rút gọn (x_new - y) thành -t*grad.
                    # Do đó, bất đẳng thức Descent Lemma (Lipschitz Upper Bound) phải được tính đầy đủ.
                    # Mỗi lần thử t, máy tính buộc phải thực hiện O(d) phép tính mảng (diff_w, dot_grad).
                    f_new = ctx["f_val"](x_new_w, x_new_b)
                    diff_w = x_new_w - y_w_bt
                    diff_b = x_new_b - y_b_bt
                    diff_sq = float(np.sum(diff_w**2) + diff_b**2)
                    dot_grad = float(np.dot(gw_bt, diff_w) + gb_bt * diff_b)
                    
                    if f_new <= f_y + dot_grad + (1.0 / (2 * t)) * diff_sq + 1e-12:
                        break
                    t *= self.beta
                    if t < 1e-12: break
                
                self.lr = t
                self.s = s_next

        self.x_prev_w = x_w.copy()
        self.x_prev_b = x_b
        return x_new_w, x_new_b
        
    def next_s(self, s, ratio):
        return (1.0 + math.sqrt(1.0 + 4 * s**2 * ratio)) / 2.0


class Newton:
    name = "newton"
    def __init__(self, backtracking: bool = False, beta: float = 0.5, alpha: float = 1e-4):
        self.backtracking = backtracking
        self.beta = beta
        self.alpha = alpha
        self.lr = 1.0  # t=1 is standard for Newton

    def reset(self, w_shape, b_shape):
        pass

    def step(self, ctx):
        if not ctx["smooth"]:
            raise ValueError("Newton method is mathematically invalid for non-smooth L1 objectives (soft-thresholding cannot apply to scaled metrics).")
            
        w, b = ctx["w"], ctx["b"]
        F0 = ctx["F_val"](w, b)
        H, gw, gb = ctx["hessian_wb"](w, b)
        # Note: hessian_wb returns g_w already including L2 grad if applicable
        
        try:
            d_w = np.linalg.solve(H, -gw)
        except np.linalg.LinAlgError:
            d_w = -gw # fallback to gradient if singular
        d_b = -gb # simple identity block for bias
        
        t = 1.0
        if self.backtracking:
            ddot = float(np.dot(gw, d_w) + gb * d_b)
            while True:
                w_new = w + t * d_w
                b_new = b + t * d_b
                F_new = ctx["F_val"](w_new, b_new)
                if F_new <= F0 + self.alpha * t * ddot + 1e-12:
                    break
                t *= self.beta
                if t < 1e-12: break
        
        self.lr = t
        return w + t * d_w, b + t * d_b


class SGD:
    name = "sgd"
    def __init__(self, lr: float = 1e-2, schedule: str = "fixed"):
        self.lr_0 = lr
        self.schedule = schedule
        self.k = 1

    def reset(self, w_shape, b_shape):
        self.k = 1

    def step(self, ctx):
        w, b = ctx["w"], ctx["b"]
        
        if self.schedule == "diminishing":
            t = self.lr_0 / math.sqrt(self.k)
        else:
            t = self.lr_0
            
        self.k += 1
        
        if ctx["smooth"]:
            # Smooth SGD
            gw = ctx["grad_w_batch"] + ctx["reg_grad"](w)
            gb = ctx["grad_b_batch"]
            w_new = w - t * gw
            b_new = b - t * gb
        else:
            # Proximal SGD
            gw = ctx["grad_w_batch"]
            gb = ctx["grad_b_batch"]
            w_new = ctx["prox"](w - t * gw, t * ctx["lam"])
            b_new = b - t * gb
            
        return w_new, b_new


def build_optimizer(name: str, lr: float = 1e-2, backtracking: bool = False, schedule: str = "fixed", **kwargs):
    name = name.lower()
    if name == "gd":
        return GD(lr=lr, backtracking=backtracking)
    if name == "nag":
        return NAG(lr=lr, backtracking=backtracking)
    if name == "newton":
        return Newton(backtracking=backtracking)
    if name == "sgd":
        return SGD(lr=lr, schedule=schedule)
    raise ValueError(f"Unknown optimizer: {name}")

FULL_BATCH_OPTIMIZERS = {"gd", "nag", "newton"}
NO_L1_OPTIMIZERS = {"newton"}
