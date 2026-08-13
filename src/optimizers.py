from abc import ABC, abstractmethod
import numpy as np


class BaseOptimizer(ABC):
    """Abstract base class for optimizers."""

    name: str = "base"

    @abstractmethod
    def step(self, w, b, grad_w, grad_b, **kwargs):
        """Perform one optimization step.

        Parameters
        ----------
        w, b       : current parameters
        grad_w     : gradient w.r.t. w
        grad_b     : gradient w.r.t. b
        **kwargs   : optional (hess_joint, obj_fn, prox_fn)

        Returns
        -------
        new_w, new_b : updated parameters
        """

    def reset(self, w_shape):
        """Reset internal state at start of each run."""

    def lookahead(self, w, b):
        """Return evaluation point for gradient (e.g., Nesterov lookahead)."""
        return w, b


class GradientDescent(BaseOptimizer):
    """
    Gradient Descent (full-batch).

    Update rule:
        w_{t+1} = w_t - η * ∇f(w_t)

    For L1 regularization (non-smooth), uses the proximal gradient variant:
        w_half   = w_t - η * ∇f_smooth(w_t)     (gradient step on smooth part)
        w_{t+1}  = prox_{η λ ||·||_1}(w_half)   (soft-thresholding)

    The step size η is either:
    - Fixed: FixedLR object (returns η = c/L, with optional decay)
    - Backtracking: ArmijoLineSearch object (searches along the gradient)
    """

    name = "gd"

    def __init__(self, step_size, regularizer=None, use_proximal: bool = False):
        self.step_size = step_size
        self.regularizer = regularizer
        self.use_proximal = use_proximal

    def step(self, w, b, grad_w, grad_b, **kwargs):
        if hasattr(self.step_size, "search"):
            obj_fn = kwargs.get("obj_fn")
            eta = self.step_size.search(w, b, grad_w, grad_b, obj_fn)
        else:
            eta = self.step_size.lr
            self.step_size.step()

        w_new = w - eta * grad_w
        b_new = b - eta * grad_b

        if self.use_proximal and self.regularizer is not None:
            w_new = self.regularizer.prox(w_new, eta)

        return w_new, b_new

    def reset(self, w_shape):
        self.step_size.reset()

    def __repr__(self):
        return f"GradientDescent(step_size={self.step_size}, proximal={self.use_proximal})"


class SGD(BaseOptimizer):
    """
    Mini-batch (or single-sample) Stochastic Gradient Descent.

    One SGD "step" processes a single mini-batch (or one sample when batch_size=1).
    One "epoch" = ceil(n / batch_size) such steps.

    Update rule per batch (X_b, y_b):
        g_b    = ∇f(w; X_b, y_b)    (stochastic gradient)
        w_{t+1} = w_t - η_t * g_b

    Proximal variant for L1:
        w_half   = w_t - η_t * g_b
        w_{t+1}  = prox_{η_t λ}(w_half)

    Note: Backtracking + SGD is theoretically unsound (stochastic gradient
    makes the Armijo guarantee meaningless).
    """

    name = "sgd"

    def __init__(self, step_size, regularizer=None, use_proximal: bool = False):
        self.step_size = step_size
        self.regularizer = regularizer
        self.use_proximal = use_proximal

    def step(self, w, b, grad_w, grad_b, **kwargs):
        if hasattr(self.step_size, "search"):
            obj_fn = kwargs.get("obj_fn")
            eta = self.step_size.search(w, b, grad_w, grad_b, obj_fn)
        else:
            eta = self.step_size.lr
            self.step_size.step()

        w_new = w - eta * grad_w
        b_new = b - eta * grad_b

        if self.use_proximal and self.regularizer is not None:
            w_new = self.regularizer.prox(w_new, eta)

        return w_new, b_new

    def reset(self, w_shape):
        self.step_size.reset()

    def __repr__(self):
        return f"SGD(step_size={self.step_size}, proximal={self.use_proximal})"


class NAG(BaseOptimizer):
    """
    Nesterov Accelerated Gradient (NAG) descent.

    Standard momentum update (Heavy Ball):
        v_{t+1} = μ * v_t - η * ∇f(w_t)
        w_{t+1} = w_t + v_{t+1}

    Nesterov's correction: evaluate the gradient at the "lookahead" point:
        y_t      = w_t + μ * v_t              (lookahead point)
        v_{t+1}  = μ * v_t - η * ∇f(y_t)
        w_{t+1}  = w_t + v_{t+1}

    Proximal variant for L1 (ISTA/FISTA-style):
        y_t     = w_t + μ * v_t
        g       = ∇f_smooth(y_t)
        w_half  = y_t - η * g
        w_{t+1} = prox_{η λ}(w_half)
        v_{t+1} = w_{t+1} - w_t

    Backtracking: the Armijo condition is evaluated at the lookahead point y_t.
    """

    name = "nag"

    def __init__(self, step_size, momentum: float = 0.9, regularizer=None, use_proximal: bool = False):
        self.step_size = step_size
        self.momentum = momentum
        self.regularizer = regularizer
        self.use_proximal = use_proximal
        self._v = None
        self._vb = 0.0

    def lookahead(self, w, b):
        """Return Nesterov lookahead point for gradient evaluation."""
        if self._v is None:
            return w, b
        y_look = w + self.momentum * self._v
        y_look_b = b + self.momentum * self._vb
        return y_look, y_look_b

    def step(self, w, b, grad_w, grad_b, **kwargs):
        if self._v is None:
            self._v = np.zeros_like(w)
            self._vb = 0.0

        if hasattr(self.step_size, "search"):
            obj_fn = kwargs.get("obj_fn")
            y_look, y_look_b = self.lookahead(w, b)
            eta = self.step_size.search(y_look, y_look_b, grad_w, grad_b, obj_fn)
        else:
            eta = self.step_size.lr
            self.step_size.step()

        y_look, y_look_b = self.lookahead(w, b)
        w_half = y_look - eta * grad_w
        b_new = y_look_b - eta * grad_b

        if self.use_proximal and self.regularizer is not None:
            w_new = self.regularizer.prox(w_half, eta)
        else:
            w_new = w_half

        self._v = w_new - w
        self._vb = b_new - b
        return w_new, b_new

    def reset(self, w_shape):
        self._v = np.zeros(w_shape)
        self._vb = 0.0
        self.step_size.reset()

    def __repr__(self):
        return f"NAG(momentum={self.momentum}, step_size={self.step_size}, proximal={self.use_proximal})"


class Newton(BaseOptimizer):
    """
    Newton's method with Hessian damping for numerical stability.

    Update rule:
        w_{t+1} = w_t - α * H(w_t)^{-1} ∇f(w_t)

    where H(w_t) is the Hessian of the loss at w_t, and α is the step size.

    Damping for numerical stability:
        H_damp = H + ε * I
    to ensure H_damp is positive definite and invertible.

    Notes:
    - Each step requires solving a d×d linear system (O(d³) but exact).
    - Newton is NOT run with L1 regularization (non-smooth Hessian undefined).
    - For L2 regularization, the Hessian gains an extra λI term.
    """

    name = "newton"
    requires_hessian = True

    def __init__(self, step_size, epsilon_damp: float = 1e-6):
        self.step_size = step_size
        self.epsilon_damp = epsilon_damp

    def step(self, w, b, grad_w, grad_b, **kwargs):
        hess_joint = kwargs.get("hess_joint")
        if hess_joint is None:
            raise ValueError("Newton requires 'hess_joint' in kwargs.")

        grad_joint = np.concatenate([grad_w, [grad_b]])

        diag_max = float(np.max(np.diag(hess_joint))) if len(hess_joint) > 0 else 1.0
        ridge = max(1e-8, self.epsilon_damp * max(diag_max, 1.0))
        H_reg = hess_joint + ridge * np.eye(len(grad_joint))

        try:
            step_joint = np.linalg.solve(H_reg, grad_joint)
        except np.linalg.LinAlgError:
            step_joint = np.linalg.lstsq(H_reg, grad_joint, rcond=None)[0]

        if hasattr(self.step_size, "search"):
            obj_fn = kwargs.get("obj_fn")
            eta = self.step_size.search(w, b, grad_w, grad_b, obj_fn,
                                        direction=step_joint)
        else:
            eta = self.step_size.lr
            self.step_size.step()

        new_w = w - eta * step_joint[:-1]
        new_b = b - eta * step_joint[-1]

        return new_w, new_b

    def reset(self, w_shape):
        self.step_size.reset()

    def __repr__(self):
        return f"Newton(epsilon_damp={self.epsilon_damp}, step_size={self.step_size})"


class LBFGS(BaseOptimizer):
    """
    L-BFGS optimizer (native two-loop recursion implementation).

    Preserves history vectors (s and y) across epoch steps.
    Implements Nocedal Algorithm 7.4 for the two-loop recursion.

    Parameters
    ----------
    maxiter : number of quasi-Newton steps per epoch when driven by a
        training loop (train_logreg runs ``maxiter`` steps per epoch).
        maxiter=1 gives an epoch-fair per-step comparison with GD/NAG/Newton;
        larger values let L-BFGS run closer to convergence per epoch.
    """

    name = "lbfgs"

    def __init__(self, step_size, m: int = 10, maxiter: int = 1, **kwargs):
        self.step_size = step_size
        self.m = m
        self.maxiter = max(int(maxiter), 1)
        self._s_hist = []
        self._y_hist = []
        self._rho_hist = []
        self._prev_w = None
        self._prev_b = None
        self._prev_grad_w = None
        self._prev_grad_b = None

    def step(self, w, b, grad_w, grad_b, **kwargs):
        theta = np.concatenate([w, [b]])
        grad_joint = np.concatenate([grad_w, [grad_b]])

        if self._prev_w is not None:
            prev_theta = np.concatenate([self._prev_w, [self._prev_b]])
            prev_grad_joint = np.concatenate([self._prev_grad_w, [self._prev_grad_b]])

            s_k = theta - prev_theta
            y_k = grad_joint - prev_grad_joint

            sy = np.dot(s_k, y_k)
            if sy > 1e-10:
                self._s_hist.append(s_k)
                self._y_hist.append(y_k)
                self._rho_hist.append(1.0 / sy)
                if len(self._s_hist) > self.m:
                    self._s_hist.pop(0)
                    self._y_hist.pop(0)
                    self._rho_hist.pop(0)

        q = grad_joint.copy()
        alphas = []
        for i in reversed(range(len(self._s_hist))):
            alpha = self._rho_hist[i] * np.dot(self._s_hist[i], q)
            alphas.append(alpha)
            q -= alpha * self._y_hist[i]

        if len(self._s_hist) > 0:
            gamma = np.dot(self._s_hist[-1], self._y_hist[-1]) / np.dot(self._y_hist[-1], self._y_hist[-1])
            z = gamma * q
        else:
            z = q

        alphas.reverse()
        for i in range(len(self._s_hist)):
            beta = self._rho_hist[i] * np.dot(self._y_hist[i], z)
            z += self._s_hist[i] * (alphas[i] - beta)

        direction_joint = z

        if hasattr(self.step_size, "search"):
            obj_fn = kwargs.get("obj_fn")
            eta = self.step_size.search(w, b, grad_w, grad_b, obj_fn,
                                        direction=direction_joint)
        else:
            eta = self.step_size.lr
            self.step_size.step()

        w_new = w - eta * direction_joint[:-1]
        b_new = b - eta * direction_joint[-1]

        self._prev_w = w.copy()
        self._prev_b = b
        self._prev_grad_w = grad_w.copy()
        self._prev_grad_b = grad_b

        return w_new, b_new

    def reset(self, w_shape):
        self._s_hist = []
        self._y_hist = []
        self._rho_hist = []
        self._prev_w = None
        self._prev_b = None
        self._prev_grad_w = None
        self._prev_grad_b = None
        self.step_size.reset()

    def __repr__(self):
        return f"LBFGS(m={self.m}, maxiter={self.maxiter}, step_size={self.step_size})"


OPTIMIZER_REGISTRY = {
    "gd": GradientDescent,
    "sgd": SGD,
    "nag": NAG,
    "newton": Newton,
    "lbfgs": LBFGS,
}