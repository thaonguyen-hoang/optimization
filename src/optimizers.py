"""Context-based first- and second-order optimizers."""

from abc import ABC, abstractmethod
import math

import numpy as np


FULL_BATCH_OPTIMIZERS = {"gd", "nag", "newton"}
NO_L1_OPTIMIZERS      = {"newton"}


class Optimizer(ABC):
    name = "optimizer"

    @abstractmethod
    def step(self, ctx): ...

    def reset(self, w_shape, b_shape=()):
        """Reset any per-run state (called once before training starts)."""
        pass


# ─── Gradient Descent ────────────────────────────────────────────────────────

class GD(Optimizer):
    """
    Gradient Descent / ISTA — full-batch, first-order.

    Smooth (L2 / None):
        fixed      : w ← w - t * ∇F(w)
        backtracking: Armijo on F — F(w - t·g) ≤ F(w) - α·t·‖g‖²

    Non-smooth (L1 / ISTA):
        fixed      : w ← prox(w - t * ∇f(w), t·λ)
        backtracking: ISTA-BT — proximal descent lemma on f
    """

    name = "gd"

    def __init__(
        self,
        lr=1e-2,
        backtracking=False,
        beta=0.5,
        alpha=1e-4,
        initial_lr=1.0,
        max_backtracks=60,
    ):
        self.lr            = lr
        self.backtracking  = bool(backtracking)
        self.beta          = float(beta)
        self.alpha         = float(alpha)
        self.initial_lr    = float(initial_lr)
        self.max_backtracks = int(max_backtracks)

    def __repr__(self):
        return (f"GD(lr={self.lr}, backtracking={self.backtracking}, "
                f"beta={self.beta}, alpha={self.alpha})")

    def step(self, ctx):
        w  = ctx["w"]
        b  = ctx["b"]
        gw = ctx["grad_w"](w, b)
        gb = ctx["grad_b"](w, b)

        if ctx["smooth"]:
            gw = gw + ctx["reg_grad"](w)

            # ── Fixed step ────────────────────────────────────────────────────
            if not self.backtracking:
                t = float(self.lr)
                return w - t * gw, b - t * gb

            # ── Armijo backtracking ───────────────────────────────────────────
            f0    = ctx["F_val"](w, b)
            norm2 = float(np.dot(gw, gw) + gb * gb)  # computed once outside loop
            t     = self.initial_lr
            for _ in range(self.max_backtracks):
                wn = w - t * gw
                bn = b - t * gb
                if ctx["F_val"](wn, bn) <= f0 - self.alpha * t * norm2 + 1e-12:
                    break
                t *= self.beta
            self.last_lr = t
            return wn, bn

        # ── L1 / ISTA — fixed step ────────────────────────────────────────────
        if not self.backtracking:
            t  = float(self.lr)
            wn = ctx["prox"](w - t * gw, t * ctx["lam"])
            bn = b - t * gb
            return wn, bn

        # ── L1 / ISTA-BT — proximal descent lemma ────────────────────────────
        f0 = ctx["f_val"](w, b)
        t  = self.initial_lr
        for _ in range(self.max_backtracks):
            wn = ctx["prox"](w - t * gw, t * ctx["lam"])
            bn = b - t * gb
            dw    = wn - w
            db    = bn - b
            bound = (f0
                     + float(np.dot(gw, dw) + gb * db)
                     + float(np.dot(dw, dw) + db * db) / (2 * t))
            if ctx["f_val"](wn, bn) <= bound + 1e-12:
                break
            t *= self.beta
        self.last_lr = t
        return wn, bn


# ─── Nesterov Accelerated Gradient ───────────────────────────────────────────

class NAG(Optimizer):
    """
    NAG (smooth) / FISTA (L1) — full-batch, accelerated first-order.

    Smooth cases (L2 / None) → NAG:
        fixed      : momentum = (k-2)/(k+1)  [Nesterov 1983, course version]
                     y  = x + momentum * (x - x_prev)
                     x' = y - t * ∇F(y)
        backtracking: s_k sequence momentum, parabol condition on F
                     F(x') ≤ F(y) - (t/2) * ‖∇F(y)‖²  (no α — computed once)

    L1 cases (composite) → FISTA:
        fixed      : same momentum, x' = prox(y - t * ∇f(y), t·λ)
        backtracking: FISTA-BT, energy-preserving s_{k+1} = f(s_k, t/t_prev)
    """

    name = "nag"

    def __init__(
        self,
        lr=1e-2,
        backtracking=False,
        beta=0.5,
        initial_lr=1.0,
        max_backtracks=60,
    ):
        self.lr             = lr
        self.backtracking   = bool(backtracking)
        self.beta           = float(beta)
        self.initial_lr     = float(initial_lr)
        self.max_backtracks = int(max_backtracks)
        self.reset_state    = True

    def __repr__(self):
        return (f"NAG(lr={self.lr}, backtracking={self.backtracking}, "
                f"beta={self.beta})")

    def reset(self, w_shape, b_shape=()):
        super().reset(w_shape, b_shape)
        self.prev_w     = np.zeros(w_shape)
        self.prev_b     = 0.0
        self.k          = 1          # step counter for fixed-step momentum
        self.s          = 1.0        # FISTA s_k sequence for backtracking
        self.prev_t     = self.initial_lr
        self.reset_state = False

    def _ensure(self, w):
        if self.reset_state or not hasattr(self, "prev_w"):
            self.reset(w.shape)

    def step(self, ctx):
        xw = ctx["w"]
        xb = ctx["b"]
        self._ensure(xw)

        # ── Fixed step (Nesterov 1983 simple momentum) ────────────────────────
        if not self.backtracking:
            momentum = (self.k - 2.0) / (self.k + 1.0)
            yw = xw + momentum * (xw - self.prev_w)
            yb = xb + momentum * (xb - self.prev_b)
            gw = ctx["grad_w"](yw, yb)
            gb = ctx["grad_b"](yw, yb)
            if ctx["smooth"]:
                gw = gw + ctx["reg_grad"](yw)
            t = float(self.lr)
            if ctx["smooth"]:
                nw = yw - t * gw
            else:
                nw = ctx["prox"](yw - t * gw, t * ctx["lam"])
            nb = yb - t * gb
            self.prev_w = xw.copy()
            self.prev_b = float(xb)
            self.k     += 1
            return nw, nb

        # ── Backtracking smooth (NAG-BT): s_k sequence + parabol condition ────
        if ctx["smooth"]:
            sn       = (1.0 + math.sqrt(1.0 + 4.0 * self.s * self.s)) / 2.0
            momentum = (self.s - 1.0) / sn
            yw = xw + momentum * (xw - self.prev_w)
            yb = xb + momentum * (xb - self.prev_b)
            gw = ctx["grad_w"](yw, yb) + ctx["reg_grad"](yw)
            gb = ctx["grad_b"](yw, yb)
            # ‖∇F(y)‖² computed once — does not depend on t (O(1) per trial)
            norm2 = float(np.dot(gw, gw) + gb * gb)
            fy    = ctx["F_val"](yw, yb)
            t     = self.initial_lr
            for _ in range(self.max_backtracks):
                nw = yw - t * gw
                nb = yb - t * gb
                if ctx["F_val"](nw, nb) <= fy - 0.5 * t * norm2 + 1e-12:
                    break
                t *= self.beta

        # ── Backtracking L1 (FISTA-BT): energy-preserving s_k update ─────────
        else:
            t = self.initial_lr
            for _ in range(self.max_backtracks):
                # s_{k+1} depends on t/t_prev — recomputed each trial
                sn       = (1.0 + math.sqrt(1.0 + 4.0 * self.s * self.s * t / self.prev_t)) / 2.0
                momentum = (self.s - 1.0) / sn
                yw = xw + momentum * (xw - self.prev_w)
                yb = xb + momentum * (xb - self.prev_b)
                gw = ctx["grad_w"](yw, yb)
                gb = ctx["grad_b"](yw, yb)
                nw = ctx["prox"](yw - t * gw, t * ctx["lam"])
                nb = yb - t * gb
                dw    = nw - yw
                db    = nb - yb
                bound = (ctx["f_val"](yw, yb)
                         + float(np.dot(gw, dw) + gb * db)
                         + float(np.dot(dw, dw) + db * db) / (2 * t))
                if ctx["f_val"](nw, nb) <= bound + 1e-12:
                    break
                t *= self.beta

        self.prev_w  = xw.copy()
        self.prev_b  = float(xb)
        self.s       = sn
        self.prev_t  = t
        self.last_lr = t
        return nw, nb


# ─── Newton ──────────────────────────────────────────────────────────────────

class Newton(Optimizer):
    """
    Newton's method — full-batch, second-order.

    Valid for smooth objectives (BCE, WeightedBCE, SquaredHinge + L2/None).
    Squared Hinge Hessian exists almost everywhere (piecewise constant).
    Not valid with L1 — raises at call time.

    fixed      : t = 1.0 (pure Newton), w ← w + t * Δw
    backtracking: Armijo on F along Newton direction d
                  F(w + t·d) ≤ F(w) + α·t·gᵀd
    """

    name = "newton"

    def __init__(
        self,
        backtracking=False,
        beta=0.5,
        alpha=1e-4,
        epsilon_damp=1e-8,
        max_backtracks=60,
        **kwargs,
    ):
        self.backtracking   = bool(backtracking)
        self.beta           = float(beta)
        self.alpha          = float(alpha)
        self.epsilon_damp   = float(epsilon_damp)
        self.max_backtracks = int(max_backtracks)
        self.lr             = 1.0   # kept for interface consistency

    def __repr__(self):
        return (f"Newton(backtracking={self.backtracking}, "
                f"alpha={self.alpha}, epsilon_damp={self.epsilon_damp})")

    def step(self, ctx):
        if not ctx["smooth"]:
            raise ValueError("Newton is invalid with L1 regularization")

        w = ctx["w"]
        b = ctx["b"]
        H, g = ctx["hessian_wb"](w, b)

        # Levenberg-Marquardt damping for numerical stability
        scale     = max(1.0, float(np.max(np.abs(np.diag(H)))))
        H_damped  = H + self.epsilon_damp * scale * np.eye(len(g))

        try:
            direction = np.linalg.solve(H_damped, -g)
        except np.linalg.LinAlgError:
            direction = np.linalg.lstsq(H_damped, -g, rcond=None)[0]

        t = 1.0
        if self.backtracking:
            f0    = ctx["F_val"](w, b)
            slope = float(np.dot(g, direction))
            if slope >= 0:
                # Direction is not a descent direction; fall back to steepest
                direction = -g
                slope     = -float(np.dot(g, g))
            for _ in range(self.max_backtracks):
                wn = w + t * direction[:-1]
                bn = b + t * direction[-1]
                if ctx["F_val"](wn, bn) <= f0 + self.alpha * t * slope + 1e-12:
                    break
                t *= self.beta

        self.last_lr = t
        return w + t * direction[:-1], b + t * direction[-1]


# ─── SGD ─────────────────────────────────────────────────────────────────────

class SGD(Optimizer):
    """
    Stochastic Gradient Descent — mini-batch, first-order.

    Smooth (L2 / None):
        fixed      : w ← w - t * (g_batch + ∇r(w))
        diminishing: t_k = t_0 / √k

    L1 (proximal SGD):
        fixed      : w ← prox(w - t * g_batch, t·λ)
        diminishing: same with diminishing t_k

    No backtracking — stochastic gradient invalidates the Armijo guarantee.
    Diminishing step is the theoretically valid alternative for SGD.
    """

    name = "sgd"

    def __init__(self, lr=1e-2, schedule="fixed"):
        if schedule not in {"fixed", "diminishing"}:
            raise ValueError("SGD schedule must be 'fixed' or 'diminishing'")
        self.lr       = lr
        self.schedule = schedule
        self.k        = 1

    def __repr__(self):
        return f"SGD(lr={self.lr}, schedule={self.schedule!r})"

    def reset(self, w_shape, b_shape=()):
        super().reset(w_shape, b_shape)
        self.k = 1

    def step(self, ctx):
        w   = ctx["w"]
        b   = ctx["b"]
        t0  = float(self.lr)
        t   = t0 / math.sqrt(self.k) if self.schedule == "diminishing" else t0
        self.k += 1

        gw = ctx["grad_w_batch"]
        gb = ctx["grad_b_batch"]

        if ctx["smooth"]:
            return w - t * (gw + ctx["reg_grad"](w)), b - t * gb
        return ctx["prox"](w - t * gw, t * ctx["lam"]), b - t * gb


# ─── Factory ─────────────────────────────────────────────────────────────────

def build_optimizer(
    name: str,
    lr=1e-2,
    backtracking=False,
    schedule="fixed",
    **kwargs,
):
    """Instantiate an optimizer by name."""
    key   = name.lower()
    beta  = kwargs.get("beta", kwargs.get("armijo_beta", 0.5))
    alpha = kwargs.get("alpha", kwargs.get("armijo_alpha", 1e-4))
    common = {
        "beta":          beta,
        "initial_lr":    kwargs.get("initial_lr", 1.0),
        "max_backtracks": kwargs.get("max_backtracks", 60),
    }

    if key == "gd":
        return GD(lr, backtracking, alpha=alpha, **common)

    if key == "nag":
        return NAG(lr, backtracking, **common)

    if key == "newton":
        return Newton(
            backtracking,
            beta=common["beta"],
            alpha=alpha,
            epsilon_damp=kwargs.get("epsilon_damp", 1e-8),
            max_backtracks=common["max_backtracks"],
        )

    if key == "sgd":
        if backtracking:
            raise ValueError(
                "SGD does not support backtracking line search. "
                "Use schedule='diminishing' instead."
            )
        return SGD(lr, schedule)

    raise ValueError(f"Unknown optimizer: {name!r}")
