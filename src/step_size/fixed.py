"""
fixed.py — Fixed learning rate strategy.

The learning rate is set to η = c / L where:
  c  : scaling constant from YAML config
  L  : Lipschitz constant of the gradient (computed from data)

For SGD an optional multiplicative decay schedule is applied:
  η_t = η_0 / (1 + decay_rate * t)   where t is the global step count.
"""


class FixedLR:
    """
    Fixed learning rate with optional inverse-time decay.

    Parameters
    ----------
    c          : numerator constant (from YAML sweep)
    L          : Lipschitz constant (computed at runtime)
    decay_rate : decay rate for SGD schedule (0.0 = no decay)
    """

    def __init__(self, c: float, L: float, decay_rate: float = 0.0):
        if L <= 0:
            raise ValueError(f"Lipschitz constant L must be > 0, got {L}")
        self.eta0 = c / L
        self.decay_rate = decay_rate
        self._step = 0

    @property
    def lr(self) -> float:
        """Current learning rate (accounts for decay)."""
        return self.eta0 / (1.0 + self.decay_rate * self._step)

    def step(self) -> float:
        """Return current LR then advance the internal step counter."""
        current = self.lr
        self._step += 1
        return current

    def reset(self) -> None:
        self._step = 0

    def __repr__(self) -> str:
        return (f"FixedLR(eta0={self.eta0:.6g}, "
                f"decay_rate={self.decay_rate})")
