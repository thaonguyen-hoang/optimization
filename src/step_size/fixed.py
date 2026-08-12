"""
Fixed learning rate strategy.

The learning rate is set to η = c / L where:
  c  : scaling constant from YAML config
  L  : Lipschitz constant of the gradient (computed from data)

For SGD an optional multiplicative decay schedule is applied:
  η_t = η_0 / (1 + decay_rate * t)   where t is the global step count.
"""


class FixedLR:
    """Fixed learning rate with optional inverse-time decay."""

    def __init__(self, c: float, L: float, decay_rate: float = 0.0):
        if L <= 0:
            raise ValueError(f"Lipschitz constant L must be > 0, got {L}")
        self.eta0 = c / L
        self.decay_rate = decay_rate
        self._step = 0

    @property
    def lr(self):
        """Current learning rate (accounts for decay)."""
        return self.eta0 / (1.0 + self.decay_rate * self._step)

    def step(self):
        """Advance the internal step counter."""
        self._step += 1

    def reset(self):
        self._step = 0

    def __repr__(self):
        return f"FixedLR(eta0={self.eta0:.6g}, decay_rate={self.decay_rate})"

