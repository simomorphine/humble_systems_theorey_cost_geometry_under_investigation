"""
Base agent — every agent in the lab inherits from this.

Concrete subclasses must implement:
    select_action(state, training) → int
    update()                       → float | None
"""

import abc
import numpy as np
import torch


class BaseAgent(abc.ABC):

    def __init__(self, cfg, device: torch.device):
        self.cfg     = cfg
        self.device  = device
        self.epsilon = cfg.EPS_START
        self.steps   = 0
        self.name    = "BaseAgent"

    # ── Abstract interface ─────────────────────────────────────────────────────

    @abc.abstractmethod
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Return a discrete action index."""

    @abc.abstractmethod
    def update(self) -> float | None:
        """
        One gradient step on a sampled mini-batch.
        Returns scalar loss, or None if buffer not ready yet.
        """

    # ── Shared concrete behaviour ──────────────────────────────────────────────

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        cost:       float,
        debt:       float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        self.buffer.push(state, action, reward, cost, debt, next_state, done)

    def sync_target(self) -> None:
        """Hard update — copy online weights to target network."""
        self.q_targ.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.cfg.EPS_END, self.epsilon * self.cfg.EPS_DECAY)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(ε={self.epsilon:.3f}, steps={self.steps})"
