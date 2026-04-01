"""
Base environment — every env in the lab inherits from this.

Concrete subclasses must implement:
    get_state_info(obs)              → Dict[str, float]
    compute_utility(obs, next_obs, action) → Tuple[float, float]  (cost, debt)
    get_dashboard_panel(obs)         → Dict[str, Any]
"""

import gymnasium as gym
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple


class BaseEnvironment(ABC):

    def __init__(self, env_name: str, render_mode: Optional[str] = None):
        self.env_name  = env_name
        self.env       = gym.make(env_name, render_mode=render_mode)

        self.action_space      = self.env.action_space
        self.observation_space = self.env.observation_space

        self.current_step: int   = 0
        self.max_steps: Optional[int] = None   # set by subclass if needed

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(seed=seed)
        self.current_step = 0
        return obs, info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    # ── Action helpers ─────────────────────────────────────────────────────────

    def get_random_action(self):
        return self.action_space.sample()

    def is_valid_action(self, action) -> bool:
        return self.action_space.contains(action)

    # ── Abstract interface ─────────────────────────────────────────────────────

    @abstractmethod
    def get_state_info(self, obs: np.ndarray) -> Dict[str, float]:
        """Human-readable breakdown of the observation."""

    @abstractmethod
    def compute_utility(
        self,
        obs:      np.ndarray,
        next_obs: np.ndarray,
        action:   Any,
    ) -> Tuple[float, float]:
        """
        HST utility for this transition.
        Returns (cost, debt) — both non-negative floats.
        cost = Re(z) — energy spent
        debt = Im(z) — imbalance remaining at next_obs
        """

    @abstractmethod
    def get_dashboard_panel(self, obs: np.ndarray) -> Dict[str, Any]:
        """
        Data the Jinja2 env-specific panel needs to render.
        Must include a 'type' key matching the panel template name.
        """

    # ── Misc ───────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(env='{self.env_name}')"
