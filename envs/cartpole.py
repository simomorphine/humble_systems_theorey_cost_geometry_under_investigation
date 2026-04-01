"""
CartPole-v1 environment wrapper.
"""

import numpy as np
from typing import Any, Dict, Optional, Tuple
from .base_env import BaseEnvironment


class CartPoleEnvironment(BaseEnvironment):

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__("CartPole-v1", render_mode=render_mode)

    # ── State info ─────────────────────────────────────────────────────────────

    def get_state_info(self, obs: np.ndarray) -> Dict[str, float]:
        return {
            "cart_position":         float(obs[0]),
            "cart_velocity":         float(obs[1]),
            "pole_angle":            float(obs[2]),
            "pole_angular_velocity": float(obs[3]),
        }

    # ── HST utility ────────────────────────────────────────────────────────────

    def compute_utility(
        self,
        obs:      np.ndarray,
        next_obs: np.ndarray,
        action:   int,
    ) -> Tuple[float, float]:
        """
        Cost  — control effort: |Δcart_velocity|
        Debt  — imbalance at next_obs: pole angle + ang_vel + cart displacement
                Spiked to 10.0 if pole has fallen (episode end with no reward)
        """
        cost = abs(float(next_obs[1]) - float(obs[1]))

        pole_angle   = abs(float(next_obs[2]))
        pole_ang_vel = abs(float(next_obs[3]))
        cart_pos     = abs(float(next_obs[0]))
        debt = pole_angle + 0.5 * pole_ang_vel + 0.1 * cart_pos

        return cost, debt

    # ── Dashboard panel ────────────────────────────────────────────────────────

    def get_dashboard_panel(self, obs: np.ndarray) -> Dict[str, Any]:
        return {
            "type":          "cartpole",
            "cart_position": float(obs[0]),
            "cart_velocity": float(obs[1]),
            "pole_angle":    float(obs[2]),
            "pole_ang_vel":  float(obs[3]),
        }
