"""
Pendulum-v1 environment wrapper.
"""

import numpy as np
from typing import Any, Dict, Optional, Tuple
from .base_env import BaseEnvironment


class PendulumEnvironment(BaseEnvironment):

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__("Pendulum-v1", render_mode=render_mode)
        self.max_steps  = 200
        self.torque_max = 2.0

    # ── State info ─────────────────────────────────────────────────────────────

    def get_state_info(self, obs: np.ndarray) -> Dict[str, float]:
        cos_theta, sin_theta, theta_dot = obs
        return {
            "angle (rad)":      float(np.arctan2(sin_theta, cos_theta)),
            "angular_velocity": float(theta_dot),
            "cos(theta)":       float(cos_theta),
            "sin(theta)":       float(sin_theta),
        }

    # ── HST utility ────────────────────────────────────────────────────────────

    def compute_utility(
        self,
        obs:      np.ndarray,
        next_obs: np.ndarray,
        action:   Any,         # continuous torque or discrete bin index
    ) -> Tuple[float, float]:
        """
        Cost  — torque applied (normalised to [0, 1])
        Debt  — distance from upright equilibrium at next_obs:
                angle from vertical + angular velocity magnitude
        """
        # Action may be a bin index (int) or continuous torque (array)
        if isinstance(action, (int, np.integer)):
            # map discrete bin → torque in [-torque_max, torque_max]
            import numpy as _np
            from configs.config import PendulumConfig
            n = PendulumConfig.N_ACTIONS
            torque = -self.torque_max + 2 * self.torque_max * action / (n - 1)
        else:
            torque = float(np.squeeze(action))

        cost = abs(torque) / self.torque_max   # normalised energy spent

        # Upright equilibrium: theta=0 (cos=1, sin=0), theta_dot=0
        cos_theta, sin_theta, theta_dot = next_obs
        angle_from_upright = abs(float(np.arctan2(sin_theta, cos_theta)))
        debt = angle_from_upright + 0.1 * abs(float(theta_dot))

        return float(cost), float(debt)

    # ── Dashboard panel ────────────────────────────────────────────────────────

    def get_dashboard_panel(self, obs: np.ndarray) -> Dict[str, Any]:
        cos_theta, sin_theta, theta_dot = obs
        return {
            "type":       "pendulum",
            "angle":      float(np.arctan2(sin_theta, cos_theta)),
            "cos_theta":  float(cos_theta),
            "sin_theta":  float(sin_theta),
            "theta_dot":  float(theta_dot),
        }
