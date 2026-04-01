"""
LunarLander-v3 environment wrapper.
"""

import numpy as np
from typing import Any, Dict, Optional, Tuple
from .base_env import BaseEnvironment


_FUEL = {0: 0.0, 1: 0.3, 2: 1.0, 3: 0.3}


class LunarLanderEnvironment(BaseEnvironment):

    def __init__(
        self,
        render_mode: Optional[str] = None,
        continuous:  bool = False,
        fuel_weight:      float = 1.0,
        kinematic_weight: float = 0.1,
    ):
        env_name = "LunarLander-v3"
        super().__init__(env_name, render_mode=render_mode)
        self.continuous       = continuous
        self.max_steps        = 1000
        self.fuel_weight      = fuel_weight
        self.kinematic_weight = kinematic_weight

    # ── State info ─────────────────────────────────────────────────────────────

    def get_state_info(self, obs: np.ndarray) -> Dict[str, float]:
        return {
            "x_position":        float(obs[0]),
            "y_position":        float(obs[1]),
            "x_velocity":        float(obs[2]),
            "y_velocity":        float(obs[3]),
            "angle":             float(obs[4]),
            "angular_velocity":  float(obs[5]),
            "left_leg_contact":  float(obs[6]),
            "right_leg_contact": float(obs[7]),
        }

    # ── HST utility ────────────────────────────────────────────────────────────

    def compute_utility(
        self,
        obs:      np.ndarray,
        next_obs: np.ndarray,
        action:   any,
        done:     bool,
    ) -> Tuple[float, float]:
        """
        Cost  — fuel burned + kinematic jolt (|Δvx| + |Δvy|)
        Debt  — imbalance at next_obs: positional + kinematic + angular
                Zeroed when both legs touch down (equilibrium reached)
        """
        fuel          = _FUEL.get(int(action), 0.0) * self.fuel_weight
        delta_vx      = abs(float(next_obs[2]) - float(obs[2]))
        delta_vy      = abs(float(next_obs[3]) - float(obs[3]))
        cost          = fuel + self.kinematic_weight * (delta_vx + delta_vy)

        left_leg  = bool(next_obs[6])
        right_leg = bool(next_obs[7])

        if left_leg and right_leg:
            debt = 0.0          # equilibrium reached — debt cleared
        else:
            positional = abs(float(next_obs[0])) + abs(float(next_obs[1]))
            kinematic  = abs(float(next_obs[2])) + abs(float(next_obs[3]))
            angular    = abs(float(next_obs[4])) + abs(float(next_obs[5]))
            debt       = positional + kinematic + angular

        return float(cost), float(debt)

    # ── Dashboard panel ────────────────────────────────────────────────────────

    def get_dashboard_panel(self, obs: np.ndarray) -> Dict[str, Any]:
        return {
            "type":       "lunarlander",
            "x":          float(obs[0]),
            "y":          float(obs[1]),
            "vx":         float(obs[2]),
            "vy":         float(obs[3]),
            "angle":      float(obs[4]),
            "ang_vel":    float(obs[5]),
            "left_leg":   bool(obs[6]),
            "right_leg":  bool(obs[7]),
        }
