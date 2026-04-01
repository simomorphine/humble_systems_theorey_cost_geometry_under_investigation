"""
Pendulum-v1 environment wrapper.
"""

import numpy as np
from typing import Any, Dict, Optional, Tuple
from .base_env import BaseEnvironment
from configs.config import PendulumConfig


class PendulumEnvironment(BaseEnvironment):

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__("Pendulum-v1", render_mode=render_mode)
        self.max_steps  = 200
        self.cfg = PendulumConfig()
        # Create discrete torque bins for action mapping
        n_actions = self.cfg.N_ACTIONS
        self.torque_bins = np.linspace(-self.cfg.TORQUE_MAX, self.cfg.TORQUE_MAX, n_actions)

    # ── State info ─────────────────────────────────────────────────────────────

    def get_state_info(self, obs: np.ndarray) -> Dict[str, float]:
        cos_theta, sin_theta, theta_dot = obs
        return {
            "angle (rad)":      float(np.arctan2(sin_theta, cos_theta)),
            "angular_velocity": float(theta_dot),
            "cos(theta)":       float(cos_theta),
            "sin(theta)":       float(sin_theta),
        }

    # ── Action conversion: discrete index → continuous torque ──────────────────

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Override step to convert discrete action index to continuous torque."""
        # Convert discrete action index to continuous torque
        if isinstance(action, (int, np.integer)):
            continuous_action = np.array([self.torque_bins[int(action)]], dtype=np.float32)
        else:
            continuous_action = np.array([float(np.squeeze(action))], dtype=np.float32)
        
        # Call parent step with continuous action
        return super().step(continuous_action)

    # ── HST utility ────────────────────────────────────────────────────────────

    def compute_utility(
        self,
        state: np.ndarray,
        next_state: np.ndarray,
        action_idx: Any,
        done: bool
    ) -> Tuple[float, float]:
        """
        Compute complex utility components for Pendulum-v1.
 
        Args:
            state      : current state  [cos(θ), sin(θ), θ̇]
            next_state : next state     [cos(θ'), sin(θ'), θ̇']
            action_idx : discrete action index (used to compute torque)
            done       : episode terminated
 
        Returns:
            cost (float) : real part      c ≥ 0, physics-based deviation cost
            debt (float) : imaginary part d = H(s') - H(s)
 
        ── REAL PART: physics-based cost ────────────────────────────────────────
 
            State encoding:
                Pendulum state = [cos(θ), sin(θ), θ̇]
                → θ = atan2(sin(θ), cos(θ)) = atan2(state[1], state[0])
                → cos(θ) = state[0]  (1=upright, -1=hanging)
                → θ_norm = θ / π ∈ [-1, 1]  (0=upright, ±1=inverted)
 
            Cost formula:
                c = θ_norm'²  +  0.1·(θ̇' / θ̇_max)²  +  0.01·(u / u_max)²
 
                Term 1: (θ_norm')²    ∈ [0, 1]
                    Squared normalised angle deviation.
                    0 = perfectly upright,  1 = fully inverted.
 
                Term 2: 0.1·(θ̇'/θ̇_max)²   ∈ [0, 0.1]
                    Penalises angular velocity even when upright.
                    Weight 0.1 makes angle the primary objective.
 
                Term 3: 0.01·(u/u_max)²   ∈ [0, 0.01]
                    Small torque penalty — encourages energy efficiency.
 
        ── IMAGINARY PART: information debt ─────────────────────────────────────
 
            H(s) = -cos(θ) = -state[0]
 
            debt = H(s') - H(s) = -cos(θ') + cos(θ) = cos(θ) - cos(θ')
 
                debt > 0  →  cos(θ') < cos(θ)  →  moved AWAY from upright  (bad)
                debt < 0  →  cos(θ') > cos(θ)  →  moved TOWARD upright     (good)
                debt = 0  →  no change in upright-ness
        """
        # ── Next-state angle (reconstruct from cos/sin) ───────────────────────
        cos_theta_p = next_state[0]   # cos(θ')
        sin_theta_p = next_state[1]   # sin(θ')
        theta_dot_p = next_state[2]   # θ̇'
 
        theta_p      = np.arctan2(sin_theta_p, cos_theta_p)   # θ' ∈ [-π, π]
        theta_norm_p = theta_p / np.pi                         # ∈ [-1, 1]
 
        # ── Action to torque mapping ───────────────────────────────────────────
        if isinstance(action_idx, (int, np.integer)):
            u = self.torque_bins[int(action_idx)]
        else:
            u = float(np.squeeze(action_idx))
 
        # ── Real part: physics-based deviation cost ───────────────────────────
        c_angle  = theta_norm_p ** 2                              # angle deviation
        c_vel    = 0.1 * (theta_dot_p / self.cfg.THETA_DOT_MAX) ** 2  # angular velocity
        c_torque = 0.01 * (u / self.cfg.TORQUE_MAX) ** 2              # torque efficiency
        cost     = float(c_angle + c_vel + c_torque)
 
        # ── Imaginary part: information debt  d = H(s') - H(s) ───────────────
        H_s  = -state[0]       # -cos(θ)
        H_sp = -next_state[0]  # -cos(θ')
        debt = float(H_sp - H_s)  # = cos(θ) - cos(θ')
 
        return cost, debt

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
