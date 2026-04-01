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

    def compute_utility(self,
                    state: np.ndarray,
                    next_state: np.ndarray,
                    action: any,
                    done: bool) -> tuple[float, float]:
        """
            Compute complex utility components for CartPole.
 
            Args:
                state      : current state  [x, ẋ, θ, θ̇]
                next_state : next state     [x, ẋ, θ, θ̇]
                reward     : gym reward (unused — kept for API symmetry with Standard DQN)
                done       : episode terminated (pole fell or cart out of bounds)
        
            Returns:
                cost (float) : real part      c  ≥ 0, physics-based deviation cost
                debt (float) : imaginary part d  = H(s') - H(s)
        
            Cost formula (physics-based, replaces naive c = -reward):
                THETA_FAIL = 0.2094 rad (12°)   X_FAIL = 2.4 m
                c = (θ' / THETA_FAIL)² + 0.1·(x' / X_FAIL)² + 10·1[done]
        
                Rationale: CartPole reward ≡ 1 every step, so c = -reward = -1
                always.  Re(G) = -Σγᵏ then grows more negative with episode
                length → |Re(G)| is LARGER for longer episodes → agent learns
                to die immediately (minimising |G| by minimising T).  The
                physics-based cost avoids this: c≈0 when pole is upright,
                c→1 near failure angle, +10 on terminal.  Surviving balanced
                accumulates near-zero cost → small Re(G) → small |G| ✓
        
            Entropy proxy:
                H(s) = -|θ + θ̇|
                θ  = state[2]  (pole angle, radians)
                θ̇ = state[3]  (pole angular velocity, rad/s)
        
            Interpretation:
                - cost ≈ 0  when pole is perfectly upright (θ'≈0, x'≈0)
                - cost large near failure boundary or on terminal step
                - debt > 0  when |θ+θ̇| decreased  (pole stabilised — good)
                - debt < 0  when |θ+θ̇| increased  (pole destabilised — bad)
                - At perfect balance: cost≈0, debt≈0, |z|≈0  ← ideal
                - |z| = √(cost² + debt²)  is what complex Q-learning minimises
        """
        # ── Physics-based cost ────────────────────────────────────────────
        THETA_FAIL      = 0.2094   # 12 degrees in radians (CartPole failure)
        X_FAIL          = 2.4      # metres (CartPole failure)
        TERMINAL_PENALTY = 10.0   # large spike on pole falling / cart leaving
    
        theta_next = next_state[2]
        x_next     = next_state[0]
    
        c_angle = (theta_next / THETA_FAIL) ** 2          # 0=vertical  1=at limit
        c_cart  = (x_next     / X_FAIL)     ** 2          # 0=centre    1=at edge
        cost    = c_angle + 0.1 * c_cart + (TERMINAL_PENALTY if done else 0.0)
    
        # ── Information debt  d = H(s') - H(s) ───────────────────────────
        H_s  = -abs(state[2]      + state[3])       # -|θ + θ̇|
        H_sp = -abs(next_state[2] + next_state[3])  # -|θ' + θ̇'|
        debt = float(H_sp - H_s)  # >0 pole stabilised  <0 pole destabilised
    
        return float(cost), debt

    # ── Dashboard panel ────────────────────────────────────────────────────────

    def get_dashboard_panel(self, obs: np.ndarray) -> Dict[str, Any]:
        return {
            "type":          "cartpole",
            "cart_position": float(obs[0]),
            "cart_velocity": float(obs[1]),
            "pole_angle":    float(obs[2]),
            "pole_ang_vel":  float(obs[3]),
        }
