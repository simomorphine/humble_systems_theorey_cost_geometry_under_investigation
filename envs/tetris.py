"""
Tetris environment wrapper (gym-tetris). Uses image-based observations and line/reward-derived utility.
"""

import numpy as np
from typing import Any, Dict, Optional, Tuple
from .base_env import BaseEnvironment

try:
    import gym
    import gym_tetris
except ImportError as e:
    raise ImportError("gym-tetris is required for Tetris env; install gym-tetris==3.0.4") from e


def _filled_ratio(obs: np.ndarray) -> float:
    # obs typically [H, W, 3] uint8 image; use grayscale and occupancy threshold
    if obs is None:
        return 0.0
    frame = np.asarray(obs, dtype=np.float32)
    if frame.ndim == 3:
        gray = frame.mean(axis=2)
    else:
        gray = frame
    occupied = np.maximum(0.0, np.minimum(1.0, (255.0 - gray) / 255.0))
    return float(np.mean(occupied))


class TetrisEnvironment(BaseEnvironment):

    def __init__(self, render_mode: Optional[str] = None):
        # gym-tetris uses TetrisA-v0/TetrisB-v0; we choose A-type default.
        self.cfg_env = "TetrisA-v0"
        self.max_steps = 2000
        self.env = gym.make(self.cfg_env)
        super().__init__(self.cfg_env, render_mode=render_mode)

    def get_state_info(self, obs: np.ndarray) -> Dict[str, float]:
        filled = _filled_ratio(obs)
        return {
            "filled_ratio": filled,
        }

    def compute_utility(
        self,
        state: np.ndarray,
        next_state: np.ndarray,
        action_idx: Any,
        done: bool
    ) -> Tuple[float, float]:
        fill_current = _filled_ratio(state)
        fill_next = _filled_ratio(next_state)

        # cost: higher occupancy is worse
        cost = float(fill_next)

        # debt: increase in filled ratio (bad if positive)
        debt = float(fill_next - fill_current)

        return cost, debt

    def get_dashboard_panel(self, obs: np.ndarray) -> Dict[str, Any]:
        filled = _filled_ratio(obs)
        return {
            "type": "tetris",
            "filled_ratio": filled,
        }
