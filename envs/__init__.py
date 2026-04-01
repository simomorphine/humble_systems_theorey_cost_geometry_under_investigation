"""
Environment registry — adding a new env = one line here.
"""

from .cartpole     import CartPoleEnvironment
from .pendulum     import PendulumEnvironment
from .lunar_lander import LunarLanderEnvironment
from .tetris       import TetrisEnvironment

ENV_REGISTRY = {
    "CartPole-v1":    CartPoleEnvironment,
    "Pendulum-v1":    PendulumEnvironment,
    "LunarLander-v3": LunarLanderEnvironment,
    "TetrisA-v0":     TetrisEnvironment,
}

__all__ = ["ENV_REGISTRY", "CartPoleEnvironment", "PendulumEnvironment", "LunarLanderEnvironment", "TetrisEnvironment"]
