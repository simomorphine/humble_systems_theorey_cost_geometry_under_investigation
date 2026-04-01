"""
Environment registry — adding a new env = one line here.
"""

from .cartpole     import CartPoleEnvironment
from .pendulum     import PendulumEnvironment
from .lunar_lander import LunarLanderEnvironment

ENV_REGISTRY = {
    "CartPole-v1":    CartPoleEnvironment,
    "Pendulum-v1":    PendulumEnvironment,
    "LunarLander-v3": LunarLanderEnvironment,
}

__all__ = ["ENV_REGISTRY", "CartPoleEnvironment", "PendulumEnvironment", "LunarLanderEnvironment"]
