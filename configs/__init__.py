from .config import CartpoleConfig, PendulumConfig, LunarLanderConfig

CONFIGS_REGISTRY = {
    "cartpole":    CartpoleConfig,
    "pendulum":    PendulumConfig,
    "lunarlander": LunarLanderConfig,
}

__all__ = ["CartpoleConfig", "PendulumConfig", "LunarLanderConfig"]