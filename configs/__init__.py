from .config import CartpoleConfig, PendulumConfig, LunarLanderConfig, TetrisConfig

CONFIGS_REGISTRY = {
    "cartpole":    CartpoleConfig,
    "pendulum":    PendulumConfig,
    "lunarlander": LunarLanderConfig,
    "tetris":      TetrisConfig,
}

__all__ = ["CartpoleConfig", "PendulumConfig", "LunarLanderConfig", "TetrisConfig"]