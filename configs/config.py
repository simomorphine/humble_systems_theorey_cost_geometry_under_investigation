"""
Experiment configurations — one class per environment.
Only numbers and strings live here. Zero logic.
"""


class CartpoleConfig:
    ENV_NAME        = "CartPole-v1"
    N_STATES        = 4
    N_ACTIONS       = 2
    N_EPISODES      = 600
    MAX_STEPS       = 500
    GAMMA           = 0.99
    LR              = 1e-3
    BATCH_SIZE      = 64
    REPLAY_CAPACITY = 50_000
    TARGET_UPDATE   = 10
    HIDDEN_DIM      = 128
    EPS_START       = 1.0
    EPS_END         = 0.02
    EPS_DECAY       = 0.995
    EVAL_EVERY      = 20
    EVAL_EPISODES   = 10
    N_SEEDS         = 3
    SMOOTH_W        = 20
    SUCCESS_THRESHOLD = 195.0


class PendulumConfig:
    ENV_NAME        = "Pendulum-v1"
    N_STATES        = 3
    N_ACTIONS       = 11
    TORQUE_MAX      = 2.0
    THETA_DOT_MAX   = 8.0
    N_EPISODES      = 800
    MAX_STEPS       = 200
    GAMMA           = 0.99
    LR              = 5e-4
    BATCH_SIZE      = 128
    REPLAY_CAPACITY = 100_000
    TARGET_UPDATE   = 20
    HIDDEN_DIM      = 256
    EPS_START       = 1.0
    EPS_END         = 0.02
    EPS_DECAY       = 0.997
    EVAL_EVERY      = 20
    EVAL_EPISODES   = 10
    N_SEEDS         = 3
    SMOOTH_W        = 25
    SUCCESS_THRESHOLD = -200.0


class LunarLanderConfig:
    ENV_NAME        = "LunarLander-v3"
    N_STATES        = 8
    N_ACTIONS       = 4
    FUEL_WEIGHT     = 1.0
    KINEMATIC_WEIGHT = 0.1
    N_EPISODES      = 1000
    MAX_STEPS       = 1000
    GAMMA           = 0.99


class TetrisConfig:
    ENV_NAME        = "TetrisA-v0"
    N_STATES        = 2   # filled_ratio + line_value
    N_ACTIONS       = 6   # discrete moves in gym-tetris
    N_EPISODES      = 500
    MAX_STEPS       = 2000
    GAMMA           = 0.99
    LR              = 1e-3
    BATCH_SIZE      = 64
    REPLAY_CAPACITY = 100_000
    TARGET_UPDATE   = 10
    HIDDEN_DIM      = 256
    EPS_START       = 1.0
    EPS_END         = 0.05
    EPS_DECAY       = 0.995
    EVAL_EVERY      = 20
    EVAL_EPISODES   = 5
    N_SEEDS         = 3
    SMOOTH_W        = 20
    SUCCESS_THRESHOLD = 1000.0
    LR              = 1e-3
    BATCH_SIZE      = 64
    REPLAY_CAPACITY = 100_000
    TARGET_UPDATE   = 10
    HIDDEN_DIM      = 128
    EPS_START       = 1.0
    EPS_END         = 0.02
    EPS_DECAY       = 0.995
    EVAL_EVERY      = 50
    EVAL_EPISODES   = 10
    N_SEEDS         = 3
    SMOOTH_W        = 20
    SUCCESS_THRESHOLD = 200.0
    EPS_DECAY       = 0.997
    EVAL_EVERY      = 25
    EVAL_EPISODES   = 10
    N_SEEDS         = 3
    SMOOTH_W        = 30
    SUCCESS_THRESHOLD = 200.0
    FUEL_WEIGHT       = 1.0
    KINEMATIC_WEIGHT  = 0.1
