"""
LunarLander experiment.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from configs.config  import LunarLanderConfig
from envs            import ENV_REGISTRY
from agents          import AGENT_REGISTRY
from training.runner import train, evaluate
from dashboard.app   import start_dashboard

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg   = LunarLanderConfig()
env   = ENV_REGISTRY[cfg.ENV_NAME](
    fuel_weight=cfg.FUEL_WEIGHT,
    kinematic_weight=cfg.KINEMATIC_WEIGHT,
)
agent = AGENT_REGISTRY["complex"](cfg, DEVICE)

start_dashboard()

print(f"Training {agent.name} on {cfg.ENV_NAME} — {cfg.N_EPISODES} episodes")
results = train(env, agent, cfg)

print("\nEvaluating...")
eval_result = evaluate(env, agent, cfg)
print(f"Mean reward : {eval_result.mean_reward:.2f} ± {eval_result.std_reward:.2f}")
print(f"Success rate: {eval_result.success_rate:.1%}")

env.close()
