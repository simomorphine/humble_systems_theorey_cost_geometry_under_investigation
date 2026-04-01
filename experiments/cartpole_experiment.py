"""
CartPole experiment.
This script is the ONLY place where config, env, agent, and runner meet.
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from configs.config  import CartpoleConfig
from envs            import ENV_REGISTRY
from agents          import AGENT_REGISTRY
from training.runner import train, evaluate
from dashboard.app   import start_dashboard
from dashboard.data_collector import get_data_collector

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg   = CartpoleConfig()
env   = ENV_REGISTRY[cfg.ENV_NAME]()
agent = AGENT_REGISTRY["complex"](cfg, DEVICE)

start_dashboard()   # opens http://127.0.0.1:5000

collector = get_data_collector()

print(f"Training {agent.name} on {cfg.ENV_NAME} — {cfg.N_EPISODES} episodes")
results = train(env, agent, cfg, collector=collector)

print("\nEvaluating...")
eval_result = evaluate(env, agent, cfg)
print(f"Mean reward : {eval_result.mean_reward:.2f} ± {eval_result.std_reward:.2f}")
print(f"Success rate: {eval_result.success_rate:.1%}")

env.close()
