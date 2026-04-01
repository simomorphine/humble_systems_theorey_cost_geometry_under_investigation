"""
Main experiment runner — combines all environments and agents.
Run with: python experiments/main_experiment.py --env cartpole --agent complex
"""

import sys, os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from configs         import CONFIGS_REGISTRY
from envs            import ENV_REGISTRY
from agents          import AGENT_REGISTRY
from training.runner import train, evaluate
from dashboard.app   import start_dashboard
from dashboard.data_collector import get_data_collector

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser(description="Run RL experiments")
    parser.add_argument("--env", type=str, required=True, 
                       choices=["cartpole", "lunarlander", "pendulum", "tetris"],
                       help="Environment to run")
    parser.add_argument("--agent", type=str, default="complex",
                       choices=["complex", "standard"],  # Add more if available
                       help="Agent type")
    args = parser.parse_args()

    cfg   = CONFIGS_REGISTRY[args.env]()
    env   = ENV_REGISTRY[cfg.ENV_NAME]()
    agent = AGENT_REGISTRY[args.agent](cfg, DEVICE)

    start_dashboard()   # opens http://127.0.0.1:5000

    collector = get_data_collector()

    print(f"Training {agent.name} on {cfg.ENV_NAME} — {cfg.N_EPISODES} episodes")
    results = train(env, agent, cfg, collector=collector)

    print("\nEvaluating...")
    eval_result = evaluate(env, agent, cfg)
    print(f"Mean reward : {eval_result.mean_reward:.2f} ± {eval_result.std_reward:.2f}")

    env.close()

if __name__ == "__main__":
    main()