"""
Training runner — the ONLY place in the codebase that sees both env and agent.

Three public functions:
    run_episode(env, agent, cfg, training, render) → EpisodeResult
    train(env, agent, cfg)                         → List[EpisodeResult]
    evaluate(env, agent, cfg)                      → EvalResult
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#                          Result dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EpisodeResult:
    episode:      int
    total_reward: float
    total_cost:   float
    total_debt:   float
    mean_modulus: float          # mean |z| = mean √(cost² + debt²)
    steps:        int
    loss:         Optional[float]
    epsilon:      float
    terminated:   bool
    truncated:    bool


@dataclass
class EvalResult:
    mean_reward:  float
    std_reward:   float
    mean_length:  float
    success_rate: float
    episodes:     List[EpisodeResult] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
#                          Core functions
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(
    env,
    agent,
    cfg,
    episode:  int  = 0,
    training: bool = True,
    render:   bool = False,
) -> EpisodeResult:
    """Run one episode. Returns a clean EpisodeResult."""

    state, _ = env.reset()
    total_reward = total_cost = total_debt = 0.0
    moduli: List[float] = []
    last_loss: Optional[float] = None
    terminated = truncated = False

    for _ in range(cfg.MAX_STEPS):
        if render:
            env.render()

        action                                   = agent.select_action(state, training)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done                                     = terminated or truncated

        cost, debt = env.compute_utility(state, next_state, action)

        if training:
            agent.push(state, action, reward, cost, debt, next_state, done)
            loss = agent.update()
            if loss is not None:
                last_loss = loss

        total_reward += reward
        total_cost   += cost
        total_debt   += debt
        moduli.append(np.sqrt(cost**2 + debt**2))

        state = next_state
        if done:
            break

    if training:
        agent.decay_epsilon()

    return EpisodeResult(
        episode=episode,
        total_reward=total_reward,
        total_cost=total_cost,
        total_debt=total_debt,
        mean_modulus=float(np.mean(moduli)) if moduli else 0.0,
        steps=env.current_step,
        loss=last_loss,
        epsilon=agent.epsilon,
        terminated=terminated,
        truncated=truncated,
    )


def train(env, agent, cfg, verbose: bool = True) -> List[EpisodeResult]:
    """Full training loop. Returns all episode results."""
    results: List[EpisodeResult] = []

    for episode in range(cfg.N_EPISODES):
        result = run_episode(env, agent, cfg, episode=episode, training=True)
        results.append(result)

        # Target network sync
        if episode % cfg.TARGET_UPDATE == 0:
            agent.sync_target()

        # Logging
        if verbose and episode % 50 == 0:
            recent  = results[-50:]
            avg_r   = np.mean([r.total_reward for r in recent])
            avg_z   = np.mean([r.mean_modulus for r in recent])
            loss_str = f"{result.loss:.4f}" if result.loss else "—"
            print(
                f"[{agent.name}] ep {episode:>4} | "
                f"avg_r={avg_r:>8.2f} | "
                f"avg_|z|={avg_z:.4f} | "
                f"loss={loss_str} | "
                f"ε={agent.epsilon:.3f}"
            )

    return results


def evaluate(env, agent, cfg) -> EvalResult:
    """Evaluate agent over cfg.EVAL_EPISODES episodes (no training)."""
    results = [
        run_episode(env, agent, cfg, episode=i, training=False)
        for i in range(cfg.EVAL_EPISODES)
    ]
    rewards = [r.total_reward for r in results]
    return EvalResult(
        mean_reward=float(np.mean(rewards)),
        std_reward=float(np.std(rewards)),
        mean_length=float(np.mean([r.steps for r in results])),
        success_rate=float(np.mean([r.total_reward >= cfg.SUCCESS_THRESHOLD for r in results])),
        episodes=results,
    )
