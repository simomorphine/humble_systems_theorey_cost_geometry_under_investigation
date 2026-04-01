"""
Thread-safe data collector for the dashboard.
Completely env-agnostic — knows nothing about CartPole, LunarLander, etc.
All env-specific data arrives pre-formatted via state_info and panel dicts.
"""

import queue
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class EnvironmentState:
    timestamp:    float
    episode:      int
    step:         int
    env_name:     str
    state_info:   Dict[str, float]   # from env.get_state_info()
    panel:        Dict[str, Any]     # from env.get_dashboard_panel()
    action:       Any
    reward:       float
    cost:         float
    debt:         float
    modulus:      float              # √(cost² + debt²)
    total_reward: float
    terminated:   bool
    truncated:    bool


@dataclass
class AgentState:
    timestamp:     float
    episode:       int
    agent_name:    str
    epsilon:       Optional[float]
    steps:         int
    learning_rate: Optional[float]
    loss:          Optional[float]


@dataclass
class EpisodeMetrics:
    timestamp:          float
    episode:            int
    env_name:           str
    agent_name:         str
    total_reward:       float
    total_cost:         float
    total_debt:         float
    mean_modulus:       float
    episode_length:     int
    avg_reward_last_10: float
    avg_cost_last_10:   float
    avg_debt_last_10:   float
    success_rate:       float
    success_threshold:  float


class DataCollector:

    def __init__(self, max_history: int = 2000):
        self.max_history = max_history
        self.lock        = threading.Lock()

        self.env_states:      List[EnvironmentState] = []
        self.agent_states:    List[AgentState]       = []
        self.episode_metrics: List[EpisodeMetrics]   = []

        # Real-time queues for SocketIO
        self.env_queue     = queue.Queue(maxsize=500)
        self.agent_queue   = queue.Queue(maxsize=500)
        self.metrics_queue = queue.Queue(maxsize=200)

        self.current_episode = 0
        self._env_name       = ""
        self._agent_name     = ""

        # Per-episode accumulators
        self._ep_rewards: List[float] = []
        self._ep_costs:   List[float] = []
        self._ep_debts:   List[float] = []
        self._ep_lengths: List[int]   = []

    # ── Writers ───────────────────────────────────────────────────────────────

    def record_step(
        self,
        env_name:     str,
        state_info:   Dict[str, float],
        panel:        Dict[str, Any],
        action:       Any,
        reward:       float,
        cost:         float,
        debt:         float,
        total_reward: float,
        step:         int,
        terminated:   bool,
        truncated:    bool,
    ) -> None:
        import math
        self._env_name = env_name
        modulus = math.sqrt(cost**2 + debt**2)

        state = EnvironmentState(
            timestamp=time.time(),
            episode=self.current_episode,
            step=step,
            env_name=env_name,
            state_info={k: float(v) for k, v in state_info.items()},
            panel=panel,
            action=action,
            reward=float(reward),
            cost=float(cost),
            debt=float(debt),
            modulus=modulus,
            total_reward=float(total_reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
        )

        self._append(self.env_states, state)
        self._enqueue(self.env_queue, asdict(state))

    def record_agent(
        self,
        agent_name: str,
        epsilon:    Optional[float],
        steps:      int,
        lr:         Optional[float],
        loss:       Optional[float],
    ) -> None:
        self._agent_name = agent_name
        state = AgentState(
            timestamp=time.time(),
            episode=self.current_episode,
            agent_name=agent_name,
            epsilon=epsilon,
            steps=steps,
            learning_rate=lr,
            loss=loss,
        )
        self._append(self.agent_states, state)
        self._enqueue(self.agent_queue, asdict(state))

    def record_episode(
        self,
        total_reward:     float,
        total_cost:       float,
        total_debt:       float,
        mean_modulus:     float,
        episode_length:   int,
        success_threshold: float,
    ) -> None:
        self._ep_rewards.append(total_reward)
        self._ep_costs.append(total_cost)
        self._ep_debts.append(total_debt)
        self._ep_lengths.append(episode_length)

        recent_r = self._ep_rewards[-10:]
        recent_c = self._ep_costs[-10:]
        recent_d = self._ep_debts[-10:]

        import numpy as np
        success_rate = float(np.mean([r >= success_threshold for r in recent_r]))

        metrics = EpisodeMetrics(
            timestamp=time.time(),
            episode=self.current_episode,
            env_name=self._env_name,
            agent_name=self._agent_name,
            total_reward=float(total_reward),
            total_cost=float(total_cost),
            total_debt=float(total_debt),
            mean_modulus=float(mean_modulus),
            episode_length=episode_length,
            avg_reward_last_10=float(np.mean(recent_r)),
            avg_cost_last_10=float(np.mean(recent_c)),
            avg_debt_last_10=float(np.mean(recent_d)),
            success_rate=success_rate,
            success_threshold=success_threshold,
        )

        self._append(self.episode_metrics, metrics)
        self._enqueue(self.metrics_queue, asdict(metrics))
        self.current_episode += 1

    # ── Readers ───────────────────────────────────────────────────────────────

    def get_latest(self, data_type: str, count: int = 100) -> List[Dict]:
        with self.lock:
            mapping = {
                "env":     self.env_states,
                "agent":   self.agent_states,
                "episode": self.episode_metrics,
            }
            data = mapping.get(data_type, [])
            return [asdict(item) for item in data[-count:]]

    def get_summary(self) -> Dict[str, Any]:
        with self.lock:
            if not self.episode_metrics:
                return {}
            m = self.episode_metrics[-1]
            return {
                "episode":      m.episode,
                "env_name":     m.env_name,
                "agent_name":   m.agent_name,
                "avg_reward":   m.avg_reward_last_10,
                "avg_cost":     m.avg_cost_last_10,
                "avg_debt":     m.avg_debt_last_10,
                "success_rate": m.success_rate,
            }

    # ── Internals ─────────────────────────────────────────────────────────────

    def _append(self, lst: list, item) -> None:
        with self.lock:
            lst.append(item)
            if len(lst) > self.max_history:
                lst.pop(0)

    @staticmethod
    def _enqueue(q: queue.Queue, item: dict) -> None:
        try:
            q.put_nowait(item)
        except queue.Full:
            pass


# Global singleton
_collector = DataCollector()

def get_data_collector() -> DataCollector:
    return _collector
