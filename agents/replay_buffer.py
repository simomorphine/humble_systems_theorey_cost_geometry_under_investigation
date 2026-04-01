"""
Experience replay buffer — shared by all DQN-family agents.
"""

import random
from collections import namedtuple, deque
from typing import List

Transition = namedtuple(
    "Transition",
    ("state", "action", "reward", "cost", "debt", "next_state", "done"),
)


class ReplayBuffer:

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, cost, debt, next_state, done):
        self.buffer.append(Transition(state, action, reward, cost, debt, next_state, done))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def ready(self, batch_size: int) -> bool:
        return len(self.buffer) >= batch_size

    def __len__(self) -> int:
        return len(self.buffer)
