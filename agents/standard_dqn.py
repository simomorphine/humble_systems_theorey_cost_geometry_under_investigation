"""
Standard Deep Q-Network agent.

Objective : maximise  E[Σ γᵏ rₖ]
Update    : Q(s,a) ← r + γ · max_{a'} Q̄(s',a')
Loss      : MSE( Q(s,a), r + γ · max_{a'} Q̄(s',a') )
"""

import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .models import StandardQNet
from .base_agent import BaseAgent
from .replay_buffer import ReplayBuffer, Transition


class StandardDQNAgent(BaseAgent):

    def __init__(self, cfg, device: torch.device):
        super().__init__(cfg, device)
        self.name = "Standard DQN"

        self.q_net  = StandardQNet(cfg.N_STATES, cfg.N_ACTIONS, cfg.HIDDEN_DIM, cfg.HIDDEN2_DIM).to(device)
        self.q_targ = copy.deepcopy(self.q_net).to(device)
        self.q_targ.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.LR)
        self.buffer    = ReplayBuffer(cfg.REPLAY_CAPACITY)

    @torch.no_grad()
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(self.cfg.N_ACTIONS)
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self.q_net(s).argmax(dim=-1).item())

    def update(self) -> float | None:
        if not self.buffer.ready(self.cfg.BATCH_SIZE):
            return None

        batch = self.buffer.sample(self.cfg.BATCH_SIZE)
        b     = Transition(*zip(*batch))

        states      = torch.tensor(np.array(b.state),      dtype=torch.float32, device=self.device)
        actions     = torch.tensor(b.action,               dtype=torch.int64,   device=self.device).unsqueeze(1)
        rewards     = torch.tensor(b.reward,               dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(b.next_state), dtype=torch.float32, device=self.device)
        dones       = torch.tensor(b.done,                 dtype=torch.float32, device=self.device)

        Q_sa = self.q_net(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            Q_next  = self.q_targ(next_states).max(dim=-1).values
            targets = rewards + self.cfg.GAMMA * Q_next * (1.0 - dones)

        loss = nn.functional.mse_loss(Q_sa, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        self.steps += 1

        return loss.item()
