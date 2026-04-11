# Copyright (C) 2026 
# Mohamed Elwardi
# This program is free software: you can redistribute it and/or 
# modify it under the terms of GNU General Public License
# by the Free Software Foundation, either version 3 of
# the License. This program is distributed in the hope
# that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY 
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.


"""
Complex Q-Network agent (HST framework).

Objective : minimise  |E[Σ γᵏ zₖ]|   where  zₖ = cₖ + i·dₖ
Update    :
    a* = argmin_{a'} |Q̄(s', a')|
    target_re = cost + γ · Re(Q̄(s', a*))
    target_im = debt + γ · Im(Q̄(s', a*))
Loss      : MSE(Re Q, target_re) + MSE(Im Q, target_im)
"""

import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .models import ComplexQNet
from .base_agent import BaseAgent
from .replay_buffer import ReplayBuffer, Transition


class ComplexDQNAgent(BaseAgent):

    def __init__(self, cfg, device: torch.device):
        super().__init__(cfg, device)
        self.name = "Complex DQN"

        self.q_net  = ComplexQNet(cfg.N_STATES, cfg.N_ACTIONS, cfg.HIDDEN_DIM, cfg.HIDDEN2_DIM).to(device)
        self.q_targ = copy.deepcopy(self.q_net).to(device)
        self.q_targ.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.LR)
        self.buffer    = ReplayBuffer(cfg.REPLAY_CAPACITY)

    @torch.no_grad()
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(self.cfg.N_ACTIONS)
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self.q_net.greedy_action(s).item())

    def update(self) -> float | None:
        if not self.buffer.ready(self.cfg.BATCH_SIZE):
            return None

        batch = self.buffer.sample(self.cfg.BATCH_SIZE)
        b     = Transition(*zip(*batch))

        states      = torch.tensor(np.array(b.state),      dtype=torch.float32, device=self.device)
        actions     = torch.tensor(b.action,               dtype=torch.int64,   device=self.device).unsqueeze(1)
        costs       = torch.tensor(b.cost,                 dtype=torch.float32, device=self.device)
        debts       = torch.tensor(b.debt,                 dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(b.next_state), dtype=torch.float32, device=self.device)
        dones       = torch.tensor(b.done,                 dtype=torch.float32, device=self.device)

        Q_re_all, Q_im_all = self.q_net(states)
        Q_re = Q_re_all.gather(1, actions).squeeze(1)
        Q_im = Q_im_all.gather(1, actions).squeeze(1)

        with torch.no_grad():
            a_star        = self.q_targ.greedy_action(next_states).unsqueeze(1)
            T_re, T_im    = self.q_targ(next_states)
            Q_next_re     = T_re.gather(1, a_star).squeeze(1)
            Q_next_im     = T_im.gather(1, a_star).squeeze(1)
            not_done      = 1.0 - dones
            target_re     = costs + self.cfg.GAMMA * Q_next_re * not_done
            target_im     = debts + self.cfg.GAMMA * Q_next_im * not_done

        loss = (
            nn.functional.mse_loss(Q_re, target_re)
            + nn.functional.mse_loss(Q_im, target_im)
        )

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        self.steps += 1

        return loss.item()
