"""
Neural network models for Q-learning agents, including standard and complex-valued Q-networks.
"""
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
#                Standard Q-Network   Q(s,a) ∈ ℝ
# ─────────────────────────────────────────────────────────────────────────────
 
class StandardQNet(nn.Module):

    """Standard Q-Network for Q-learning with real-valued Q-function outputs."""

    def __init__(self,
                 n_states:  int,
                 n_actions: int,
                 hidden:    int,
                 hidden2:   int = None):
        super().__init__()
        hidden2 = hidden2 or hidden
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, n_actions),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:   x  : (batch, 3)
        Returns: Q : (batch, N_ACTIONS)
        """
        return self.net(x)
 
 
# ─────────────────────────────────────────────────────────────────────────────
#                   Complex Q-Network   Q(s,a) ∈ ℂ
# ─────────────────────────────────────────────────────────────────────────────
 
class ComplexQNet(nn.Module):

    def __init__(self,
                 n_states:  int,
                 n_actions: int,
                 hidden:    int,
                 hidden2:   int = None):
        super().__init__()
        hidden2 = hidden2 or hidden
 
        # Shared feature extractor
        self.encoder = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden2),
            nn.ReLU(),
        )
 
        # Re-head: Re(Q(s,·)) ∈ ℝ^{N_ACTIONS}  — cost axis
        self.re_head = nn.Linear(hidden2, n_actions)
 
        # Im-head: Im(Q(s,·)) ∈ ℝ^{N_ACTIONS}  — debt axis
        self.im_head = nn.Linear(hidden2, n_actions)
 
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:   x    : (batch, 3)
        Returns Q_re : (batch, N_ACTIONS)   real part
                Q_im : (batch, N_ACTIONS)   imaginary part
        """
        features = self.encoder(x)
        return self.re_head(features), self.im_head(features)
 
    def modulus(self, x: torch.Tensor) -> torch.Tensor:
        """
        |Q(s,·)| = √(Re(Q)² + Im(Q)²)
 
        Args:   x   : (batch, 3)
        Returns mod : (batch, N_ACTIONS)
        """
        Q_re, Q_im = self.forward(x)
        return torch.sqrt(Q_re ** 2 + Q_im ** 2 + 1e-8)
 
    def greedy_action(self, x: torch.Tensor) -> torch.Tensor:
        """
        a* = argmin_a |Q(s,a)|  —  action closest to complex origin
 
        Args:   x       : (batch, 3)
        Returns actions : (batch,)  int64
        """
        return self.modulus(x).argmin(dim=-1)