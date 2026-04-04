"""AlphaZero-style network: representation + prediction only (no dynamics).

Changes from MuZero:
  - No dynamics model (uses real chess engine for MCTS state transitions)
  - WDL value head (win/draw/loss probabilities instead of scalar)
  - ~3.5M params vs 6.1M (freed from dynamics)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .chess_model import ResBlock


class AlphaZeroNet(nn.Module):
    """Board observation → policy + WDL value."""

    def __init__(self, obs_channels=19, hidden_channels=128, num_blocks=10,
                 action_size=4672):
        super().__init__()
        self.action_size = action_size
        C = hidden_channels

        # Backbone
        self.conv = nn.Conv2d(obs_channels, C, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(C)
        self.blocks = nn.Sequential(*[ResBlock(C) for _ in range(num_blocks)])

        # Spatial policy head
        self.policy_conv = nn.Conv2d(C, 73, 1)

        # WDL value head (3 outputs: win, draw, loss)
        self.value_conv = nn.Conv2d(C, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(64, 128)
        self.value_fc2 = nn.Linear(128, 3)  # WDL

    def forward(self, obs):
        """obs: (B, 19, 8, 8) → policy: (B, 4672), wdl: (B, 3)"""
        x = F.relu(self.bn(self.conv(obs)))
        x = self.blocks(x)

        # Policy
        p = self.policy_conv(x)
        p = p.permute(0, 2, 3, 1).reshape(p.size(0), -1)

        # WDL value
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = self.value_fc2(F.relu(self.value_fc1(v)))

        return p, v

    def evaluate(self, obs):
        """Convenience: returns policy logits and scalar value (for MCTS)."""
        p, wdl_logits = self(obs)
        wdl = F.softmax(wdl_logits, dim=-1)
        # Convert WDL to scalar: value = P(win) - P(loss)
        value = wdl[:, 0] - wdl[:, 2]
        return p, value, wdl
