"""ResNet-based MuZero network for chess."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .chess_game import actions_to_planes


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class ChessMuZero(nn.Module):
    """MuZero network for chess with ResNet backbone.

    Representation:  (B, 19, 8, 8) observation  -> (B, C, 8, 8) hidden
    Dynamics:        (B, C, 8, 8) + action       -> (B, C, 8, 8) hidden, (B, 1) reward
    Prediction:      (B, C, 8, 8) hidden         -> (B, 4672) policy, (B, 1) value
    """

    def __init__(self, obs_channels=19, hidden_channels=128, num_blocks=6,
                 action_size=4672):
        super().__init__()
        self.action_size = action_size
        C = hidden_channels

        # --- Representation ---
        self.repr_conv = nn.Conv2d(obs_channels, C, 3, padding=1, bias=False)
        self.repr_bn = nn.BatchNorm2d(C)
        self.repr_blocks = nn.Sequential(*[ResBlock(C) for _ in range(num_blocks)])

        # --- Dynamics (hidden + 2 action planes) ---
        self.dyn_conv = nn.Conv2d(C + 2, C, 3, padding=1, bias=False)
        self.dyn_bn = nn.BatchNorm2d(C)
        self.dyn_blocks = nn.Sequential(*[ResBlock(C) for _ in range(num_blocks)])

        # Reward head
        self.reward_conv = nn.Conv2d(C, 1, 1)
        self.reward_bn = nn.BatchNorm2d(1)
        self.reward_fc1 = nn.Linear(64, 64)
        self.reward_fc2 = nn.Linear(64, 1)

        # --- Prediction ---
        # Spatial policy head: Conv2d outputs 73 planes (move types per square)
        # Reshape (B,73,8,8) → (B,4672) matching action = from_sq*73 + move_type
        self.policy_conv = nn.Conv2d(C, 73, 1)

        # Value head
        self.value_conv = nn.Conv2d(C, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(64, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def represent(self, obs):
        x = F.relu(self.repr_bn(self.repr_conv(obs)))
        return self.repr_blocks(x)

    def dynamics(self, hidden, action_planes):
        x = torch.cat([hidden, action_planes], dim=1)
        x = F.relu(self.dyn_bn(self.dyn_conv(x)))
        return self.dyn_blocks(x)

    def predict_reward(self, hidden):
        x = F.relu(self.reward_bn(self.reward_conv(hidden)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.reward_fc1(x))
        return torch.tanh(self.reward_fc2(x))

    def predict(self, hidden):
        # Spatial policy: (B,73,8,8) → permute to (B,8,8,73) → flatten to (B,4672)
        p = self.policy_conv(hidden)
        p = p.permute(0, 2, 3, 1).reshape(p.size(0), -1)
        # Value
        v = F.relu(self.value_bn(self.value_conv(hidden)))
        v = v.view(v.size(0), -1)
        v = torch.tanh(self.value_fc2(F.relu(self.value_fc1(v))))
        return p, v

    def initial_inference(self, obs):
        hidden = self.represent(obs)
        policy, value = self.predict(hidden)
        return hidden, policy, value

    def recurrent_inference(self, hidden, action):
        """action: (B,) LongTensor of action indices."""
        ap = actions_to_planes(action, hidden.device)
        next_hidden = self.dynamics(hidden, ap)
        reward = self.predict_reward(next_hidden)
        policy, value = self.predict(next_hidden)
        return next_hidden, reward, policy, value
