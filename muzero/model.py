import torch
import torch.nn as nn
import torch.nn.functional as F


class MuZeroNetwork(nn.Module):
    """
    MuZero's three learned functions:
      - Representation:  observation -> hidden state
      - Dynamics:        (hidden, action) -> (next_hidden, reward)
      - Prediction:      hidden -> (policy_logits, value)
    """

    def __init__(self, obs_size, action_size, hidden_size):
        super().__init__()
        self.action_size = action_size

        # Representation: obs -> hidden
        self.repr_net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.repr_norm = nn.LayerNorm(hidden_size)

        # Dynamics: (hidden, action_onehot) -> next_hidden
        self.dyn_net = nn.Sequential(
            nn.Linear(hidden_size + action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.dyn_norm = nn.LayerNorm(hidden_size)

        # Dynamics reward head
        self.dyn_reward = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

        # Prediction: hidden -> policy logits
        self.pred_policy = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

        # Prediction: hidden -> value
        self.pred_value = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

    def represent(self, obs):
        """obs: (B, obs_size) -> hidden: (B, hidden_size)"""
        return self.repr_norm(self.repr_net(obs))

    def dynamics(self, hidden, action_onehot):
        """
        hidden: (B, hidden_size), action_onehot: (B, action_size)
        -> next_hidden: (B, hidden_size), reward: (B, 1)
        """
        x = torch.cat([hidden, action_onehot], dim=-1)
        next_hidden = self.dyn_norm(self.dyn_net(x))
        reward = self.dyn_reward(next_hidden)
        return next_hidden, reward

    def predict(self, hidden):
        """hidden: (B, hidden_size) -> policy_logits: (B, A), value: (B, 1)"""
        return self.pred_policy(hidden), self.pred_value(hidden)

    def initial_inference(self, obs):
        """Full initial step: obs -> (hidden, policy_logits, value)"""
        hidden = self.represent(obs)
        policy_logits, value = self.predict(hidden)
        return hidden, policy_logits, value

    def recurrent_inference(self, hidden, action):
        """Full recurrent step. action: (B,) LongTensor of action indices."""
        action_onehot = F.one_hot(action, self.action_size).float()
        next_hidden, reward = self.dynamics(hidden, action_onehot)
        policy_logits, value = self.predict(next_hidden)
        return next_hidden, reward, policy_logits, value
