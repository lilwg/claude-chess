import random
import numpy as np


class GameHistory:
    """Stores one complete game of self-play data."""

    def __init__(self):
        self.observations = []   # observation from current player's view
        self.actions = []        # action taken
        self.rewards = []        # reward from acting player's perspective
        self.policies = []       # MCTS visit-count distribution
        self.root_values = []    # MCTS root value
        self.players = []        # +1 or -1 (who is acting)
        self.outcome = 0         # game result from player +1's perspective

    @property
    def length(self):
        return len(self.actions)

    def value_target(self, pos):
        """Value target from the perspective of the player at `pos`."""
        return self.outcome * self.players[pos]


class ReplayBuffer:
    def __init__(self, config):
        self.config = config
        self.buffer = []

    def save_game(self, history):
        self.buffer.append(history)
        while len(self.buffer) > self.config.buffer_size:
            self.buffer.pop(0)

    def total_games(self):
        return len(self.buffer)

    def sample_batch(self, batch_size):
        """Sample a batch of (observation, actions, targets) for training."""
        K = self.config.unroll_steps
        A = self.config.action_space_size
        batch = []

        for _ in range(batch_size):
            game = random.choice(self.buffer)
            pos = random.randint(0, game.length - 1)

            observation = np.array(game.observations[pos], dtype=np.float32)

            actions = []
            target_rewards = []
            target_policies = [np.array(game.policies[pos], dtype=np.float32)]
            target_values = [game.value_target(pos)]

            for k in range(K):
                idx = pos + k          # action index
                nxt = pos + k + 1      # next state index

                if idx < game.length:
                    actions.append(game.actions[idx])
                    target_rewards.append(game.rewards[idx])
                else:
                    actions.append(random.randint(0, A - 1))
                    target_rewards.append(0.0)

                if nxt < game.length:
                    target_policies.append(
                        np.array(game.policies[nxt], dtype=np.float32)
                    )
                    target_values.append(game.value_target(nxt))
                else:
                    target_policies.append(np.ones(A, dtype=np.float32) / A)
                    target_values.append(0.0)

            batch.append(
                {
                    "observation": observation,
                    "actions": np.array(actions, dtype=np.int64),
                    "target_rewards": np.array(target_rewards, dtype=np.float32),
                    "target_policies": np.stack(target_policies),  # (K+1, A)
                    "target_values": np.array(target_values, dtype=np.float32),  # (K+1,)
                }
            )

        # Collate into tensors
        return {
            key: np.stack([b[key] for b in batch])
            for key in batch[0]
        }
