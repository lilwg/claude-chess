import random
import numpy as np
import torch
import torch.nn.functional as F

from .game import TicTacToe, minimax_action
from .mcts import run_mcts
from .replay_buffer import GameHistory


# ---------------------------------------------------------------------------
# Self-play
# ---------------------------------------------------------------------------

def self_play_game(config, model, device, game_cls=None):
    """Play one complete game using MCTS, returning a GameHistory."""
    if game_cls is None:
        game_cls = TicTacToe
    game = game_cls()
    history = GameHistory()
    step = 0

    while True:
        obs = game.get_observation()
        legal = game.legal_actions()
        player = game.current_player

        root = run_mcts(config, model, obs, legal, device, add_noise=True)

        # Visit-count policy
        visits = np.zeros(config.action_space_size, dtype=np.float32)
        visits[root.child_actions] = root.child_visits.astype(np.float32)

        # Apply temperature
        temp = config.temperature if step < config.temp_threshold else config.temp_final
        if temp < 1e-6:
            action = int(np.argmax(visits))
            policy = np.zeros(config.action_space_size, dtype=np.float32)
            policy[action] = 1.0
        else:
            v = visits ** (1.0 / temp)
            policy = v / v.sum()
            action = int(np.random.choice(config.action_space_size, p=policy))

        # Record state BEFORE taking the action
        history.observations.append(obs)
        history.policies.append(policy)
        history.root_values.append(root.value())
        history.players.append(player)

        # Step the game
        done, reward = game.step(action)
        history.actions.append(action)
        history.rewards.append(reward)

        step += 1
        if done:
            break

    # Outcome from player +1's perspective
    if reward != 0:
        history.outcome = history.players[-1]  # winner's id
    else:
        history.outcome = 0
    return history


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def update_weights(model, optimizer, replay_buffer, config, device):
    """Run one training step. Returns dict of scalar losses."""
    batch = replay_buffer.sample_batch(config.batch_size)

    obs = torch.FloatTensor(batch["observation"]).to(device)
    actions = torch.LongTensor(batch["actions"]).to(device)            # (B, K)
    t_policies = torch.FloatTensor(batch["target_policies"]).to(device) # (B, K+1, A)
    t_values = torch.FloatTensor(batch["target_values"]).to(device)     # (B, K+1)
    t_rewards = torch.FloatTensor(batch["target_rewards"]).to(device)   # (B, K)

    K = config.unroll_steps
    A = config.action_space_size

    # Initial inference
    hidden, policy_logits, value = model.initial_inference(obs)
    # Losses at step 0
    p_loss = soft_cross_entropy(policy_logits, t_policies[:, 0])
    v_loss = F.mse_loss(value.squeeze(-1), t_values[:, 0])
    r_loss = torch.zeros(1, device=device)

    # Unroll K steps
    for k in range(K):
        # Scale gradient entering dynamics (MuZero trick)
        hidden = scale_gradient(hidden, 0.5)

        hidden, reward, policy_logits, value = model.recurrent_inference(
            hidden, actions[:, k]
        )
        p_loss += soft_cross_entropy(policy_logits, t_policies[:, k + 1])
        v_loss += F.mse_loss(value.squeeze(-1), t_values[:, k + 1])
        r_loss += F.mse_loss(reward.squeeze(-1), t_rewards[:, k])

    # Average over unroll steps
    scale = 1.0 / (K + 1)
    total = scale * (p_loss + v_loss + r_loss)

    optimizer.zero_grad()
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return {
        "total": total.item(),
        "policy": (scale * p_loss).item(),
        "value": (scale * v_loss).item(),
        "reward": (scale * r_loss).item(),
    }


def soft_cross_entropy(logits, targets):
    """Cross-entropy with soft (probability) targets, averaged over batch."""
    log_probs = F.log_softmax(logits, dim=-1)
    return -(targets * log_probs).sum(dim=-1).mean()


def scale_gradient(tensor, scale):
    """Scale gradient on backward pass while keeping forward pass unchanged."""
    return tensor * scale + tensor.detach() * (1 - scale)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(config, model, device, num_episodes=None):
    """Play against random and minimax opponents. Returns results dict."""
    if num_episodes is None:
        num_episodes = config.eval_episodes

    results = {}
    for opp in ("random", "minimax"):
        w, l, d = 0, 0, 0
        for ep in range(num_episodes):
            muzero_player = 1 if ep % 2 == 0 else -1
            game = TicTacToe()

            while True:
                if game.current_player == muzero_player:
                    obs = game.get_observation()
                    legal = game.legal_actions()
                    root = run_mcts(config, model, obs, legal, device, add_noise=False)
                    action = int(root.child_actions[np.argmax(root.child_visits)])
                else:
                    if opp == "random":
                        action = random.choice(game.legal_actions())
                    else:
                        action = minimax_action(game)

                done, reward = game.step(action)
                if done:
                    if reward == 0:
                        d += 1
                    elif game.current_player == muzero_player:
                        w += 1
                    else:
                        l += 1
                    break
        results[opp] = {"win": w, "loss": l, "draw": d}
    return results
