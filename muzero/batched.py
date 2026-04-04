"""Batched MCTS: run multiple games simultaneously, batch neural net evaluations.

Optimized to minimize GPU↔CPU transfers: all tensors are bulk-transferred once
per simulation round instead of per-game.
"""

import numpy as np
import torch

from .mcts import (
    Node, MinMaxStats, select_child,
    backpropagate, add_exploration_noise,
)
from .replay_buffer import GameHistory


def _expand_from_arrays(node, actions, priors, hidden_state, reward):
    """Expand node from pre-computed numpy arrays (no GPU ops)."""
    node.hidden_state = hidden_state
    node.reward = reward
    n = len(actions)
    node.child_actions = actions.astype(np.int32)
    node.child_priors = priors.astype(np.float64)
    node.child_visits = np.zeros(n, dtype=np.int32)
    node.child_value_sums = np.zeros(n, dtype=np.float64)
    node.child_rewards = np.zeros(n, dtype=np.float64)
    node.child_nodes = [None] * n


@torch.no_grad()
def batched_self_play(config, model, device, num_games, game_cls):
    """Play num_games simultaneously using batched MCTS.

    Optimized for MPS/CUDA: bulk GPU→CPU transfers instead of per-game.
    """
    games = [game_cls() for _ in range(num_games)]
    histories = [GameHistory() for _ in range(num_games)]
    steps = [0] * num_games
    active = list(range(num_games))

    max_children = getattr(config, "max_children", None) or config.action_space_size
    use_topk = max_children < config.action_space_size

    while active:
        N = len(active)

        # --- Collect state for active games ---
        observations = [games[i].get_observation() for i in active]
        legal_actions_list = [games[i].legal_actions() for i in active]

        # --- Batched initial inference ---
        obs_batch = torch.stack(
            [torch.as_tensor(o, dtype=torch.float32) for o in observations]
        ).to(device)
        hiddens, policies, values = model.initial_inference(obs_batch)

        # Bulk transfer for root expansion
        all_policies_cpu = torch.softmax(policies, dim=-1).cpu().numpy()

        roots = []
        mm_stats = []
        for j in range(N):
            root = Node(prior=0)
            legal = legal_actions_list[j]
            priors = all_policies_cpu[j][legal]
            priors /= priors.sum() + 1e-8  # renormalize over legal moves
            _expand_from_arrays(
                root, np.array(legal, dtype=np.int32), priors, hiddens[j], 0.0
            )
            add_exploration_noise(
                root, config.dirichlet_alpha, config.root_exploration_fraction
            )
            roots.append(root)
            mm_stats.append(MinMaxStats())

        # --- Simulations (batched across games) ---
        for _ in range(config.num_simulations):
            search_paths = []
            leaf_actions = []
            parent_hiddens = []
            child_indices_list = []

            # Selection (pure Python/numpy, no GPU)
            for j in range(N):
                node = roots[j]
                path = [node]
                c_indices = []
                while node.expanded():
                    act, node, ci = select_child(node, mm_stats[j], config)
                    c_indices.append(ci)
                    path.append(node)
                search_paths.append(path)
                leaf_actions.append(act)
                parent_hiddens.append(path[-2].hidden_state)
                child_indices_list.append(c_indices)

            # ONE batched GPU forward pass
            h_batch = torch.stack(parent_hiddens)
            a_batch = torch.LongTensor(leaf_actions).to(device)
            hiddens, rewards, policies, values = model.recurrent_inference(
                h_batch, a_batch
            )

            # ONE bulk GPU→CPU transfer (instead of N individual ones)
            if use_topk:
                all_probs = torch.softmax(policies, dim=-1)
                topk_vals, topk_idx = all_probs.topk(max_children, dim=-1)
                topk_vals_np = topk_vals.cpu().numpy()
                topk_idx_np = topk_idx.cpu().numpy()
            rewards_np = rewards.squeeze(-1).cpu().numpy()
            values_np = values.squeeze(-1).cpu().numpy()

            # Expand and backpropagate (pure Python/numpy, no GPU)
            for j in range(N):
                leaf = search_paths[j][-1]
                if use_topk:
                    _expand_from_arrays(
                        leaf, topk_idx_np[j], topk_vals_np[j],
                        hiddens[j], float(rewards_np[j])
                    )
                else:
                    probs = torch.softmax(policies[j], dim=0).cpu().numpy()
                    _expand_from_arrays(
                        leaf, np.arange(config.action_space_size, dtype=np.int32),
                        probs, hiddens[j], float(rewards_np[j])
                    )
                backpropagate(
                    search_paths[j], float(values_np[j]),
                    config.discount, mm_stats[j], child_indices_list[j]
                )

        # --- Extract policies and step games ---
        newly_done = []
        for idx, i in enumerate(active):
            root = roots[idx]

            visits = np.zeros(config.action_space_size, dtype=np.float32)
            visits[root.child_actions] = root.child_visits.astype(np.float32)

            temp = (
                config.temperature
                if steps[i] < config.temp_threshold
                else config.temp_final
            )
            if temp < 1e-6:
                action = int(np.argmax(visits))
                policy = np.zeros(config.action_space_size, dtype=np.float32)
                policy[action] = 1.0
            else:
                v = visits ** (1.0 / temp)
                policy = v / v.sum()
                action = int(np.random.choice(config.action_space_size, p=policy))

            histories[i].observations.append(observations[idx])
            histories[i].policies.append(policy)
            histories[i].root_values.append(root.value())
            histories[i].players.append(games[i].current_player)

            done, reward = games[i].step(action)
            histories[i].actions.append(action)
            histories[i].rewards.append(reward)
            steps[i] += 1

            if done:
                histories[i].outcome = (
                    histories[i].players[-1] if reward != 0 else 0
                )
                newly_done.append(i)

        active = [i for i in active if i not in newly_done]

    return histories
