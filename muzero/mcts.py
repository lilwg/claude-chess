"""MCTS with numpy-vectorized UCB selection."""

import math
import numpy as np
import torch


class MinMaxStats:
    """Tracks min/max Q-values for normalization in UCB."""

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

    def normalize_batch(self, values):
        if self.maximum > self.minimum:
            return (values - self.minimum) / (self.maximum - self.minimum)
        return values


class Node:
    __slots__ = (
        "prior", "visit_count", "value_sum", "hidden_state", "reward",
        "child_actions", "child_priors", "child_visits", "child_value_sums",
        "child_rewards", "child_nodes",
    )

    def __init__(self, prior):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.hidden_state = None
        self.reward = 0.0
        # Array-backed children (set on expansion)
        self.child_actions = None
        self.child_priors = None
        self.child_visits = None
        self.child_value_sums = None
        self.child_rewards = None
        self.child_nodes = None

    def expanded(self):
        return self.child_actions is not None

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def select_child(node, min_max_stats, config):
    """Vectorized UCB child selection."""
    n = len(node.child_actions)
    visits = node.child_visits
    priors = node.child_priors

    # Exploration term
    pb_c = (
        math.log((node.visit_count + config.pb_c_base + 1) / config.pb_c_base)
        + config.pb_c_init
    )
    prior_scores = pb_c * priors * math.sqrt(node.visit_count) / (1.0 + visits)

    # Exploitation: Q from parent's perspective
    child_values = np.divide(
        node.child_value_sums, visits,
        out=np.zeros(n, dtype=np.float64), where=visits > 0
    )
    q_values = np.where(
        visits > 0,
        node.child_rewards - config.discount * child_values,
        0.0,
    )
    value_scores = min_max_stats.normalize_batch(q_values)

    best = int(np.argmax(value_scores + prior_scores))
    action = int(node.child_actions[best])

    # Lazily create child node
    if node.child_nodes[best] is None:
        node.child_nodes[best] = Node(prior=float(priors[best]))
    return action, node.child_nodes[best], best


def expand_node(node, actions, policy_logits, hidden_state, reward):
    """Expand with given actions, storing children as numpy arrays."""
    node.hidden_state = hidden_state
    node.reward = reward
    policy = torch.softmax(policy_logits, dim=0).detach().cpu().numpy()
    n = len(actions)
    node.child_actions = np.array(actions, dtype=np.int32)
    node.child_priors = np.array([policy[a] for a in actions], dtype=np.float64)
    node.child_visits = np.zeros(n, dtype=np.int32)
    node.child_value_sums = np.zeros(n, dtype=np.float64)
    node.child_rewards = np.zeros(n, dtype=np.float64)
    node.child_nodes = [None] * n


def expand_node_topk(node, policy_logits, hidden_state, reward, k):
    """Expand with top-k actions by prior."""
    node.hidden_state = hidden_state
    node.reward = reward
    policy = torch.softmax(policy_logits, dim=0)
    topk_vals, topk_idx = policy.topk(min(k, len(policy)))
    topk_vals = topk_vals.detach().cpu().numpy()
    topk_idx = topk_idx.detach().cpu().numpy()
    n = len(topk_idx)
    node.child_actions = topk_idx.astype(np.int32)
    node.child_priors = topk_vals.astype(np.float64)
    node.child_visits = np.zeros(n, dtype=np.int32)
    node.child_value_sums = np.zeros(n, dtype=np.float64)
    node.child_rewards = np.zeros(n, dtype=np.float64)
    node.child_nodes = [None] * n


def backpropagate(search_path, value, discount, min_max_stats,
                  child_indices=None):
    """Propagate leaf value up the tree, alternating perspectives.

    child_indices: list of int indices into parent.child_* arrays,
                   one per edge in the path (len = len(search_path) - 1).
    """
    for i, node in enumerate(reversed(search_path)):
        node.value_sum += value
        node.visit_count += 1

        # Update parent's child arrays
        if child_indices is not None and i < len(child_indices):
            ci = child_indices[-(i + 1)]
            parent = search_path[-(i + 2)] if i + 2 <= len(search_path) else None
            if parent is not None:
                parent.child_visits[ci] = node.visit_count
                parent.child_value_sums[ci] = node.value_sum
                parent.child_rewards[ci] = node.reward

        q = node.reward - discount * node.value()
        min_max_stats.update(q)
        value = node.reward - discount * value


def add_exploration_noise(node, alpha, frac):
    n = len(node.child_priors)
    noise = np.random.dirichlet([alpha] * n)
    node.child_priors = (1 - frac) * node.child_priors + frac * noise


@torch.no_grad()
def run_mcts(config, model, observation, legal_actions, device, add_noise=True):
    """Run MCTS. Returns root node with visit counts reflecting search policy."""
    root = Node(prior=0)

    obs_t = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)
    hidden, policy_logits, value = model.initial_inference(obs_t)

    pl = policy_logits.squeeze(0)
    mask = torch.full_like(pl, float("-inf"))
    for a in legal_actions:
        mask[a] = 0.0
    pl = pl + mask

    expand_node(root, legal_actions, pl, hidden.squeeze(0), 0.0)

    if add_noise:
        add_exploration_noise(root, config.dirichlet_alpha, config.root_exploration_fraction)

    min_max_stats = MinMaxStats()
    max_children = getattr(config, "max_children", None)
    use_topk = max_children and config.action_space_size > max_children

    for _ in range(config.num_simulations):
        node = root
        search_path = [node]
        child_indices = []

        while node.expanded():
            action, child, ci = select_child(node, min_max_stats, config)
            child_indices.append(ci)
            node = child
            search_path.append(node)

        parent = search_path[-2]

        action_t = torch.LongTensor([action]).to(device)
        hidden, reward, policy_logits, value = model.recurrent_inference(
            parent.hidden_state.unsqueeze(0), action_t
        )

        if use_topk:
            expand_node_topk(
                node, policy_logits.squeeze(0), hidden.squeeze(0),
                reward.squeeze().item(), max_children,
            )
        else:
            expand_node(
                node, list(range(config.action_space_size)),
                policy_logits.squeeze(0), hidden.squeeze(0),
                reward.squeeze().item(),
            )

        backpropagate(
            search_path, value.squeeze().item(), config.discount,
            min_max_stats, child_indices,
        )

    return root
