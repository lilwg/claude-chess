"""AlphaZero MCTS: uses real chess board for state transitions (not learned dynamics).

More accurate than MuZero MCTS because it uses real game rules.
Each node stores an actual chess.Board — no dynamics model needed.
"""

import math
import numpy as np
import chess
import torch

from .chess_game import encode_board, encode_move, decode_move


class AZNode:
    """MCTS node backed by a real chess position."""
    __slots__ = (
        "board", "visit_count", "value_sum",
        "child_actions", "child_priors", "child_visits",
        "child_value_sums", "child_nodes",
    )

    def __init__(self, board=None):
        self.board = board
        self.visit_count = 0
        self.value_sum = 0.0
        self.child_actions = None  # numpy int32 array of action indices
        self.child_priors = None   # numpy float64 array
        self.child_visits = None
        self.child_value_sums = None
        self.child_nodes = None    # list of AZNode or None

    def expanded(self):
        return self.child_actions is not None

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def _select_child(node, pb_c_base, pb_c_init):
    """Vectorized UCB selection. Returns (action, child_index)."""
    n = len(node.child_actions)
    visits = node.child_visits
    priors = node.child_priors

    pb_c = math.log((node.visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init
    exploration = pb_c * priors * math.sqrt(node.visit_count) / (1.0 + visits)

    # Q-values from current player's perspective
    # child value is from opponent's perspective → negate
    child_q = np.divide(
        node.child_value_sums, visits,
        out=np.zeros(n, dtype=np.float64), where=visits > 0
    )
    exploitation = np.where(visits > 0, -child_q, 0.0)

    best = int(np.argmax(exploitation + exploration))
    return int(node.child_actions[best]), best


def _expand(node, policy_np, legal_actions, max_children=32):
    """Expand node with legal actions, priors from policy."""
    priors = policy_np[legal_actions]
    priors = priors / (priors.sum() + 1e-8)

    # Top-K if too many
    if len(legal_actions) > max_children:
        top_idx = np.argsort(priors)[-max_children:]
        legal_actions = [legal_actions[i] for i in top_idx]
        priors = priors[top_idx]
        priors = priors / (priors.sum() + 1e-8)

    n = len(legal_actions)
    node.child_actions = np.array(legal_actions, dtype=np.int32)
    node.child_priors = priors.astype(np.float64)
    node.child_visits = np.zeros(n, dtype=np.int32)
    node.child_value_sums = np.zeros(n, dtype=np.float64)
    node.child_nodes = [None] * n


def _backpropagate(path, child_indices, value):
    """Propagate value up the path, alternating sign at each level."""
    for i in range(len(path) - 1, -1, -1):
        node = path[i]
        node.value_sum += value
        node.visit_count += 1
        if i > 0:
            parent = path[i - 1]
            ci = child_indices[i - 1]
            parent.child_visits[ci] = node.visit_count
            parent.child_value_sums[ci] = node.value_sum
        value = -value  # flip for parent (opponent's perspective)


@torch.no_grad()
def alphazero_mcts(model, board, device, num_sims=50, max_children=32,
                   pb_c_base=19652, pb_c_init=1.25,
                   dirichlet_alpha=0.3, noise_frac=0.25, add_noise=True):
    """Run AlphaZero MCTS on a chess position.

    Uses real chess.Board for state transitions — no dynamics model.
    Returns root AZNode with visit counts.
    """
    root = AZNode(board)

    # Evaluate root
    obs = torch.as_tensor(encode_board(board), dtype=torch.float32)
    obs = obs.unsqueeze(0).to(device)
    policy_logits, value, _ = model.evaluate(obs)

    policy_np = torch.softmax(policy_logits.squeeze(0), dim=0).cpu().numpy()
    legal = [encode_move(m, board.turn) for m in board.legal_moves]

    _expand(root, policy_np, legal, max_children)

    if add_noise:
        n = len(root.child_priors)
        noise = np.random.dirichlet([dirichlet_alpha] * n)
        root.child_priors = (1 - noise_frac) * root.child_priors + noise_frac * noise

    for _ in range(num_sims):
        node = root
        path = [node]
        child_indices = []

        # Selection
        while node.expanded():
            action, ci = _select_child(node, pb_c_base, pb_c_init)
            child_indices.append(ci)
            if node.child_nodes[ci] is None:
                # Create child with real board
                child_board = node.board.copy()
                move = decode_move(action, child_board)
                child_board.push(move)
                node.child_nodes[ci] = AZNode(child_board)
            node = node.child_nodes[ci]
            path.append(node)

        # Expansion + evaluation
        if node.board.is_game_over(claim_draw=True):
            result = node.board.result(claim_draw=True)
            if result == "1/2-1/2":
                value = 0.0
            elif (result == "1-0") == (node.board.turn == chess.BLACK):
                # The side that just moved won
                value = -1.0  # bad for current player (they're in checkmate)
            else:
                value = 1.0
        else:
            obs = torch.as_tensor(
                encode_board(node.board), dtype=torch.float32
            ).unsqueeze(0).to(device)
            policy_logits, value_t, _ = model.evaluate(obs)

            policy_np = torch.softmax(policy_logits.squeeze(0), dim=0).cpu().numpy()
            legal = [encode_move(m, node.board.turn) for m in node.board.legal_moves]
            _expand(node, policy_np, legal, max_children)
            value = value_t.item()

        # Backpropagation
        _backpropagate(path, child_indices, value)

    return root


@torch.no_grad()
def batched_alphazero_mcts(model, boards, device, num_sims=50, max_children=32,
                           pb_c_base=19652, pb_c_init=1.25,
                           dirichlet_alpha=0.3, noise_frac=0.25):
    """Batched AlphaZero MCTS across multiple boards.

    Batches leaf evaluations for GPU efficiency.
    """
    N = len(boards)
    roots = []

    # Batch initial evaluation
    obs_batch = torch.stack([
        torch.as_tensor(encode_board(b), dtype=torch.float32) for b in boards
    ]).to(device)
    policy_logits, values, _ = model.evaluate(obs_batch)
    all_policies = torch.softmax(policy_logits, dim=-1).cpu().numpy()

    for j in range(N):
        root = AZNode(boards[j])
        legal = [encode_move(m, boards[j].turn) for m in boards[j].legal_moves]
        _expand(root, all_policies[j], legal, max_children)
        # Add noise
        n = len(root.child_priors)
        noise = np.random.dirichlet([dirichlet_alpha] * n)
        root.child_priors = (1 - noise_frac) * root.child_priors + noise_frac * noise
        roots.append(root)

    for _ in range(num_sims):
        # Selection phase (all games)
        leaf_boards = []
        leaf_paths = []
        leaf_child_indices = []
        terminal_info = []  # (path_idx, value) for terminal nodes

        for j in range(N):
            node = roots[j]
            path = [node]
            c_indices = []

            while node.expanded():
                action, ci = _select_child(node, pb_c_base, pb_c_init)
                c_indices.append(ci)
                if node.child_nodes[ci] is None:
                    child_board = node.board.copy()
                    child_board.push(decode_move(action, child_board))
                    node.child_nodes[ci] = AZNode(child_board)
                node = node.child_nodes[ci]
                path.append(node)

            if node.board.is_game_over(claim_draw=True):
                result = node.board.result(claim_draw=True)
                if result == "1/2-1/2":
                    v = 0.0
                elif (result == "1-0") == (node.board.turn == chess.BLACK):
                    v = -1.0
                else:
                    v = 1.0
                terminal_info.append((j, v))
            else:
                leaf_boards.append(node.board)

            leaf_paths.append(path)
            leaf_child_indices.append(c_indices)

        # Batch evaluate non-terminal leaves
        if leaf_boards:
            non_terminal = [j for j in range(N) if j not in {t[0] for t in terminal_info}]
            obs_batch = torch.stack([
                torch.as_tensor(encode_board(b), dtype=torch.float32)
                for b in leaf_boards
            ]).to(device)
            policy_logits, values, _ = model.evaluate(obs_batch)
            all_policies = torch.softmax(policy_logits, dim=-1).cpu().numpy()
            all_values = values.cpu().numpy()

            bi = 0
            for j in non_terminal:
                node = leaf_paths[j][-1]
                legal = [encode_move(m, node.board.turn) for m in node.board.legal_moves]
                _expand(node, all_policies[bi], legal, max_children)
                _backpropagate(leaf_paths[j], leaf_child_indices[j], float(all_values[bi]))
                bi += 1

        # Backpropagate terminal nodes
        for j, v in terminal_info:
            _backpropagate(leaf_paths[j], leaf_child_indices[j], v)

    return roots
