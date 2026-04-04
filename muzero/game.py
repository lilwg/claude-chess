LINES = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
    [0, 3, 6], [1, 4, 7], [2, 5, 8],  # cols
    [0, 4, 8], [2, 4, 6],              # diags
]


class TicTacToe:
    """Tic-tac-toe environment. Players are +1 and -1."""

    def __init__(self):
        self.board = [0] * 9
        self.current_player = 1

    def get_observation(self):
        """Board from current player's perspective: +1=mine, -1=opponent, 0=empty."""
        return [self.board[i] * self.current_player for i in range(9)]

    def legal_actions(self):
        return [i for i in range(9) if self.board[i] == 0]

    def step(self, action):
        """Returns (done, reward). Reward is from the acting player's perspective."""
        assert self.board[action] == 0, f"Illegal action {action}"
        self.board[action] = self.current_player

        if self._check_winner():
            return True, 1.0  # acting player wins

        if not self.legal_actions():
            return True, 0.0  # draw

        self.current_player *= -1
        return False, 0.0

    def _check_winner(self):
        for a, b, c in LINES:
            if self.board[a] == self.board[b] == self.board[c] != 0:
                return True
        return False

    def render(self):
        sym = {0: ".", 1: "X", -1: "O"}
        rows = []
        for r in range(3):
            rows.append(" ".join(sym[self.board[r * 3 + c]] for c in range(3)))
        return "\n".join(rows)


# --- Minimax for evaluation ---

def minimax_value(board, player, alpha=-2, beta=2):
    """Negamax with alpha-beta. Returns value from `player`'s perspective."""
    for a, b, c in LINES:
        if board[a] == board[b] == board[c] != 0:
            return 1 if board[a] == player else -1

    legal = [i for i in range(9) if board[i] == 0]
    if not legal:
        return 0

    best = -2
    for action in legal:
        board[action] = player
        val = -minimax_value(board, -player, -beta, -alpha)
        board[action] = 0
        best = max(best, val)
        alpha = max(alpha, best)
        if alpha >= beta:
            break
    return best


def minimax_action(game):
    """Returns the optimal action for the current player."""
    board = game.board[:]
    player = game.current_player
    best_action = game.legal_actions()[0]
    best_val = -2
    for action in game.legal_actions():
        board[action] = player
        val = -minimax_value(board, -player)
        board[action] = 0
        if val > best_val:
            best_val = val
            best_action = action
    return best_action
