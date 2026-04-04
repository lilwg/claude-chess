"""Chess environment and AlphaZero-style move encoding for MuZero."""

import chess
import numpy as np
import torch

# ---- Action space: 64 squares × 73 move types = 4672 ----

NUM_ACTIONS = 4672

# Queen-move directions: (delta_rank, delta_file)
QUEEN_DIRS = [
    (1, 0), (1, 1), (0, 1), (-1, 1),    # N, NE, E, SE
    (-1, 0), (-1, -1), (0, -1), (1, -1),  # S, SW, W, NW
]

# Knight-move deltas
KNIGHT_DELTAS = [
    (2, 1), (2, -1), (1, 2), (1, -2),
    (-1, 2), (-1, -2), (-2, 1), (-2, -1),
]

UNDERPROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

# Lookup tables for fast encoding
_DELTA_TO_QUEEN = {}
for _d, (_dr, _df) in enumerate(QUEEN_DIRS):
    for _dist in range(1, 8):
        _DELTA_TO_QUEEN[(_dr * _dist, _df * _dist)] = (_d, _dist)

_DELTA_TO_KNIGHT = {d: i for i, d in enumerate(KNIGHT_DELTAS)}

# Precomputed action → (from_r, from_f, to_r, to_f) for fast plane creation
_ACTION_COORDS = np.zeros((NUM_ACTIONS, 4), dtype=np.int32)
for _a in range(NUM_ACTIONS):
    _fsq = _a // 73
    _mt = _a % 73
    _fr = _fsq // 8
    _ff = _fsq % 8
    if _mt < 56:
        _d2 = _mt // 7
        _dist2 = _mt % 7 + 1
        _tr = _fr + QUEEN_DIRS[_d2][0] * _dist2
        _tf = _ff + QUEEN_DIRS[_d2][1] * _dist2
    elif _mt < 64:
        _ki = _mt - 56
        _tr = _fr + KNIGHT_DELTAS[_ki][0]
        _tf = _ff + KNIGHT_DELTAS[_ki][1]
    else:
        _pi = _mt - 64
        _di = _pi % 3
        _tr = _fr + 1
        _tf = _ff + (_di - 1)
    _ACTION_COORDS[_a] = [_fr, _ff, np.clip(_tr, 0, 7), np.clip(_tf, 0, 7)]


# ---- Move encoding / decoding ----

def _mirror(sq):
    """Flip square vertically (rank 0 ↔ 7)."""
    return sq ^ 56


def encode_move(move, turn):
    """Convert chess.Move to action index in [0, 4672)."""
    from_sq = move.from_square
    to_sq = move.to_square

    if turn == chess.BLACK:
        from_sq = _mirror(from_sq)
        to_sq = _mirror(to_sq)

    dr = to_sq // 8 - from_sq // 8
    df = to_sq % 8 - from_sq % 8

    if move.promotion and move.promotion != chess.QUEEN:
        piece_idx = UNDERPROMO_PIECES.index(move.promotion)
        move_type = 64 + piece_idx * 3 + (df + 1)
    elif (dr, df) in _DELTA_TO_KNIGHT:
        move_type = 56 + _DELTA_TO_KNIGHT[(dr, df)]
    else:
        direction, distance = _DELTA_TO_QUEEN[(dr, df)]
        move_type = direction * 7 + (distance - 1)

    return from_sq * 73 + move_type


def decode_move(action, board):
    """Convert action index back to chess.Move."""
    from_sq = action // 73
    move_type = action % 73
    turn = board.turn

    from_r = from_sq // 8
    from_f = from_sq % 8
    promotion = None

    if move_type < 56:
        d = move_type // 7
        dist = move_type % 7 + 1
        to_r = from_r + QUEEN_DIRS[d][0] * dist
        to_f = from_f + QUEEN_DIRS[d][1] * dist
    elif move_type < 64:
        ki = move_type - 56
        to_r = from_r + KNIGHT_DELTAS[ki][0]
        to_f = from_f + KNIGHT_DELTAS[ki][1]
    else:
        idx = move_type - 64
        promotion = UNDERPROMO_PIECES[idx // 3]
        to_r = from_r + 1
        to_f = from_f + (idx % 3) - 1

    to_sq = to_r * 8 + to_f

    real_from = _mirror(from_sq) if turn == chess.BLACK else from_sq
    real_to = _mirror(to_sq) if turn == chess.BLACK else to_sq

    # Auto-detect queen promotion
    if promotion is None:
        piece = board.piece_at(real_from)
        if piece and piece.piece_type == chess.PAWN:
            if chess.square_rank(real_to) in (0, 7):
                promotion = chess.QUEEN

    return chess.Move(real_from, real_to, promotion=promotion)


# ---- Board observation encoding ----

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def _bb_to_plane(bb, flip):
    """Convert 64-bit bitboard to 8x8 float array (fast, no Python loops)."""
    if bb == 0:
        return np.zeros((8, 8), dtype=np.float32)
    arr = np.frombuffer(int(bb).to_bytes(8, "little"), dtype=np.uint8)
    plane = np.unpackbits(arr, bitorder="little").reshape(8, 8).astype(np.float32)
    if flip:
        plane = plane[::-1].copy()
    return plane


def encode_board(board):
    """Encode board as (19, 8, 8) from current player's perspective.

    Planes 0-5:  my P N B R Q K
    Planes 6-11: opponent P N B R Q K
    Plane 12-15: castling rights (my KS, my QS, opp KS, opp QS)
    Plane 16:    en passant square
    Plane 17:    fifty-move counter / 100
    Plane 18:    ones (color flag)
    """
    planes = np.zeros((19, 8, 8), dtype=np.float32)
    turn = board.turn
    flip = turn == chess.BLACK

    for i, pt in enumerate(PIECE_TYPES):
        planes[i] = _bb_to_plane(board.pieces(pt, turn), flip)
        planes[6 + i] = _bb_to_plane(board.pieces(pt, not turn), flip)

    planes[12] = float(board.has_kingside_castling_rights(turn))
    planes[13] = float(board.has_queenside_castling_rights(turn))
    planes[14] = float(board.has_kingside_castling_rights(not turn))
    planes[15] = float(board.has_queenside_castling_rights(not turn))

    if board.ep_square is not None:
        r, f = board.ep_square // 8, board.ep_square % 8
        if flip:
            r = 7 - r
        planes[16, r, f] = 1.0

    planes[17] = board.halfmove_clock / 100.0
    planes[18] = 1.0
    return planes


# ---- Action planes for dynamics model ----

def actions_to_planes(actions, device="cpu"):
    """Convert (B,) action indices to (B, 2, 8, 8) from/to spatial planes."""
    if isinstance(actions, torch.Tensor):
        actions_np = actions.cpu().numpy()
    else:
        actions_np = np.asarray(actions)

    B = len(actions_np)
    coords = _ACTION_COORDS[actions_np]  # (B, 4)

    planes = torch.zeros(B, 2, 8, 8, device=device)
    b = torch.arange(B, device=device)
    c = torch.from_numpy(coords.astype(np.int64)).to(device)
    planes[b, 0, c[:, 0], c[:, 1]] = 1.0
    planes[b, 1, c[:, 2], c[:, 3]] = 1.0
    return planes


# ---- Game wrapper ----

class ChessGame:
    """Chess environment with the same interface as TicTacToe."""

    def __init__(self):
        self.board = chess.Board()

    @property
    def current_player(self):
        return 1 if self.board.turn == chess.WHITE else -1

    def get_observation(self):
        return encode_board(self.board)

    def legal_actions(self):
        return [encode_move(m, self.board.turn) for m in self.board.legal_moves]

    def step(self, action):
        """Returns (done, reward). Reward is from the acting player's perspective."""
        move = decode_move(action, self.board)
        self.board.push(move)

        if self.board.is_checkmate():
            return True, 1.0

        if self.board.is_game_over(claim_draw=True):
            return True, 0.0

        if self.board.fullmove_number > 256:
            return True, 0.0

        return False, 0.0

    def render(self):
        return str(self.board)


# ---- Verification ----

def verify_encoding():
    """Check encode/decode roundtrips for all legal moves in several positions."""
    fens = [
        chess.STARTING_FEN,
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "r1bqkb1r/pppppppp/2n2n2/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "8/2P5/8/8/8/8/6k1/4K3 w - - 0 1",        # white promotion
        "4k3/8/8/8/8/8/2p5/4K3 b - - 0 1",        # black promotion
        "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
        "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1",
        "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",  # en passant
    ]
    total = 0
    for fen in fens:
        board = chess.Board(fen)
        for move in board.legal_moves:
            action = encode_move(move, board.turn)
            decoded = decode_move(action, board)
            assert move == decoded, f"FAIL: {move} -> {action} -> {decoded} [{fen}]"
            assert 0 <= action < NUM_ACTIONS
            total += 1
    print(f"Verified {total} moves across {len(fens)} positions")
