#!/usr/bin/env python3
"""Flask backend for MuZero chess web interface."""

import sys
import os
import time

# Add parent dir so we can import muzero.*
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import numpy as np
import torch
from flask import Flask, render_template, jsonify, request

from muzero.chess_model import ChessMuZero
from muzero.chess_config import ChessConfig
from muzero.chess_game import encode_board, encode_move, decode_move
from muzero.alphazero_mcts import alphazero_mcts

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHECKPOINT = os.environ.get(
    "CHESS_CHECKPOINT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "checkpoints", "stage6_38M_epoch3_elo2000.pt"),
)
NUM_SIMS = int(os.environ.get("CHESS_SIMS", "50"))

# ---------------------------------------------------------------------------
# Model wrapper (adapts ChessMuZero to the evaluate() interface MCTS expects)
# ---------------------------------------------------------------------------

class MuZeroWrapper:
    def __init__(self, model):
        self.model = model

    def evaluate(self, obs):
        h, p, v = self.model.initial_inference(obs)
        return p, v.squeeze(-1), None


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)

# Global state
device = torch.device("cpu")  # CPU is fine for single-game inference
model = None
wrapper = None
board = chess.Board()


def load_model():
    global model, wrapper
    config = ChessConfig()
    model = ChessMuZero(
        obs_channels=config.observation_channels,
        hidden_channels=config.hidden_channels,
        num_blocks=config.num_blocks,
        action_size=config.action_space_size,
    )
    state = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    wrapper = MuZeroWrapper(model)
    print(f"Loaded model from {CHECKPOINT}")
    print(f"  hidden_channels={config.hidden_channels}, num_blocks={config.num_blocks}")
    print(f"  MCTS simulations: {NUM_SIMS}")


def evaluate_position(b):
    """Return evaluation from White's perspective in [-1, +1]."""
    obs = torch.as_tensor(encode_board(b), dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        _, _, v = model.initial_inference(obs)
    val = v.item()
    # val is from current player's perspective; flip if Black to move
    if b.turn == chess.BLACK:
        val = -val
    return val


def get_model_move(b):
    """Run MCTS and return (uci_move, eval_from_white)."""
    t0 = time.time()
    root = alphazero_mcts(
        wrapper, b, device,
        num_sims=NUM_SIMS,
        add_noise=False,
    )
    # Pick best by visit count
    best_idx = int(np.argmax(root.child_visits))
    action = int(root.child_actions[best_idx])
    move = decode_move(action, b)

    # Value from root (Black's perspective since it's Black to move)
    root_val = root.value()
    # Convert to White's perspective
    eval_white = -root_val if b.turn == chess.BLACK else root_val

    elapsed = time.time() - t0
    print(f"  Model move: {move.uci()} (eval={eval_white:+.3f}, "
          f"visits={root.child_visits[best_idx]}, {elapsed:.2f}s)")

    return move, eval_white


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/new_game", methods=["POST"])
def new_game():
    global board
    board = chess.Board()
    return jsonify({
        "fen": board.fen(),
        "eval": 0.0,
        "game_over": False,
        "result": None,
    })


@app.route("/move", methods=["POST"])
def make_move():
    global board
    data = request.get_json()
    uci = data.get("move")

    if not uci:
        return jsonify({"error": "No move provided"}), 400

    try:
        user_move = chess.Move.from_uci(uci)
    except ValueError:
        return jsonify({"error": f"Invalid UCI move: {uci}"}), 400

    if user_move not in board.legal_moves:
        return jsonify({"error": f"Illegal move: {uci}"}), 400

    # Apply user's move (White)
    board.push(user_move)
    print(f"User played: {user_move.uci()}")

    # Check if game is over after user's move
    if board.is_game_over(claim_draw=True):
        result = board.result(claim_draw=True)
        ev = evaluate_position(board)
        return jsonify({
            "fen": board.fen(),
            "user_move": user_move.uci(),
            "model_move": None,
            "eval": ev,
            "game_over": True,
            "result": result,
        })

    # Model plays Black
    model_move, ev = get_model_move(board)
    board.push(model_move)

    game_over = board.is_game_over(claim_draw=True)
    result = board.result(claim_draw=True) if game_over else None

    return jsonify({
        "fen": board.fen(),
        "user_move": user_move.uci(),
        "model_move": model_move.uci(),
        "eval": ev,
        "game_over": game_over,
        "result": result,
    })


@app.route("/evaluate", methods=["GET"])
def evaluate():
    ev = evaluate_position(board)
    return jsonify({
        "fen": board.fen(),
        "eval": ev,
    })


@app.route("/legal_moves", methods=["GET"])
def legal_moves():
    """Return legal moves for the current position (for debugging)."""
    moves = [m.uci() for m in board.legal_moves]
    return jsonify({"moves": moves, "fen": board.fen()})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    load_model()
    port = int(os.environ.get("PORT", "5000"))
    print(f"Starting chess server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
