#!/usr/bin/env python3
"""Evaluate AlphaZero model Elo against Stockfish."""
import sys
sys.stdout.reconfigure(line_buffering=True)

import torch, numpy as np
from muzero.alphazero_model import AlphaZeroNet
from muzero.alphazero_mcts import alphazero_mcts
from muzero.chess_game import ChessGame, decode_move
import chess, chess.engine, random

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/alphazero_wdl_2M.pt"
NUM_SIMS = 50
GAMES_PER_LEVEL = 6
SF_ELOS = [1320, 1500, 1700, 2000, 2400]

device = torch.device("cpu")
model = AlphaZeroNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location=device))
model.eval()
print(f"Loaded {MODEL_PATH}", flush=True)

for sf_elo in SF_ELOS:
    w, d, l = 0, 0, 0
    for g in range(GAMES_PER_LEVEL):
        model_white = g % 2 == 0
        engine = chess.engine.SimpleEngine.popen_uci("stockfish")
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo})
        board = chess.Board()

        try:
            while not board.is_game_over(claim_draw=True) and board.fullmove_number <= 256:
                is_model = (board.turn == chess.WHITE) == model_white
                if is_model:
                    root = alphazero_mcts(model, board, device, num_sims=NUM_SIMS,
                                          add_noise=False)
                    action = int(root.child_actions[np.argmax(root.child_visits)])
                    move = decode_move(action, board)
                else:
                    result = engine.play(board, chess.engine.Limit(time=0.1))
                    move = result.move
                board.push(move)

            result = board.result(claim_draw=True)
            if result == "1-0":
                (w if model_white else l) and None or (w := w + 1) if model_white else (l := l + 1)
            elif result == "0-1":
                (l if model_white else w) and None or (l := l + 1) if model_white else (w := w + 1)
            else:
                d += 1
        finally:
            engine.quit()
        sys.stdout.write(".")
        sys.stdout.flush()

    score = (w + 0.5 * d) / GAMES_PER_LEVEL
    print(f"\n  vs SF {sf_elo}: +{w} ={d} -{l} (score: {score:.1%})", flush=True)
    if l >= GAMES_PER_LEVEL - 1 and w == 0:
        break

print("Done.", flush=True)
