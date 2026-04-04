"""Evaluate model Elo by playing against Stockfish at calibrated levels."""

import sys
import chess
import chess.engine
import torch
import numpy as np

from .chess_game import ChessGame, encode_move, decode_move
from .chess_model import ChessMuZero
from .chess_config import ChessConfig
from .mcts import run_mcts


def play_game(model, config, device, stockfish_path, stockfish_elo,
              model_is_white, use_mcts=True):
    """Play one game: model vs Stockfish. Returns 1 (model wins), 0 (draw), -1 (loss)."""
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": stockfish_elo})

    game = ChessGame()

    try:
        while not game.board.is_game_over(claim_draw=True) and game.board.fullmove_number <= 256:
            is_model_turn = (game.board.turn == chess.WHITE) == model_is_white

            if is_model_turn:
                if use_mcts:
                    obs = game.get_observation()
                    legal = game.legal_actions()
                    root = run_mcts(config, model, obs, legal, device, add_noise=False)
                    action = int(root.child_actions[np.argmax(root.child_visits)])
                else:
                    # Greedy policy (no search)
                    obs = torch.as_tensor(game.get_observation(), dtype=torch.float32)
                    obs = obs.unsqueeze(0).to(device)
                    with torch.no_grad():
                        _, policy, _ = model.initial_inference(obs)
                    probs = torch.softmax(policy.squeeze(0), dim=0)
                    legal = game.legal_actions()
                    action = max(legal, key=lambda a: probs[a].item())
                move = decode_move(action, game.board)
            else:
                result = engine.play(game.board, chess.engine.Limit(time=0.1))
                move = result.move

            game.board.push(move)

        # Determine result
        result = game.board.result(claim_draw=True)
        if result == "1-0":
            return 1 if model_is_white else -1
        elif result == "0-1":
            return -1 if model_is_white else 1
        else:
            return 0
    finally:
        engine.quit()


def estimate_elo(model, config, device, stockfish_path="stockfish",
                 use_mcts=True, games_per_level=10):
    """Binary search for the model's Elo by playing Stockfish at various levels."""
    print(f"Evaluating {'with' if use_mcts else 'without'} MCTS "
          f"({config.num_simulations} sims)" if use_mcts else "", flush=True)

    # Test at specific Elo levels
    test_elos = [1320, 1500, 1700, 2000, 2400]
    results = {}

    for sf_elo in test_elos:
        wins, draws, losses = 0, 0, 0
        for g in range(games_per_level):
            model_white = (g % 2 == 0)
            r = play_game(model, config, device, stockfish_path, sf_elo,
                          model_white, use_mcts)
            if r == 1:
                wins += 1
            elif r == 0:
                draws += 1
            else:
                losses += 1
            sys.stdout.write(".")
            sys.stdout.flush()

        score = (wins + 0.5 * draws) / games_per_level
        results[sf_elo] = {"wins": wins, "draws": draws, "losses": losses, "score": score}
        print(
            f"\n  vs SF {sf_elo}: +{wins} ={draws} -{losses} "
            f"(score: {score:.1%})",
            flush=True,
        )

        # Early stop if getting crushed
        if losses >= games_per_level - 1 and wins == 0:
            break

    # Estimate Elo from scores using the simplistic approach:
    # If score=50% against SF X, model Elo ≈ X
    # Interpolate between levels
    estimated = _interpolate_elo(results)
    return estimated, results


def _interpolate_elo(results):
    """Estimate Elo from win rates against calibrated opponents."""
    # Find the crossover point where score goes from >50% to <50%
    elos = sorted(results.keys())
    scores = [results[e]["score"] for e in elos]

    # If we beat everyone, estimate is above the highest tested
    if all(s >= 0.5 for s in scores):
        return elos[-1] + 100

    # If we lose to everyone, estimate is below the lowest tested
    if all(s <= 0.5 for s in scores):
        # Rough extrapolation from lowest level
        s = scores[0]
        if s > 0:
            # Elo difference from expected score: dElo = -400*log10(1/score - 1)
            import math
            d = -400 * math.log10(max(1 / s - 1, 0.01))
            return int(elos[0] + d)
        return elos[0] - 400

    # Interpolate
    for i in range(len(elos) - 1):
        if scores[i] >= 0.5 and scores[i + 1] < 0.5:
            # Linear interpolation
            frac = (0.5 - scores[i]) / (scores[i + 1] - scores[i])
            return int(elos[i] + frac * (elos[i + 1] - elos[i]))

    return elos[0]
