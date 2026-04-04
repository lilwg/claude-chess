#!/usr/bin/env python3
"""MuZero Chess — full training pipeline with checkpointing."""

import os
import sys
import time
import torch

from muzero.chess_config import ChessConfig
from muzero.chess_model import ChessMuZero
from muzero.chess_game import ChessGame, decode_move
from muzero.replay_buffer import ReplayBuffer
from muzero.batched import batched_self_play
from muzero.trainer import update_weights
from muzero.main import get_device

CHECKPOINT_DIR = "checkpoints"


def train(config=None, resume_from=None):
    if config is None:
        config = ChessConfig()

    train_device = get_device()
    sp_device = torch.device("cpu")  # batched CPU is fastest for self-play

    print(f"Training device: {train_device}")
    print(f"Self-play device: {sp_device}")

    model = ChessMuZero(
        config.observation_channels, config.hidden_channels,
        config.num_blocks, config.action_space_size,
    ).to(train_device)

    if resume_from:
        model.load_state_dict(
            torch.load(resume_from, weights_only=True, map_location=train_device)
        )
        print(f"Resumed from {resume_from}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    replay_buffer = ReplayBuffer(config)

    # CPU copy for self-play
    sp_model = ChessMuZero(
        config.observation_channels, config.hidden_channels,
        config.num_blocks, config.action_space_size,
    )

    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params:,} params ({config.num_blocks} blocks, {config.hidden_channels}ch)")
    print(f"Self-play: {config.games_per_iteration} games/iter, "
          f"{config.num_simulations} sims/move, batch={config.games_per_iteration}")
    print(f"Training: {config.training_steps} steps/iter, batch={config.batch_size}")
    print()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for iteration in range(1, config.num_iterations + 1):
        t0 = time.time()

        # --- Sync weights to CPU ---
        sp_model.load_state_dict(
            {k: v.cpu() for k, v in model.state_dict().items()}
        )
        sp_model.eval()

        # --- Batched self-play ---
        histories = batched_self_play(
            config, sp_model, sp_device,
            num_games=config.games_per_iteration, game_cls=ChessGame
        )
        for h in histories:
            replay_buffer.save_game(h)

        outcomes = {1: 0, -1: 0, 0: 0}
        total_moves = 0
        for h in histories:
            outcomes[h.outcome] += 1
            total_moves += h.length
        avg_len = total_moves / len(histories)
        t_sp = time.time() - t0

        # --- Training ---
        t1 = time.time()
        model.train()
        losses = {"total": 0, "policy": 0, "value": 0, "reward": 0}
        for _ in range(config.training_steps):
            loss = update_weights(model, optimizer, replay_buffer, config, train_device)
            for k in losses:
                losses[k] += loss[k]
        for k in losses:
            losses[k] /= config.training_steps
        t_tr = time.time() - t1

        elapsed = time.time() - t0
        print(
            f"Iter {iteration:3d}/{config.num_iterations} | "
            f"Buf: {replay_buffer.total_games():4d} | "
            f"Loss: {losses['total']:.3f} (p:{losses['policy']:.3f} v:{losses['value']:.3f}) | "
            f"W/B/D: {outcomes[1]}/{outcomes[-1]}/{outcomes[0]} | "
            f"Avg len: {avg_len:.0f} | "
            f"sp:{t_sp:.0f}s tr:{t_tr:.1f}s",
            flush=True,
        )

        # --- Checkpoint ---
        if iteration % config.eval_interval == 0:
            path = os.path.join(CHECKPOINT_DIR, f"chess_muzero_iter{iteration}.pt")
            torch.save(model.state_dict(), path)
            print(f"  Saved {path}", flush=True)

    # Final save
    final = os.path.join(CHECKPOINT_DIR, "chess_muzero_latest.pt")
    torch.save(model.state_dict(), final)
    print(f"\nTraining complete. Final model: {final}")
    return model, config


def play(model_path=None, model=None):
    """Play interactively against the chess model."""
    import chess
    from muzero.mcts import run_mcts

    config = ChessConfig(num_simulations=200)
    device = torch.device("cpu")

    if model is None:
        model = ChessMuZero(
            config.observation_channels, config.hidden_channels,
            config.num_blocks, config.action_space_size,
        )
        if model_path is None:
            # Try to find the best checkpoint
            for p in [
                "checkpoints/chess_muzero_latest.pt",
                "chess_pretrained.pt",
                "checkpoints/stage1_supervised_10k_games.pt",
            ]:
                if os.path.exists(p):
                    model_path = p
                    break
        if model_path:
            model.load_state_dict(
                torch.load(model_path, weights_only=True, map_location=device)
            )
            print(f"Loaded: {model_path}")
        else:
            print("No model found — playing with random weights")
    model.eval()

    print("\nYou are White. Enter moves in UCI format (e.g., e2e4).")
    print("Type 'quit' to exit.\n")

    game = ChessGame()

    while True:
        print(game.board)
        print()

        if game.board.is_game_over(claim_draw=True):
            print(f"Game over: {game.board.result(claim_draw=True)}")
            break

        if game.board.turn == chess.WHITE:
            # Human's turn
            while True:
                try:
                    uci = input("Your move: ").strip()
                    if uci == "quit":
                        return
                    move = chess.Move.from_uci(uci)
                    if move in game.board.legal_moves:
                        break
                    print(f"Illegal. Legal: {', '.join(m.uci() for m in game.board.legal_moves)}")
                except (ValueError, EOFError):
                    print("Enter a valid UCI move (e.g., e2e4)")
            from muzero.chess_game import encode_move
            action = encode_move(move, game.board.turn)
        else:
            # MuZero's turn
            obs = game.get_observation()
            legal = game.legal_actions()
            root = run_mcts(config, model, obs, legal, device, add_noise=False)
            action = int(root.child_actions[np.argmax(root.child_visits)])
            move = decode_move(action, game.board)
            print(f"MuZero plays: {move.uci()} ({game.board.san(move)})")

        game.board.push(move)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "play":
        model_path = sys.argv[2] if len(sys.argv) > 2 else None
        play(model_path)
    else:
        resume = None
        if len(sys.argv) > 1:
            resume = sys.argv[1]
        elif os.path.exists("chess_pretrained.pt"):
            resume = "chess_pretrained.pt"
            print("Starting from supervised pretrained model")
        train(resume_from=resume)
