#!/usr/bin/env python3
"""MuZero training on Tic-Tac-Toe."""

import time
import torch
from .config import MuZeroConfig
from .model import MuZeroNetwork
from .replay_buffer import ReplayBuffer
from .trainer import self_play_game, update_weights, evaluate


def get_device(force=None):
    """Auto-detect best available device. Override with force='cpu'/'mps'/'cuda'."""
    if force:
        return torch.device(force)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train(config=None, device=None):
    if config is None:
        config = MuZeroConfig()

    train_device = get_device(device)
    # Self-play uses CPU (MCTS is sequential single-sample inference;
    # GPU kernel launch overhead makes it slower for small models).
    # Training uses GPU when available (batched forward/backward).
    sp_device = torch.device("cpu")

    print(f"Training device: {train_device}")
    print(f"Self-play device: {sp_device}")

    model = MuZeroNetwork(
        config.observation_size, config.action_space_size, config.hidden_size
    ).to(train_device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    replay_buffer = ReplayBuffer(config)

    # CPU copy for self-play
    sp_model = MuZeroNetwork(
        config.observation_size, config.action_space_size, config.hidden_size
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print(f"Training for {config.num_iterations} iterations")
    print(f"  {config.games_per_iteration} self-play games / iter")
    print(f"  {config.training_steps} training steps / iter")
    print(f"  {config.num_simulations} MCTS simulations / move")
    print()

    for iteration in range(1, config.num_iterations + 1):
        t0 = time.time()

        # --- Sync weights to CPU model ---
        sp_model.load_state_dict(
            {k: v.cpu() for k, v in model.state_dict().items()}
        )
        sp_model.eval()

        # --- Self-play (CPU) ---
        outcomes = {1: 0, -1: 0, 0: 0}
        for _ in range(config.games_per_iteration):
            history = self_play_game(config, sp_model, sp_device)
            replay_buffer.save_game(history)
            outcomes[history.outcome] += 1
        t_sp = time.time() - t0

        # --- Training (GPU) ---
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
            f"Buffer: {replay_buffer.total_games():4d} | "
            f"Loss: {losses['total']:.3f} "
            f"(p:{losses['policy']:.3f} v:{losses['value']:.3f} r:{losses['reward']:.3f}) | "
            f"P1/P2/D: {outcomes[1]}/{outcomes[-1]}/{outcomes[0]} | "
            f"sp:{t_sp:.1f}s tr:{t_tr:.1f}s"
        )

        # --- Evaluation ---
        if iteration % config.eval_interval == 0 or iteration == 1:
            sp_model.load_state_dict(
                {k: v.cpu() for k, v in model.state_dict().items()}
            )
            sp_model.eval()
            results = evaluate(config, sp_model, sp_device)
            for opp, r in results.items():
                total = r["win"] + r["loss"] + r["draw"]
                print(
                    f"  vs {opp:8s}: "
                    f"W:{r['win']:2d} L:{r['loss']:2d} D:{r['draw']:2d}  "
                    f"({100*r['win']/total:.0f}% win, {100*r['draw']/total:.0f}% draw)"
                )

    # --- Save model ---
    torch.save(model.state_dict(), "muzero_tictactoe.pt")
    print("\nModel saved to muzero_tictactoe.pt")

    return model, config


def play_interactive(config=None, model=None, device=None):
    """Play against the trained model."""
    from .game import TicTacToe
    from .mcts import run_mcts

    if config is None:
        config = MuZeroConfig()
    if device is None:
        device = torch.device("cpu")
    if model is None:
        model = MuZeroNetwork(
            config.observation_size, config.action_space_size, config.hidden_size
        ).to(device)
        model.load_state_dict(
            torch.load("muzero_tictactoe.pt", weights_only=True, map_location=device)
        )
    model.eval()

    print("You are X (player 1), MuZero is O (player -1)")
    print("Board positions:")
    print(" 0 | 1 | 2")
    print(" 3 | 4 | 5")
    print(" 6 | 7 | 8")
    print()

    game = TicTacToe()
    muzero_player = -1

    while True:
        print(game.render())
        print()

        if game.current_player == muzero_player:
            obs = game.get_observation()
            legal = game.legal_actions()
            root = run_mcts(config, model, obs, legal, device, add_noise=False)
            action = int(root.child_actions[np.argmax(root.child_visits)])
            print(f"MuZero plays: {action}")
        else:
            legal = game.legal_actions()
            while True:
                try:
                    action = int(input(f"Your move {legal}: "))
                    if action in legal:
                        break
                    print("Illegal move!")
                except (ValueError, EOFError):
                    print("Enter a number.")

        done, reward = game.step(action)
        if done:
            print()
            print(game.render())
            if reward == 0:
                print("Draw!")
            elif game.current_player == muzero_player:
                print("MuZero wins!")
            else:
                print("You win!")
            break


if __name__ == "__main__":
    train()
