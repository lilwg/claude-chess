"""Supervised pretraining on Lichess games.

Trains the representation + prediction networks to predict:
  - Policy: the move played by the human player
  - Value: the game outcome from the current player's perspective
"""

import io
import os
import random
import time
import chess
import chess.pgn
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .chess_game import encode_board, encode_move, NUM_ACTIONS


# ---------------------------------------------------------------------------
# Data loading — stores FENs to save memory, encodes on the fly
# ---------------------------------------------------------------------------

def parse_pgn_file(path, max_games=None, min_elo=1200):
    """Parse PGN into lightweight samples: (fen, action_index, value)."""
    if path.endswith(".zst"):
        import zstandard as zstd
        with open(path, "rb") as fh:
            dctx = zstd.ZstdDecompressor()
            stream = dctx.stream_reader(fh)
            text = io.TextIOWrapper(stream, encoding="utf-8")
            return _parse_pgn_stream(text, max_games, min_elo)
    else:
        with open(path) as fh:
            return _parse_pgn_stream(fh, max_games, min_elo)


def _parse_pgn_stream(stream, max_games, min_elo):
    samples = []  # list of (fen, action_idx, value)
    game_count = 0

    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            break

        try:
            white_elo = int(game.headers.get("WhiteElo", "0"))
            black_elo = int(game.headers.get("BlackElo", "0"))
            if white_elo < min_elo or black_elo < min_elo:
                continue
        except ValueError:
            continue

        result = game.headers.get("Result", "*")
        if result == "1-0":
            outcome = 1
        elif result == "0-1":
            outcome = -1
        elif result == "1/2-1/2":
            outcome = 0
        else:
            continue

        board = game.board()
        for move in game.mainline_moves():
            action = encode_move(move, board.turn)
            value = outcome if board.turn == chess.WHITE else -outcome
            samples.append((board.fen(), action, value))
            board.push(move)

        game_count += 1
        if game_count % 5000 == 0:
            print(f"  Parsed {game_count} games, {len(samples)} positions...", flush=True)

        if max_games and game_count >= max_games:
            break

    print(f"  Done: {game_count} games, {len(samples)} positions", flush=True)
    return samples


class ChessDataset(Dataset):
    """Memory-efficient dataset: stores FENs, encodes boards on the fly."""

    def __init__(self, samples):
        # samples: list of (fen, action, value)
        self.fens = [s[0] for s in samples]
        self.actions = np.array([s[1] for s in samples], dtype=np.int64)
        self.values = np.array([s[2] for s in samples], dtype=np.float32)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        board = chess.Board(self.fens[idx])
        obs = encode_board(board)
        return torch.from_numpy(obs), self.actions[idx], self.values[idx]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def pretrain(model, pgn_path, device, max_games=50000, min_elo=1200,
             epochs=3, batch_size=256, lr=0.001):
    """Pretrain model on human games. Returns the model."""
    print(f"Loading games from {os.path.basename(pgn_path)}...", flush=True)
    samples = parse_pgn_file(pgn_path, max_games=max_games, min_elo=min_elo)

    random.shuffle(samples)
    split = int(0.95 * len(samples))
    train_ds = ChessDataset(samples[:split])
    val_ds = ChessDataset(samples[split:])
    del samples  # free memory

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=pin)

    print(f"Training: {len(train_ds)} positions, Validation: {len(val_ds)} positions",
          flush=True)
    print(f"Device: {device}, Batch size: {batch_size}, Epochs: {epochs}", flush=True)
    print(flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        train_p, train_v, train_n = 0.0, 0.0, 0
        train_correct = 0

        for obs, actions, values in train_loader:
            obs = obs.to(device)
            actions = actions.to(device)
            values = values.to(device)

            hidden, policy_logits, value_pred = model.initial_inference(obs)

            p_loss = F.cross_entropy(policy_logits, actions)
            v_loss = F.mse_loss(value_pred.squeeze(-1), values)
            loss = p_loss + v_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_p += p_loss.item() * obs.size(0)
            train_v += v_loss.item() * obs.size(0)
            train_n += obs.size(0)
            train_correct += (policy_logits.argmax(dim=1) == actions).sum().item()
            scheduler.step()

        model.eval()
        val_p, val_v, val_n = 0.0, 0.0, 0
        val_correct = 0

        with torch.no_grad():
            for obs, actions, values in val_loader:
                obs = obs.to(device)
                actions = actions.to(device)
                values = values.to(device)

                _, policy_logits, value_pred = model.initial_inference(obs)
                val_p += F.cross_entropy(policy_logits, actions).item() * obs.size(0)
                val_v += F.mse_loss(value_pred.squeeze(-1), values).item() * obs.size(0)
                val_n += obs.size(0)
                val_correct += (policy_logits.argmax(dim=1) == actions).sum().item()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch}/{epochs} ({elapsed:.0f}s) | "
            f"Train: p={train_p/train_n:.3f} v={train_v/train_n:.3f} "
            f"acc={100*train_correct/train_n:.1f}% | "
            f"Val: p={val_p/val_n:.3f} v={val_v/val_n:.3f} "
            f"acc={100*val_correct/val_n:.1f}%",
            flush=True,
        )

    return model
