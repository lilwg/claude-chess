#!/usr/bin/env python3
"""Train on Lichess's 362M Stockfish-evaluated positions (streamed from HuggingFace).

Each position has Stockfish's best move + centipawn evaluation.
This gives us unlimited high-quality data without running Stockfish locally.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import os, time, math, random
import chess
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from muzero.chess_game import encode_board, encode_move, NUM_ACTIONS
from muzero.chess_model import ChessMuZero
from muzero.chess_config import ChessConfig

# --- Config ---
NUM_POSITIONS = 2_000_000   # how many to load (2M = good balance of speed vs quality)
EPOCHS = 3
BATCH_SIZE = 256
LR = 0.001
MIN_DEPTH = 20              # only use deep evaluations


def cp_to_value(cp):
    """Convert centipawn to [-1, 1]."""
    return math.tanh(cp / 400.0)


def load_lichess_evals(num_positions, min_depth):
    """Stream positions from HuggingFace Lichess eval dataset."""
    from datasets import load_dataset

    print(f"Streaming {num_positions} positions (min depth {min_depth})...", flush=True)
    ds = load_dataset("Lichess/chess-position-evaluations", split="train", streaming=True)

    samples = []  # (fen, action, value)
    skipped = 0

    for row in ds:
        if len(samples) >= num_positions:
            break

        # Filter for quality
        if row["depth"] < min_depth:
            skipped += 1
            continue
        if row["line"] is None or len(row["line"].strip()) == 0:
            skipped += 1
            continue

        fen = row["fen"]
        try:
            board = chess.Board(fen)
        except ValueError:
            skipped += 1
            continue

        if board.is_game_over():
            skipped += 1
            continue

        # Extract best move from the PV line
        best_move_uci = row["line"].strip().split()[0]
        try:
            move = chess.Move.from_uci(best_move_uci)
            if move not in board.legal_moves:
                skipped += 1
                continue
        except (ValueError, IndexError):
            skipped += 1
            continue

        # Value from current player's perspective
        if row["mate"] is not None:
            val = 1.0 if row["mate"] > 0 else -1.0
        elif row["cp"] is not None:
            val = cp_to_value(row["cp"])
        else:
            skipped += 1
            continue

        action = encode_move(move, board.turn)
        samples.append((fen, action, val))

        if len(samples) % 100000 == 0:
            print(f"  {len(samples)} positions loaded (skipped {skipped})...", flush=True)

    print(f"  Done: {len(samples)} positions (skipped {skipped})", flush=True)
    return samples


class EvalDataset(Dataset):
    def __init__(self, samples):
        self.fens = [s[0] for s in samples]
        self.actions = np.array([s[1] for s in samples], dtype=np.int64)
        self.values = np.array([s[2] for s in samples], dtype=np.float32)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        board = chess.Board(self.fens[idx])
        obs = encode_board(board)
        return torch.from_numpy(obs), self.actions[idx], self.values[idx]


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    t0 = time.time()
    samples = load_lichess_evals(NUM_POSITIONS, MIN_DEPTH)
    print(f"Loading took {time.time()-t0:.0f}s\n", flush=True)

    random.shuffle(samples)
    split = int(0.95 * len(samples))
    train_ds = EvalDataset(samples[:split])
    val_ds = EvalDataset(samples[split:])
    del samples

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}", flush=True)

    config = ChessConfig()
    model = ChessMuZero(
        config.observation_channels, config.hidden_channels,
        config.num_blocks, config.action_space_size,
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    total_steps = EPOCHS * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    print(f"Model: {params:,} params, {EPOCHS} epochs, {total_steps} steps\n", flush=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        tp, tv, tn, tc = 0.0, 0.0, 0, 0
        for batch_idx, (obs, actions, values) in enumerate(train_loader):
            obs, actions, values = obs.to(device), actions.to(device), values.to(device)
            _, pl, vp = model.initial_inference(obs)
            p_loss = F.cross_entropy(pl, actions)
            v_loss = F.mse_loss(vp.squeeze(-1), values)
            loss = p_loss + v_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            tp += p_loss.item() * obs.size(0)
            tv += v_loss.item() * obs.size(0)
            tn += obs.size(0)
            tc += (pl.argmax(1) == actions).sum().item()
            if (batch_idx + 1) % 1000 == 0:
                print(f"  Epoch {epoch} batch {batch_idx+1}/{len(train_loader)}: "
                      f"p={tp/tn:.3f} v={tv/tn:.3f} acc={100*tc/tn:.1f}%", flush=True)

        model.eval()
        vp2, vv2, vn2, vc2 = 0.0, 0.0, 0, 0
        with torch.no_grad():
            for obs, actions, values in val_loader:
                obs, actions, values = obs.to(device), actions.to(device), values.to(device)
                _, pl, vpr = model.initial_inference(obs)
                vp2 += F.cross_entropy(pl, actions).item() * obs.size(0)
                vv2 += F.mse_loss(vpr.squeeze(-1), values).item() * obs.size(0)
                vn2 += obs.size(0)
                vc2 += (pl.argmax(1) == actions).sum().item()

        print(
            f"Epoch {epoch}/{EPOCHS} ({time.time()-t0:.0f}s) | "
            f"Train: p={tp/tn:.3f} v={tv/tn:.3f} acc={100*tc/tn:.1f}% | "
            f"Val: p={vp2/vn2:.3f} v={vv2/vn2:.3f} acc={100*vc2/vn2:.1f}%",
            flush=True,
        )
        os.makedirs("checkpoints", exist_ok=True)
        path = f"checkpoints/stage7_lichess_evals_epoch{epoch}.pt"
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, path)
        print(f"  Saved {path}", flush=True)

    final = "checkpoints/stage7_lichess_evals.pt"
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, final)
    print(f"\nDone. Saved {final}", flush=True)


if __name__ == "__main__":
    main()
