#!/usr/bin/env python3
"""Train AlphaZero model on Lichess Stockfish evaluations.

Uses:
  - WDL value targets (win/draw/loss from centipawn scores)
  - Soft policy targets (label smoothing over legal moves)
  - 2M+ positions from Lichess evaluation database (depth 20+)
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
from muzero.alphazero_model import AlphaZeroNet

# --- Config ---
NUM_POSITIONS = 2_000_000
MIN_DEPTH = 20
EPOCHS = 5
BATCH_SIZE = 256
LR = 0.001
LABEL_SMOOTHING = 0.1       # 10% probability spread to non-target moves


def cp_to_wdl(cp, mate=None):
    """Convert centipawn/mate to (win, draw, loss) target probabilities."""
    if mate is not None:
        if mate > 0:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # Win probability from logistic model (calibrated to Lichess data)
    win_rate = 1.0 / (1.0 + math.exp(-cp / 111.0))

    # Draw probability: peaks near eval=0, falls off for decisive evals
    draw_rate = max(0.0, 0.5 * math.exp(-(cp / 200.0) ** 2))

    w = win_rate * (1.0 - draw_rate)
    l = (1.0 - win_rate) * (1.0 - draw_rate)
    d = draw_rate

    total = w + d + l
    return np.array([w / total, d / total, l / total], dtype=np.float32)


def load_lichess_evals(num_positions, min_depth):
    """Stream from HuggingFace Lichess evaluation dataset."""
    from datasets import load_dataset

    print(f"Streaming {num_positions} positions (min depth {min_depth})...", flush=True)
    ds = load_dataset("Lichess/chess-position-evaluations", split="train", streaming=True)

    samples = []
    skipped = 0

    for row in ds:
        if len(samples) >= num_positions:
            break

        if row["depth"] < min_depth or not row["line"] or not row["line"].strip():
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

        best_uci = row["line"].strip().split()[0]
        try:
            move = chess.Move.from_uci(best_uci)
            if move not in board.legal_moves:
                skipped += 1
                continue
        except (ValueError, IndexError):
            skipped += 1
            continue

        action = encode_move(move, board.turn)
        wdl = cp_to_wdl(row.get("cp"), row.get("mate"))

        samples.append((fen, action, wdl))

        if len(samples) % 200000 == 0:
            print(f"  {len(samples)} loaded (skipped {skipped})...", flush=True)

    print(f"  Done: {len(samples)} positions", flush=True)
    return samples


class AZDataset(Dataset):
    def __init__(self, samples):
        self.fens = [s[0] for s in samples]
        self.actions = np.array([s[1] for s in samples], dtype=np.int64)
        self.wdl = np.stack([s[2] for s in samples])  # (N, 3)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        board = chess.Board(self.fens[idx])
        obs = encode_board(board)
        return torch.from_numpy(obs), self.actions[idx], torch.from_numpy(self.wdl[idx])


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    t0 = time.time()
    samples = load_lichess_evals(NUM_POSITIONS, MIN_DEPTH)
    print(f"Loaded in {time.time()-t0:.0f}s\n", flush=True)

    random.shuffle(samples)
    split = int(0.95 * len(samples))
    train_ds = AZDataset(samples[:split])
    val_ds = AZDataset(samples[split:])
    del samples

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}", flush=True)

    model = AlphaZeroNet().to(device)
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    total_steps = EPOCHS * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    print(f"AlphaZero model: {params:,} params, WDL value head", flush=True)
    print(f"{EPOCHS} epochs, label_smoothing={LABEL_SMOOTHING}\n", flush=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        tp, tv, tn, tc = 0.0, 0.0, 0, 0

        for batch_idx, (obs, actions, wdl_targets) in enumerate(train_loader):
            obs = obs.to(device)
            actions = actions.to(device)
            wdl_targets = wdl_targets.to(device)

            policy_logits, wdl_logits = model(obs)

            # Policy loss with label smoothing (soft targets)
            p_loss = F.cross_entropy(policy_logits, actions,
                                     label_smoothing=LABEL_SMOOTHING)

            # WDL loss: cross-entropy with soft WDL targets
            v_loss = -(wdl_targets * F.log_softmax(wdl_logits, dim=-1)).sum(dim=-1).mean()

            loss = p_loss + v_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            tp += p_loss.item() * obs.size(0)
            tv += v_loss.item() * obs.size(0)
            tn += obs.size(0)
            tc += (policy_logits.argmax(1) == actions).sum().item()

            if (batch_idx + 1) % 1000 == 0:
                print(f"  Epoch {epoch} batch {batch_idx+1}/{len(train_loader)}: "
                      f"p={tp/tn:.3f} v={tv/tn:.3f} acc={100*tc/tn:.1f}%", flush=True)

        model.eval()
        vp, vv, vn, vc = 0.0, 0.0, 0, 0
        with torch.no_grad():
            for obs, actions, wdl_targets in val_loader:
                obs = obs.to(device)
                actions = actions.to(device)
                wdl_targets = wdl_targets.to(device)
                pl, wdl_logits = model(obs)
                vp += F.cross_entropy(pl, actions).item() * obs.size(0)
                vv += (-(wdl_targets * F.log_softmax(wdl_logits, dim=-1)).sum(-1)).mean().item() * obs.size(0)
                vn += obs.size(0)
                vc += (pl.argmax(1) == actions).sum().item()

        print(
            f"Epoch {epoch}/{EPOCHS} ({time.time()-t0:.0f}s) | "
            f"Train: p={tp/tn:.3f} v={tv/tn:.3f} acc={100*tc/tn:.1f}% | "
            f"Val: p={vp/vn:.3f} v={vv/vn:.3f} acc={100*vc/vn:.1f}%",
            flush=True,
        )
        os.makedirs("checkpoints", exist_ok=True)
        path = f"checkpoints/alphazero_wdl_epoch{epoch}.pt"
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, path)
        print(f"  Saved {path}", flush=True)

    final = "checkpoints/alphazero_wdl_2M.pt"
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, final)
    print(f"\nDone. Saved {final}", flush=True)


if __name__ == "__main__":
    main()
