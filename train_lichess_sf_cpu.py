#!/usr/bin/env python3
"""Train on Lichess Stockfish evals (362M positions) — runs on CPU alongside MPS training."""
import sys
sys.stdout.reconfigure(line_buffering=True)

import os, time, math, random
import chess
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from muzero.chess_game import encode_board, encode_move

# Use MORE positions this time — 5M instead of 2M
NUM_POSITIONS = 5_000_000
MIN_DEPTH = 20
EPOCHS = 3
BATCH_SIZE = 256
LR = 0.001

device = torch.device("cpu")
print(f"Device: {device} (MPS busy with epoch 3)", flush=True)

# Stream from HuggingFace
from datasets import load_dataset
print(f"Streaming {NUM_POSITIONS} SF-evaluated positions (depth>={MIN_DEPTH})...", flush=True)
ds = load_dataset("Lichess/chess-position-evaluations", split="train", streaming=True)

samples = []
skipped = 0
for row in ds:
    if len(samples) >= NUM_POSITIONS:
        break
    if row["depth"] < MIN_DEPTH or not row["line"] or not row["line"].strip():
        skipped += 1; continue
    try:
        board = chess.Board(row["fen"])
    except ValueError:
        skipped += 1; continue
    if board.is_game_over():
        skipped += 1; continue
    best_uci = row["line"].strip().split()[0]
    try:
        move = chess.Move.from_uci(best_uci)
        if move not in board.legal_moves:
            skipped += 1; continue
    except (ValueError, IndexError):
        skipped += 1; continue

    action = encode_move(move, board.turn)
    if row.get("mate") is not None:
        val = 1.0 if row["mate"] > 0 else -1.0
    elif row.get("cp") is not None:
        val = math.tanh(row["cp"] / 400.0)
    else:
        skipped += 1; continue

    samples.append((row["fen"], action, val))
    if len(samples) % 500000 == 0:
        print(f"  {len(samples)} loaded...", flush=True)

print(f"  Done: {len(samples)} positions\n", flush=True)

# Dataset with on-the-fly encoding
class SFDataset(Dataset):
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

random.shuffle(samples)
split = int(0.95 * len(samples))
train_ds = SFDataset(samples[:split])
val_ds = SFDataset(samples[split:])
del samples

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}", flush=True)

from muzero.chess_model import ChessMuZero
from muzero.chess_config import ChessConfig
config = ChessConfig()
model = ChessMuZero(config.observation_channels, config.hidden_channels,
                    config.num_blocks, config.action_space_size)
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
        _, pl, vp = model.initial_inference(obs)
        p_loss = F.cross_entropy(pl, actions)
        v_loss = F.mse_loss(vp.squeeze(-1), values)
        loss = p_loss + v_loss
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        tp += p_loss.item() * obs.size(0)
        tv += v_loss.item() * obs.size(0)
        tn += obs.size(0)
        tc += (pl.argmax(1) == actions).sum().item()
        if (batch_idx + 1) % 2000 == 0:
            print(f"  Epoch {epoch} batch {batch_idx+1}/{len(train_loader)}: "
                  f"p={tp/tn:.3f} v={tv/tn:.3f} acc={100*tc/tn:.1f}%", flush=True)

    model.eval()
    vp2, vv2, vn2, vc2 = 0.0, 0.0, 0, 0
    with torch.no_grad():
        for obs, actions, values in val_loader:
            _, pl, vpr = model.initial_inference(obs)
            vp2 += F.cross_entropy(pl, actions).item() * obs.size(0)
            vv2 += F.mse_loss(vpr.squeeze(-1), values).item() * obs.size(0)
            vn2 += obs.size(0)
            vc2 += (pl.argmax(1) == actions).sum().item()

    print(f"Epoch {epoch}/{EPOCHS} ({time.time()-t0:.0f}s) | "
          f"Train: p={tp/tn:.3f} v={tv/tn:.3f} acc={100*tc/tn:.1f}% | "
          f"Val: p={vp2/vn2:.3f} v={vv2/vn2:.3f} acc={100*vc2/vn2:.1f}%", flush=True)
    os.makedirs("checkpoints", exist_ok=True)
    path = f"checkpoints/stage7_sf_5M_epoch{epoch}.pt"
    torch.save(model.state_dict(), path)
    print(f"  Saved {path}", flush=True)

print(f"\nDone.", flush=True)
