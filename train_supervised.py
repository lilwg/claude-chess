#!/usr/bin/env python3
"""Supervised pretraining on multiple Lichess PGN files.

Strategy: parse all files into FEN samples, train on MPS with on-the-fly encoding.
Memory-safe: stores FEN strings (~100 bytes each), not encoded arrays (~5KB each).
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import os, glob, torch, time, random
import torch.nn.functional as F
from torch.utils.data import DataLoader
from muzero.chess_model import ChessMuZero
from muzero.chess_config import ChessConfig
from muzero.supervised import parse_pgn_file, ChessDataset

# --- Config ---
PGN_PATTERN = "data/lichess_2013-*.pgn"
MAX_GAMES_PER_FILE = None   # None = all games
MIN_ELO = 1500              # higher quality games
EPOCHS = 3
BATCH_SIZE = 256
LR = 0.001
RESUME_FROM = None          # start fresh with new architecture

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- Parse all files ---
pgn_files = sorted(glob.glob(PGN_PATTERN))
print(f"Found {len(pgn_files)} PGN files, device={device}", flush=True)

all_samples = []
for pgn in pgn_files:
    t0 = time.time()
    samples = parse_pgn_file(pgn, max_games=MAX_GAMES_PER_FILE, min_elo=MIN_ELO)
    all_samples.extend(samples)
    print(f"  {os.path.basename(pgn)}: {len(samples)} positions ({time.time()-t0:.0f}s), "
          f"total: {len(all_samples)}", flush=True)

print(f"\nTotal: {len(all_samples)} positions from {len(pgn_files)} files", flush=True)

# --- Split and create datasets ---
random.shuffle(all_samples)
split = int(0.95 * len(all_samples))
train_ds = ChessDataset(all_samples[:split])
val_ds = ChessDataset(all_samples[split:])
del all_samples

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}\n", flush=True)

# --- Model ---
config = ChessConfig()
model = ChessMuZero(
    config.observation_channels, config.hidden_channels,
    config.num_blocks, config.action_space_size,
).to(device)

if RESUME_FROM and os.path.exists(RESUME_FROM):
    model.load_state_dict(torch.load(RESUME_FROM, weights_only=True, map_location=device))
    print(f"Resumed from {RESUME_FROM}", flush=True)

params = sum(p.numel() for p in model.parameters())
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
total_steps = EPOCHS * len(train_loader)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
print(f"Model: {params:,} params, {EPOCHS} epochs, {total_steps} total steps", flush=True)

# --- Train ---
for epoch in range(1, EPOCHS + 1):
    model.train()
    t0 = time.time()
    tp, tn, tc = 0.0, 0, 0
    for batch_idx, (obs, actions, values) in enumerate(train_loader):
        obs, actions, values = obs.to(device), actions.to(device), values.to(device)
        _, pl, vp = model.initial_inference(obs)
        loss = F.cross_entropy(pl, actions) + F.mse_loss(vp.squeeze(-1), values)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        tp += F.cross_entropy(pl, actions).item() * obs.size(0)
        tn += obs.size(0)
        tc += (pl.argmax(1) == actions).sum().item()
        if (batch_idx + 1) % 2000 == 0:
            print(f"  Epoch {epoch} batch {batch_idx+1}/{len(train_loader)}: "
                  f"p={tp/tn:.3f} acc={100*tc/tn:.1f}%", flush=True)

    model.eval()
    vp2, vn2, vc2 = 0.0, 0, 0
    with torch.no_grad():
        for obs, actions, values in val_loader:
            obs, actions, values = obs.to(device), actions.to(device), values.to(device)
            _, pl, vpr = model.initial_inference(obs)
            vp2 += F.cross_entropy(pl, actions).item() * obs.size(0)
            vn2 += obs.size(0)
            vc2 += (pl.argmax(1) == actions).sum().item()

    elapsed = time.time() - t0
    print(
        f"Epoch {epoch}/{EPOCHS} ({elapsed:.0f}s) | "
        f"Train: p={tp/tn:.3f} acc={100*tc/tn:.1f}% | "
        f"Val: p={vp2/vn2:.3f} acc={100*vc2/vn2:.1f}%",
        flush=True,
    )

    # Save after each epoch
    os.makedirs("checkpoints", exist_ok=True)
    path = f"checkpoints/stage6_supervised_6months_epoch{epoch}.pt"
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, path)
    print(f"  Saved {path}", flush=True)

final = "checkpoints/stage6_supervised_6months.pt"
torch.save({k: v.cpu() for k, v in model.state_dict().items()}, final)
print(f"\nDone. Saved {final}", flush=True)
