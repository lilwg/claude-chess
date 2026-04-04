#!/usr/bin/env python3
"""Resume training from epoch 2, run epoch 3 only."""
import sys
sys.stdout.reconfigure(line_buffering=True)

import os, time, random, glob
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from muzero.chess_model import ChessMuZero
from muzero.chess_config import ChessConfig
from muzero.supervised import parse_pgn_file, ChessDataset

PGN_PATTERN = "data/lichess_2013-*.pgn"
MIN_ELO = 1500
BATCH_SIZE = 256
RESUME = "checkpoints/stage6_supervised_6months_epoch2.pt"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}", flush=True)

# Parse all files
pgn_files = sorted(glob.glob(PGN_PATTERN))
all_samples = []
for pgn in pgn_files:
    t0 = time.time()
    samples = parse_pgn_file(pgn, max_games=None, min_elo=MIN_ELO)
    all_samples.extend(samples)
    print(f"  {os.path.basename(pgn)}: {len(samples)} pos ({time.time()-t0:.0f}s), total: {len(all_samples)}", flush=True)

random.shuffle(all_samples)
split = int(0.95 * len(all_samples))
train_ds = ChessDataset(all_samples[:split])
val_ds = ChessDataset(all_samples[split:])
del all_samples

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}", flush=True)

config = ChessConfig()
model = ChessMuZero(
    config.observation_channels, config.hidden_channels,
    config.num_blocks, config.action_space_size,
).to(device)
model.load_state_dict(torch.load(RESUME, weights_only=True, map_location=device))
print(f"Resumed from {RESUME}", flush=True)

# LR for epoch 3: cosine schedule would have decayed to ~0.0002 by epoch 3
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-4)
print(f"Epoch 3 with LR=0.0002\n", flush=True)

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
    tp += F.cross_entropy(pl, actions).item() * obs.size(0)
    tn += obs.size(0)
    tc += (pl.argmax(1) == actions).sum().item()
    if (batch_idx + 1) % 2000 == 0:
        print(f"  batch {batch_idx+1}/{len(train_loader)}: p={tp/tn:.3f} acc={100*tc/tn:.1f}%", flush=True)

model.eval()
vp2, vn2, vc2 = 0.0, 0, 0
with torch.no_grad():
    for obs, actions, values in val_loader:
        obs, actions, values = obs.to(device), actions.to(device), values.to(device)
        _, pl, vpr = model.initial_inference(obs)
        vp2 += F.cross_entropy(pl, actions).item() * obs.size(0)
        vn2 += obs.size(0)
        vc2 += (pl.argmax(1) == actions).sum().item()

print(f"\nEpoch 3 ({time.time()-t0:.0f}s) | Train: p={tp/tn:.3f} acc={100*tc/tn:.1f}% | Val: p={vp2/vn2:.3f} acc={100*vc2/vn2:.1f}%", flush=True)

path = "checkpoints/stage6_supervised_6months_epoch3.pt"
torch.save({k: v.cpu() for k, v in model.state_dict().items()}, path)
print(f"Saved {path}", flush=True)
