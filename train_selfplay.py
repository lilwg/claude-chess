#!/usr/bin/env python3
"""MuZero self-play training for chess — fast version with MPS inference."""
import sys
sys.stdout.reconfigure(line_buffering=True)

import os, time, torch
from muzero.chess_config import ChessConfig
from muzero.chess_model import ChessMuZero
from muzero.chess_game import ChessGame
from muzero.replay_buffer import ReplayBuffer
from muzero.batched import batched_self_play
from muzero.trainer import update_weights

# --- Config ---
RESUME_FROM = "checkpoints/stage3_full_90k_elo1600.pt"
CHECKPOINT_DIR = "checkpoints"
NUM_ITERATIONS = 50
GAMES_PER_ITER = 32        # large batch = MPS efficient
NUM_SIMS = 25              # fewer sims, still good with 48% policy
TRAINING_STEPS = 10        # conservative to prevent forgetting
BATCH_SIZE = 64
CHECKPOINT_EVERY = 10
LR = 0.00005

config = ChessConfig(
    num_simulations=NUM_SIMS,
    games_per_iteration=GAMES_PER_ITER,
    training_steps=TRAINING_STEPS,
    batch_size=BATCH_SIZE,
    num_iterations=NUM_ITERATIONS,
    buffer_size=50000,
    lr=LR,
    temperature=1.0,
    temp_threshold=15,
    temp_final=0.05,
)

# --- Both self-play AND training on MPS ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}", flush=True)
print(f"Config: {GAMES_PER_ITER} games/iter, {NUM_SIMS} sims, "
      f"{TRAINING_STEPS} train steps, LR={LR}", flush=True)

model = ChessMuZero(
    config.observation_channels, config.hidden_channels,
    config.num_blocks, config.action_space_size,
).to(device)

if os.path.exists(RESUME_FROM):
    model.load_state_dict(torch.load(RESUME_FROM, weights_only=True, map_location=device))
    print(f"Resumed from {RESUME_FROM}", flush=True)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=config.weight_decay)
replay_buffer = ReplayBuffer(config)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print(f"Training {NUM_ITERATIONS} iterations...\n", flush=True)

for iteration in range(1, NUM_ITERATIONS + 1):
    t0 = time.time()

    # Self-play on MPS (batched inference is 15x faster than CPU at batch=32)
    model.eval()
    histories = batched_self_play(config, model, device,
                                  num_games=GAMES_PER_ITER, game_cls=ChessGame)
    for h in histories:
        replay_buffer.save_game(h)

    outcomes = {1: 0, -1: 0, 0: 0}
    total_moves = 0
    for h in histories:
        outcomes[h.outcome] += 1
        total_moves += h.length
    avg_len = total_moves / len(histories)
    t_sp = time.time() - t0

    # Training on MPS
    t1 = time.time()
    model.train()
    losses = {"total": 0, "policy": 0, "value": 0, "reward": 0}
    for _ in range(TRAINING_STEPS):
        loss = update_weights(model, optimizer, replay_buffer, config, device)
        for k in losses:
            losses[k] += loss[k]
    for k in losses:
        losses[k] /= TRAINING_STEPS
    t_tr = time.time() - t1

    games_per_sec = GAMES_PER_ITER / t_sp
    print(
        f"Iter {iteration:3d}/{NUM_ITERATIONS} | "
        f"Buf: {replay_buffer.total_games():5d} | "
        f"Loss: {losses['total']:.3f} (p:{losses['policy']:.3f} v:{losses['value']:.3f}) | "
        f"W/B/D: {outcomes[1]}/{outcomes[-1]}/{outcomes[0]} | "
        f"Avg len: {avg_len:.0f} | "
        f"sp:{t_sp:.0f}s ({games_per_sec:.1f}g/s) tr:{t_tr:.1f}s",
        flush=True,
    )

    if iteration % CHECKPOINT_EVERY == 0:
        path = os.path.join(CHECKPOINT_DIR, f"stage5_fast_iter{iteration}.pt")
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, path)
        print(f"  Saved {path}", flush=True)

final = os.path.join(CHECKPOINT_DIR, "stage5_fast_latest.pt")
torch.save({k: v.cpu() for k, v in model.state_dict().items()}, final)
print(f"\nDone. {replay_buffer.total_games()} total games. Saved {final}", flush=True)
