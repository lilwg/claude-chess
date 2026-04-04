#!/usr/bin/env python3
"""Fine-tune the 2000-Elo human-games model on soft Stockfish targets.

The model already knows chess from 38M human positions.
SF soft targets refine it toward optimal play.
Very low LR to preserve existing knowledge.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import os, time, math, random
import chess, chess.engine, chess.pgn
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from muzero.chess_game import encode_board, encode_move, NUM_ACTIONS
from muzero.chess_model import ChessMuZero
from muzero.chess_config import ChessConfig

NUM_POSITIONS = 200_000
TOP_K_MOVES = 5
SF_TIME = 0.005
SOFT_TEMP = 100.0
EPOCHS = 2          # just 2 epochs — gentle refinement
BATCH_SIZE = 256
LR = 0.00005        # very low LR to preserve human-game knowledge
RESUME = "checkpoints/stage6_supervised_6months_epoch3.pt"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}", flush=True)
print(f"Fine-tuning {RESUME} with SF soft targets", flush=True)
print(f"LR={LR} (very low to preserve existing knowledge)\n", flush=True)

# --- Generate soft targets (reuse logic from train_sf_soft.py) ---
import glob
pgn_files = sorted(glob.glob("data/lichess_2013-*.pgn"))
engine = chess.engine.SimpleEngine.popen_uci("stockfish")

samples = []
games_parsed = 0
print(f"Generating {NUM_POSITIONS} positions with top-{TOP_K_MOVES} SF evals...", flush=True)

for pgn_path in pgn_files:
    if len(samples) >= NUM_POSITIONS:
        break
    with open(pgn_path) as f:
        while len(samples) < NUM_POSITIONS:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            try:
                w_elo = int(game.headers.get("WhiteElo", "0"))
                b_elo = int(game.headers.get("BlackElo", "0"))
                if w_elo < 1400 or b_elo < 1400:
                    continue
            except ValueError:
                continue

            board = game.board()
            moves = list(game.mainline_moves())
            if len(moves) < 10:
                continue

            indices = random.sample(
                range(min(5, len(moves)), len(moves) - 1),
                min(3, len(moves) - 6)
            )
            for i, move in enumerate(moves):
                board.push(move)
                if i not in indices or board.is_game_over():
                    continue
                legal = list(board.legal_moves)
                if len(legal) < 3:
                    continue

                try:
                    info = engine.analyse(board, chess.engine.Limit(time=SF_TIME))
                    pos_score = info["score"].white()
                    pos_cp = 10000 if pos_score.is_mate() and pos_score.mate() > 0 else \
                             -10000 if pos_score.is_mate() else pos_score.score()
                    value = math.tanh(pos_cp / 400.0)
                    if board.turn == chess.BLACK:
                        value = -value
                except Exception:
                    continue

                move_scores = []
                for m in random.sample(legal, min(TOP_K_MOVES, len(legal))):
                    board.push(m)
                    try:
                        info = engine.analyse(board, chess.engine.Limit(time=SF_TIME))
                        sc = info["score"].white()
                        cp = 10000 if sc.is_mate() and sc.mate() > 0 else \
                             -10000 if sc.is_mate() else sc.score()
                        cp = -cp if board.turn == chess.WHITE else cp
                        move_scores.append((m, cp))
                    except Exception:
                        pass
                    board.pop()

                if len(move_scores) < 2:
                    continue

                turn = board.turn
                actions = [encode_move(m, turn) for m, _ in move_scores]
                scores = np.array([cp / SOFT_TEMP for _, cp in move_scores], dtype=np.float64)
                scores -= scores.max()
                probs = np.exp(scores)
                probs /= probs.sum()

                policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
                for a, p in zip(actions, probs):
                    policy[a] = p

                samples.append((board.fen(), policy, value))

            games_parsed += 1
            if games_parsed % 2000 == 0:
                print(f"  {games_parsed} games, {len(samples)} positions...", flush=True)

engine.quit()
print(f"  Done: {len(samples)} positions\n", flush=True)

# --- Dataset ---
class SoftDS(Dataset):
    def __init__(self, samples):
        self.fens = [s[0] for s in samples]
        self.policies = np.stack([s[1] for s in samples])
        self.values = np.array([s[2] for s in samples], dtype=np.float32)
    def __len__(self):
        return len(self.values)
    def __getitem__(self, idx):
        board = chess.Board(self.fens[idx])
        obs = encode_board(board)
        return torch.from_numpy(obs), torch.from_numpy(self.policies[idx]), self.values[idx]

random.shuffle(samples)
split = int(0.95 * len(samples))
train_loader = DataLoader(SoftDS(samples[:split]), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(SoftDS(samples[split:]), batch_size=BATCH_SIZE, num_workers=0)
del samples
print(f"Train: {split}, Val: {len(samples) - split if 'samples' in dir() else '~10K'}", flush=True)

# --- Model ---
config = ChessConfig()
model = ChessMuZero(config.observation_channels, config.hidden_channels,
                    config.num_blocks, config.action_space_size).to(device)
model.load_state_dict(torch.load(RESUME, weights_only=True, map_location=device))
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
print(f"Loaded {RESUME}, fine-tuning {EPOCHS} epochs\n", flush=True)

for epoch in range(1, EPOCHS + 1):
    model.train()
    t0 = time.time()
    tp, tv, tn = 0.0, 0.0, 0
    for obs, target_policy, values in train_loader:
        obs, target_policy, values = obs.to(device), target_policy.to(device), values.to(device)
        _, pl, vp = model.initial_inference(obs)
        log_probs = F.log_softmax(pl, dim=-1)
        p_loss = -(target_policy * log_probs).sum(dim=-1).mean()
        v_loss = F.mse_loss(vp.squeeze(-1), values)
        loss = p_loss + v_loss
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tp += p_loss.item() * obs.size(0)
        tv += v_loss.item() * obs.size(0)
        tn += obs.size(0)

    model.eval()
    vp2, vv2, vn2 = 0.0, 0.0, 0
    with torch.no_grad():
        for obs, target_policy, values in val_loader:
            obs, target_policy, values = obs.to(device), target_policy.to(device), values.to(device)
            _, pl, vpr = model.initial_inference(obs)
            vp2 += -(target_policy * F.log_softmax(pl, -1)).sum(-1).mean().item() * obs.size(0)
            vv2 += F.mse_loss(vpr.squeeze(-1), values).item() * obs.size(0)
            vn2 += obs.size(0)

    print(f"Epoch {epoch}/{EPOCHS} ({time.time()-t0:.0f}s) | "
          f"Train: p={tp/tn:.3f} v={tv/tn:.3f} | Val: p={vp2/vn2:.3f} v={vv2/vn2:.3f}", flush=True)

os.makedirs("checkpoints", exist_ok=True)
path = "checkpoints/stage9_finetune_sf_soft.pt"
torch.save({k: v.cpu() for k, v in model.state_dict().items()}, path)
print(f"\nSaved {path}", flush=True)
