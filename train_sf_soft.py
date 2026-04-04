#!/usr/bin/env python3
"""Stockfish distillation with SOFT policy targets.

Instead of "only SF's best move is correct", evaluate the top 5 legal moves
with Stockfish and create a probability distribution weighted by their scores.

e.g., if SF says: e4=+50cp, d4=+45cp, Nf3=+30cp, c4=+25cp, g3=+10cp
target = softmax([50, 45, 30, 25, 10] / temperature)
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import os, time, math, random
import chess
import chess.engine
import chess.pgn
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from muzero.chess_game import encode_board, encode_move, NUM_ACTIONS
from muzero.chess_model import ChessMuZero
from muzero.chess_config import ChessConfig

# --- Config ---
NUM_POSITIONS = 200_000     # fewer positions but MUCH richer targets
TOP_K_MOVES = 5             # evaluate top 5 moves per position
SF_TIME = 0.005             # 5ms per move eval (fast, ~depth 10)
SOFT_TEMP = 100.0           # temperature for converting cp to probabilities
EPOCHS = 5
BATCH_SIZE = 256
LR = 0.001
STOCKFISH_PATH = "stockfish"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def generate_soft_targets(pgn_files, num_positions, sf_time, top_k):
    """For each position, evaluate top-K legal moves with Stockfish.
    
    Returns: list of (fen, soft_policy_array, value)
    """
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    
    samples = []
    games_parsed = 0
    
    for pgn_path in pgn_files:
        print(f"  Reading {os.path.basename(pgn_path)}...", flush=True)
        with open(pgn_path) as f:
            while len(samples) < num_positions:
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
                
                # Sample 3 positions per game
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
                    
                    # Get SF's evaluation of the position (for value target)
                    try:
                        info = engine.analyse(board, chess.engine.Limit(time=sf_time))
                        pos_score = info["score"].white()
                        if pos_score.is_mate():
                            pos_cp = 10000 if pos_score.mate() > 0 else -10000
                        else:
                            pos_cp = pos_score.score()
                        value = math.tanh(pos_cp / 400.0)
                        if board.turn == chess.BLACK:
                            value = -value
                    except Exception:
                        continue
                    
                    # Evaluate top-K moves by doing a quick search on each
                    move_scores = []
                    for m in random.sample(legal, min(top_k, len(legal))):
                        board.push(m)
                        try:
                            info = engine.analyse(board, chess.engine.Limit(time=sf_time))
                            sc = info["score"].white()
                            if sc.is_mate():
                                cp = 10000 if sc.mate() > 0 else -10000
                            else:
                                cp = sc.score()
                            # From the mover's perspective (before push, board.turn was the mover)
                            # After push, we're looking at opponent's eval
                            # Negate to get mover's perspective
                            cp = -cp if board.turn == chess.WHITE else cp
                            move_scores.append((m, cp))
                        except Exception:
                            pass
                        board.pop()
                    
                    if len(move_scores) < 2:
                        continue
                    
                    # Create soft target: softmax over centipawn scores
                    actions = []
                    scores = []
                    turn = board.turn
                    for m, cp in move_scores:
                        actions.append(encode_move(m, turn))
                        scores.append(cp / SOFT_TEMP)
                    
                    # Softmax
                    scores = np.array(scores, dtype=np.float64)
                    scores -= scores.max()  # numerical stability
                    probs = np.exp(scores)
                    probs /= probs.sum()
                    
                    # Build sparse policy target
                    policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
                    for a, p in zip(actions, probs):
                        policy[a] = p
                    
                    samples.append((board.fen(), policy, value))
                
                games_parsed += 1
                if games_parsed % 2000 == 0:
                    print(f"    {games_parsed} games, {len(samples)} positions...", flush=True)
        
        if len(samples) >= num_positions:
            break
    
    engine.quit()
    print(f"  Done: {len(samples)} positions from {games_parsed} games", flush=True)
    return samples


class SoftTargetDataset(Dataset):
    def __init__(self, samples):
        self.fens = [s[0] for s in samples]
        self.policies = np.stack([s[1] for s in samples])  # (N, 4672)
        self.values = np.array([s[2] for s in samples], dtype=np.float32)
    
    def __len__(self):
        return len(self.values)
    
    def __getitem__(self, idx):
        board = chess.Board(self.fens[idx])
        obs = encode_board(board)
        return (torch.from_numpy(obs), 
                torch.from_numpy(self.policies[idx]),
                self.values[idx])


def main():
    import glob
    pgn_files = sorted(glob.glob("data/lichess_2013-*.pgn"))
    print(f"Device: {device}, {len(pgn_files)} PGN files", flush=True)
    print(f"Generating {NUM_POSITIONS} positions with top-{TOP_K_MOVES} SF evals...", flush=True)
    
    t0 = time.time()
    samples = generate_soft_targets(pgn_files, NUM_POSITIONS, SF_TIME, TOP_K_MOVES)
    print(f"Generation: {time.time()-t0:.0f}s ({len(samples)/(time.time()-t0):.0f} pos/sec)\n", flush=True)
    
    random.shuffle(samples)
    split = int(0.95 * len(samples))
    train_ds = SoftTargetDataset(samples[:split])
    val_ds = SoftTargetDataset(samples[split:])
    del samples
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}", flush=True)
    
    config = ChessConfig()
    model = ChessMuZero(config.observation_channels, config.hidden_channels,
                        config.num_blocks, config.action_space_size).to(device)
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    total_steps = EPOCHS * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    print(f"Model: {params:,} params, {EPOCHS} epochs", flush=True)
    print(f"Soft targets: top-{TOP_K_MOVES} moves, temp={SOFT_TEMP}\n", flush=True)
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        tp, tv, tn = 0.0, 0.0, 0
        
        for obs, target_policy, values in train_loader:
            obs = obs.to(device)
            target_policy = target_policy.to(device)
            values = values.to(device)
            
            _, policy_logits, value_pred = model.initial_inference(obs)
            
            # Soft cross-entropy: -sum(target * log_softmax(logits))
            log_probs = F.log_softmax(policy_logits, dim=-1)
            p_loss = -(target_policy * log_probs).sum(dim=-1).mean()
            v_loss = F.mse_loss(value_pred.squeeze(-1), values)
            loss = p_loss + v_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            tp += p_loss.item() * obs.size(0)
            tv += v_loss.item() * obs.size(0)
            tn += obs.size(0)
        
        # Validation
        model.eval()
        vp, vv, vn = 0.0, 0.0, 0
        with torch.no_grad():
            for obs, target_policy, values in val_loader:
                obs, target_policy, values = obs.to(device), target_policy.to(device), values.to(device)
                _, pl, vpr = model.initial_inference(obs)
                log_p = F.log_softmax(pl, dim=-1)
                vp += -(target_policy * log_p).sum(-1).mean().item() * obs.size(0)
                vv += F.mse_loss(vpr.squeeze(-1), values).item() * obs.size(0)
                vn += obs.size(0)
        
        print(f"Epoch {epoch}/{EPOCHS} ({time.time()-t0:.0f}s) | "
              f"Train: p={tp/tn:.3f} v={tv/tn:.3f} | "
              f"Val: p={vp/vn:.3f} v={vv/vn:.3f}", flush=True)
        
        os.makedirs("checkpoints", exist_ok=True)
        path = f"checkpoints/stage8_sf_soft_epoch{epoch}.pt"
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, path)
        print(f"  Saved {path}", flush=True)
    
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
