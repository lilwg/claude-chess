#!/usr/bin/env python3
"""Distill Stockfish into the chess model.

For each position from Lichess games:
  - Policy target: Stockfish's best move (much stronger than human moves)
  - Value target: Stockfish evaluation (much more accurate than game outcome)

This is far more data-efficient than self-play or human game imitation.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import os, glob, time, random, math
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
PGN_PATTERN = "data/lichess_2013-*.pgn"
MAX_POSITIONS = 500_000     # how many positions to evaluate with Stockfish
MIN_ELO = 1400              # source games (just for diverse positions)
STOCKFISH_TIME = 0.01       # seconds per position (10ms = ~depth 12-15)
STOCKFISH_PATH = "stockfish"
EPOCHS = 5
BATCH_SIZE = 256
LR = 0.001
RESUME_FROM = None


def cp_to_value(cp):
    """Convert centipawn score to [-1, 1] value using tanh scaling."""
    return math.tanh(cp / 400.0)


def generate_stockfish_targets(pgn_files, max_positions, min_elo, sf_time):
    """Extract positions from games, evaluate each with Stockfish."""
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    # Collect diverse positions from human games
    positions = []  # (board_fen, sf_move_action, sf_value)
    games_parsed = 0

    for pgn_path in pgn_files:
        print(f"  Reading {os.path.basename(pgn_path)}...", flush=True)
        with open(pgn_path) as f:
            while len(positions) < max_positions:
                game = chess.pgn.read_game(f)
                if game is None:
                    break

                try:
                    w_elo = int(game.headers.get("WhiteElo", "0"))
                    b_elo = int(game.headers.get("BlackElo", "0"))
                    if w_elo < min_elo or b_elo < min_elo:
                        continue
                except ValueError:
                    continue

                # Sample ~4 random positions per game (not every position)
                board = game.board()
                moves = list(game.mainline_moves())
                if len(moves) < 10:
                    continue

                sample_indices = random.sample(
                    range(min(5, len(moves)), len(moves) - 1),
                    min(4, len(moves) - 6)
                )

                for i, move in enumerate(moves):
                    board.push(move)
                    if i not in sample_indices:
                        continue
                    if board.is_game_over():
                        continue

                    # Get Stockfish evaluation
                    try:
                        result = engine.analyse(
                            board, chess.engine.Limit(time=sf_time),
                            info=chess.engine.INFO_ALL
                        )
                        sf_move = result.get("pv", [None])[0]
                        score = result["score"].white()

                        if sf_move is None or sf_move not in board.legal_moves:
                            continue

                        # Convert score to value from current player's perspective
                        if score.is_mate():
                            mate_in = score.mate()
                            val = 1.0 if mate_in > 0 else -1.0
                        else:
                            cp = score.score()
                            val = cp_to_value(cp)

                        # Flip value for black's perspective
                        if board.turn == chess.BLACK:
                            val = -val

                        action = encode_move(sf_move, board.turn)
                        positions.append((board.fen(), action, val))
                    except Exception:
                        continue

                games_parsed += 1
                if games_parsed % 1000 == 0:
                    print(f"    {games_parsed} games, {len(positions)} positions...",
                          flush=True)

        if len(positions) >= max_positions:
            break

    engine.quit()
    print(f"  Done: {len(positions)} positions from {games_parsed} games", flush=True)
    return positions


class StockfishDataset(Dataset):
    """On-the-fly FEN encoding with Stockfish targets."""
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
    pgn_files = sorted(glob.glob(PGN_PATTERN))
    print(f"Device: {device}, {len(pgn_files)} PGN files", flush=True)
    print(f"Generating {MAX_POSITIONS} Stockfish-evaluated positions "
          f"(depth ~12-15, {STOCKFISH_TIME}s/pos)...", flush=True)

    t0 = time.time()
    samples = generate_stockfish_targets(pgn_files, MAX_POSITIONS, MIN_ELO, STOCKFISH_TIME)
    gen_time = time.time() - t0
    print(f"Generation took {gen_time:.0f}s ({len(samples)/gen_time:.0f} pos/sec)\n", flush=True)

    random.shuffle(samples)
    split = int(0.95 * len(samples))
    train_ds = StockfishDataset(samples[:split])
    val_ds = StockfishDataset(samples[split:])
    del samples

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}", flush=True)

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
    print(f"Model: {params:,} params, {EPOCHS} epochs\n", flush=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        tp, tv, tn, tc = 0.0, 0.0, 0, 0
        for obs, actions, values in train_loader:
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
        path = f"checkpoints/stage6_distill_epoch{epoch}.pt"
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, path)
        print(f"  Saved {path}", flush=True)

    final = "checkpoints/stage6_distill.pt"
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, final)
    print(f"\nDone. Saved {final}", flush=True)


if __name__ == "__main__":
    main()
