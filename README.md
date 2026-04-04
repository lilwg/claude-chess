# MuZero Chess

A from-scratch implementation of [MuZero](https://arxiv.org/abs/1911.08265) and [AlphaZero](https://arxiv.org/abs/1712.01815) for chess, trained entirely on a Mac M3.

Started from zero — no chess knowledge, no pretrained weights — and reached **~2000 Elo** (beats Stockfish at calibrated 2000-level) through supervised learning on Lichess human games.

## Play it

### Web interface (recommended)
```bash
cd /Users/pat/claude/chess
python3 web/server.py
# Open http://localhost:5000
```
Drag pieces or click-click to move. Eval bar shows the model's assessment.

### Terminal
```bash
python3 chess_main.py play checkpoints/stage6_38M_epoch3_elo2000.pt
```
Enter moves in UCI format (`e2e4`, `g1f3`, `e7e8q`).

## Checkpoints

All in `checkpoints/`. Elo measured against Stockfish (UCI_LimitStrength) with 50-sim AlphaZero MCTS.

| Checkpoint | What | Elo |
|---|---|---|
| `tictactoe_muzero_100iter.pt` | Tic-tac-toe MuZero (plays perfectly) | N/A |
| `stage1_supervised_10k_games.pt` | Old arch (13.3M, FC policy head), 10K games | ~980 |
| `stage3_spatial_5k_elo1380.pt` | Spatial policy head (6.1M), 5K games | ~1380 |
| `stage3_full_90k_REAL_elo1950.pt` | Spatial, 90K games | ~1950 |
| `stage6_38M_epoch1_elo1500.pt` | Spatial, 38M positions, epoch 1 | ~1500 |
| `stage6_38M_epoch2_elo1900.pt` | Spatial, 38M positions, epoch 2 | ~1900 |
| **`stage6_38M_epoch3_elo2000.pt`** | **Spatial, 38M positions, epoch 3** | **~2000** |
| `stage6_distill_500k_REAL_elo1550.pt` | Stockfish distillation, 500K pos | ~1550 |
| `stage7_sf_5M_epoch3.pt` | Stockfish distillation, 5M pos | ~1200 |
| `stage9_finetune_sf_soft.pt` | 2000-Elo fine-tuned on SF soft targets (degraded — too drawish) | ~1400 |
| `stage4_selfplay_50iter_DEGRADED.pt` | MuZero self-play (catastrophic forgetting) | ~920 |

## How we got to 2000 Elo

| Change | Elo gain | Lesson |
|---|---|---|
| First model: 10K games, FC policy head | ~980 | Starting point |
| **Spatial policy head** (Conv2d 128→73 instead of FC) | **+400** | 72% of params were wasted on the FC layer |
| More data: 5K → 90K games | +570 | Data volume dominates |
| More data: 90K → 38M positions (6 months of Lichess) | +50 | Diminishing returns, but still helps |
| **AlphaZero MCTS** (real board) instead of MuZero MCTS (learned dynamics) | **+500** | Untrained dynamics model was sabotaging search |
| MuZero self-play (1,600 games) | **-600** | Catastrophic forgetting — not enough self-play data |
| Stockfish distillation (5M positions) | ~1200 | Too hard to predict SF's exact move at this model size |
| SF soft targets fine-tuning | ~1400 | Made model drawish — lost ability to win against weaker players |

### Key lessons
- **Architecture > compute > data engineering.** The spatial policy head was the single biggest improvement (+400 Elo) and cost zero extra compute.
- **Real-board MCTS is critical.** MuZero's learned dynamics model needs massive self-play to be accurate. Without it, use real game rules for MCTS (AlphaZero-style).
- **Human games > Stockfish distillation** at this scale. Predicting human moves teaches general chess patterns. Predicting Stockfish's single best move is too hard for a 6M-param model.
- **Self-play needs 10,000x more games than we generated.** AlphaZero used 44M self-play games. Our 1,600 games just added noise. On consumer hardware, supervised learning on freely available human games is far more practical.
- **More data helps, then plateaus.** 5K→90K games: huge. 90K→38M positions: modest. The model capacity (6.1M params) limits how much data it can absorb.

## Architecture

**ResNet with spatial policy head, 6.1M parameters:**
- Input: 19 planes × 8×8 (piece positions, castling, en passant, 50-move counter)
- Backbone: Conv2d(19→128) + 10 ResBlocks (each: Conv-BN-ReLU-Conv-BN + skip)
- Policy head: Conv2d(128→73) → reshape to 4,672 actions (64 squares × 73 move types)
- Value head: Conv2d(128→1) → FC(64→128→1) → tanh

Move encoding follows AlphaZero: 8×8×73 = 4,672 actions. The 73 planes cover queen moves (56), knight moves (8), and underpromotions (9).

## Training

### Supervised pretraining on Lichess games
```bash
# Download Lichess data
curl -L -o data/lichess_2013-01.pgn.zst https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst

# Decompress
python3 -c "import zstandard; dctx=zstandard.ZstdDecompressor(); open('data/lichess_2013-01.pgn','wb').write(dctx.decompress(open('data/lichess_2013-01.pgn.zst','rb').read()))"

# Train (edit train_supervised.py for config)
python3 train_supervised.py
```

### Stockfish distillation
```bash
# From Lichess eval database (362M SF-evaluated positions, streamed)
python3 train_distill_lichess_evals.py

# Local SF evaluation with soft targets
python3 train_sf_soft.py
```

### MuZero self-play (experimental — did not improve on supervised)
```bash
python3 train_selfplay.py
```

### Evaluate against Stockfish
```bash
brew install stockfish
python3 eval_alphazero.py checkpoints/stage6_38M_epoch3_elo2000.pt
```

## Project structure

```
muzero/
  config.py              Tic-tac-toe config
  game.py                Tic-tac-toe environment + minimax
  model.py               MLP model for tic-tac-toe (78K params)
  chess_config.py        Chess hyperparameters
  chess_game.py          Chess env + AlphaZero move encoding (4,672 actions)
  chess_model.py         ResNet with spatial policy head (6.1M params)
  alphazero_model.py     AlphaZero variant with WDL value head (3M params)
  mcts.py                MuZero MCTS (numpy-vectorized UCB)
  alphazero_mcts.py      AlphaZero MCTS (real board states, batched)
  batched.py             Batched MuZero MCTS (multiple games in parallel)
  replay_buffer.py       Game storage + MuZero training targets
  trainer.py             Self-play + weight updates
  supervised.py          Lichess PGN parsing + supervised pretraining
  eval_elo.py            Elo measurement vs Stockfish (MuZero MCTS)
web/
  server.py              Flask backend for web interface
  templates/index.html   Chess.com-style web UI
train_supervised.py      Supervised pretraining script
train_selfplay.py        MuZero self-play training
train_distill.py         Local Stockfish distillation
train_distill_lichess_evals.py  Lichess SF eval database training
train_sf_soft.py         Soft Stockfish targets training
train_alphazero.py       AlphaZero WDL model training
chess_main.py            Chess entry point (train + play)
main.py                  Tic-tac-toe entry point
eval_alphazero.py        Elo measurement with AlphaZero MCTS
data/                    Lichess PGN files (not in git)
checkpoints/             Model weights (not in git)
```

## Dependencies

```
pip install torch numpy chess zstandard flask datasets
brew install stockfish  # for Elo evaluation
```

Requires Python 3.10+ and PyTorch 2.0+. MPS (Apple Silicon GPU) recommended for training.
