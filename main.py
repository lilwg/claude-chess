#!/usr/bin/env python3
"""Entry point for MuZero Tic-Tac-Toe training."""

import sys

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "play":
        from muzero.main import play_interactive
        play_interactive()
    else:
        from muzero.main import train
        train()
