from dataclasses import dataclass


@dataclass
class ChessConfig:
    # Game
    observation_channels: int = 19
    action_space_size: int = 4672

    # Model
    hidden_channels: int = 128
    num_blocks: int = 10

    # MCTS
    num_simulations: int = 200
    max_children: int = 32     # top-K expansion at internal nodes
    discount: float = 1.0
    dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25
    pb_c_base: float = 19652
    pb_c_init: float = 1.25

    # Self-play temperature
    temperature: float = 1.0
    temp_threshold: int = 30
    temp_final: float = 0.1
    max_moves: int = 512

    # Training
    num_iterations: int = 100
    games_per_iteration: int = 10
    training_steps: int = 100
    batch_size: int = 128
    unroll_steps: int = 5
    lr: float = 0.002
    weight_decay: float = 1e-4

    # Replay buffer
    buffer_size: int = 10000

    # Evaluation
    eval_episodes: int = 10
    eval_interval: int = 10
