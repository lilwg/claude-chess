from dataclasses import dataclass


@dataclass
class MuZeroConfig:
    # Game
    observation_size: int = 9
    action_space_size: int = 9

    # Model
    hidden_size: int = 128

    # MCTS
    num_simulations: int = 50
    discount: float = 1.0
    dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25
    pb_c_base: float = 19652
    pb_c_init: float = 1.25

    # Self-play temperature
    temperature: float = 1.0
    temp_threshold: int = 4  # after this many moves, use low temp
    temp_final: float = 0.1

    # Training
    num_iterations: int = 100
    games_per_iteration: int = 25
    training_steps: int = 50
    batch_size: int = 64
    unroll_steps: int = 5
    lr: float = 0.001
    weight_decay: float = 1e-4

    # Replay buffer
    buffer_size: int = 2000

    # Evaluation
    eval_episodes: int = 20
    eval_interval: int = 10
