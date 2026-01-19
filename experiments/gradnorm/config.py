"""
Hyperparameter configuration for GradNorm DQN.

Same base hyperparameters as Shared DQN for fair comparison.
GradNorm adds learnable task weights with dynamic balancing.
"""

GRADNORM_CONFIG = {
    # Training
    'num_episodes_per_task': 500,     # 1500 total episodes (500 x 3 tasks)
    'batch_size': 64,
    'replay_buffer_size': 100000,     # Total across all tasks (~33k per task)
    'min_replay_size': 2000,          # Per-task minimum before training

    # Learning
    'learning_rate': 5e-4,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,

    # Target network
    'target_update_freq': 10,         # Episodes between target network updates

    # Task-specific timeouts (same as Shared DQN for fair comparison)
    'max_episode_steps': {
        'standard': 1000,  # Easiest task, generous timeout
        'windy': 400,      # Tight timeout to prevent hovering
        'heavy': 800,      # Moderate timeout for 1.25x gravity
    },

    # Evaluation
    'eval_freq': 50,                  # Evaluate every 50 episodes
    'eval_episodes': 5,               # 5 episodes per task during evaluation
    'save_freq': 100,                 # Save checkpoint every 100 episodes

    # Architecture (same as Shared DQN)
    'embedding_dim': 8,               # Task embedding size
    'hidden_dims': (256, 128),        # Same as Independent DQN
    'use_task_embedding': False,      # Task-blind mode

    # GradNorm-specific
    'gradnorm_alpha': 1.5,            # Asymmetry parameter (recommended: 1.5)
    'gradnorm_lr': 0.01,              # Learning rate for task weights
    'weight_log_freq': 10,            # Log weight stats every 10 episodes

    # Output directory
    'output_dir': 'gradnorm_blind',
}


def get_config():
    """Get a copy of the configuration dictionary."""
    return GRADNORM_CONFIG.copy()
