"""
Hyperparameter configuration for Shared DQN.

All 3 tasks train together with a single shared network.
Episodes are distributed round-robin across tasks (500 episodes per task).
"""

SHARED_DQN_CONFIG = {
    # Training
    'num_episodes_per_task': 500,     # 1500 total episodes (500 × 3 tasks)
    'batch_size': 64,
    'replay_buffer_size': 100000,
    'min_replay_size': 2000,          # Start training after 2000 transitions (matches Windy/Heavy)

    # Learning
    'learning_rate': 5e-4,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,

    # Target network
    'target_update_freq': 10,         # Episodes between target network updates

    # Task-specific timeouts (same as Independent DQN for fair comparison)
    'max_episode_steps': {
        'standard': 1000,  # Easiest task, generous timeout
        'windy': 400,      # Tight timeout to prevent hovering
        'heavy': 800,      # Moderate timeout for 1.25x gravity
    },

    # Evaluation
    'eval_freq': 50,                  # Evaluate every 50 episodes
    'eval_episodes': 5,               # 5 episodes per task during evaluation
    'save_freq': 100,                 # Save checkpoint every 100 episodes

    # Architecture
    'embedding_dim': 8,               # Task embedding size (moderate capacity)
    'hidden_dims': (256, 128),        # Same as Independent DQN
}


def get_config():
    """Get a copy of the configuration dictionary."""
    return SHARED_DQN_CONFIG.copy()
