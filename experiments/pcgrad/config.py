"""
Hyperparameter configuration for PCGrad DQN.

Same hyperparameters as Shared DQN for fair comparison.
PCGrad uses separate per-task replay buffers and gradient projection.
"""

PCGRAD_CONFIG = {
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
    'use_task_embedding': False,      # Task-blind experiment

    # PCGrad-specific
    'gradient_log_freq': 10,          # Log conflict stats every 10 episodes

    # Output
    'output_dir': 'pcgrad_blind',     # Task-blind output directory
}


def get_config():
    """Get a copy of the configuration dictionary."""
    return PCGRAD_CONFIG.copy()
