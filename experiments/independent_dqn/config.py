"""
Hyperparameter configurations for Independent DQN experiments.

Each task variant has its own tuned configuration based on task difficulty.
"""

# Standard Task Configuration (Baseline)
STANDARD_CONFIG = {
    'task_name': 'standard',
    'num_episodes': 1500,  # Matched to Windy/Heavy for fair comparison
    'batch_size': 64,
    'replay_buffer_size': 100000,
    'min_replay_size': 1000,
    'learning_rate': 5e-4,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    'target_update_freq': 10,
    'max_episode_steps': 1000,
    'eval_freq': 50,
    'eval_episodes': 5,
    'save_freq': 100,
}

# Windy Task Configuration
# - Heavy-style tuning: lower LR, slower epsilon decay, less frequent target updates
# - Reduced timeout to 400 (forces landing, prevents hovering)
# - Increased episodes (harder task needs more training)
WINDY_CONFIG = {
    **STANDARD_CONFIG,
    'task_name': 'windy',
    'num_episodes': 1500,           # Increased from 1000 (harder task needs more training)
    'min_replay_size': 2000,        # Increased from 1000 (more buffer for stability)
    'learning_rate': 2.5e-4,        # Reduced from 5e-4 (halved for stability)
    'epsilon_decay': 0.992,         # Reduced from 0.995 (slower decay, more exploration)
    'target_update_freq': 20,       # Increased from 10 (less frequent for stability)
    'max_episode_steps': 400,       # Reduced from 800 (forces landing urgency)
}

# Heavy Weight Task Configuration
# - Increased episodes (harder task needs more training)
# - Lower learning rate for stability
# - Slower epsilon decay for more exploration
# - Less frequent target updates to reduce Q-value instability
# - Larger initial replay buffer for better data diversity
# - Increased timeout to 800 (allows full descent with 1.25x gravity)
HEAVY_CONFIG = {
    **STANDARD_CONFIG,
    'task_name': 'heavy',
    'num_episodes': 1500,
    'min_replay_size': 2000,
    'learning_rate': 2.5e-4,
    'epsilon_decay': 0.992,
    'target_update_freq': 20,
    'max_episode_steps': 800,
}

# Configuration lookup dictionary
CONFIGS = {
    'standard': STANDARD_CONFIG,
    'windy': WINDY_CONFIG,
    'heavy': HEAVY_CONFIG,
}


def get_config(task_name):
    """
    Get configuration for a specific task.

    Args:
        task_name (str): One of 'standard', 'windy', 'heavy'

    Returns:
        dict: Configuration dictionary for the task
    """
    if task_name not in CONFIGS:
        raise ValueError(f"Unknown task: {task_name}. Choose from: {list(CONFIGS.keys())}")
    return CONFIGS[task_name].copy()
