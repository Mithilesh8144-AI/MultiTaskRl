"""
Configuration for BRC (Bigger, Regularized, Categorical) Multi-Task Training

BRC uses larger networks with residual connections and categorical DQN loss
to handle gradient conflicts in multi-task learning.
"""


def get_config():
    """
    Get BRC hyperparameters for multi-task training.

    Key differences from Shared DQN:
    - Larger network (256 hidden dim with 3 residual blocks)
    - Categorical DQN (51 atoms, cross-entropy loss)
    - Weight decay for regularization (1e-4)
    - Slightly lower learning rate for stability (3e-4 vs 5e-4)

    Returns:
        dict: Configuration dictionary
    """
    config = {
        # Training
        'num_episodes_per_task': 500,      # 500 episodes per task × 3 tasks = 1500 total
        'batch_size': 64,
        'replay_buffer_size': 100000,
        'min_replay_size': 2000,           # Match Shared DQN (stability for harder tasks)

        # Network architecture (BroNet)
        'hidden_dim': 256,                 # Can increase to 512 for more capacity
        'num_blocks': 3,                   # Number of residual blocks
        'embedding_dim': 32,               # Task embedding size (larger than Shared DQN's 8)

        # Categorical DQN parameters
        'num_atoms': 51,                   # Standard C51
        'v_min': -100.0,                   # Minimum return (adjust based on env)
        'v_max': 300.0,                    # Maximum return (successful landing ~200-300)

        # Optimization
        'learning_rate': 3e-4,             # Slightly lower than Shared DQN for stability
        'weight_decay': 1e-4,              # L2 regularization (the "Regularized" part)
        'gamma': 0.99,

        # Exploration
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,            # Exponential decay per episode

        # Target network
        'target_update_freq': 10,          # Update target every 10 episodes

        # Evaluation
        'eval_freq': 50,                   # Evaluate every 50 episodes
        'eval_episodes': 5,                # 5 episodes per task during training
        'save_freq': 100,                  # Save checkpoint every 100 episodes

        # Task-specific episode timeouts
        'max_episode_steps': {
            'standard': 1000,              # Episodes naturally finish in 100-300 steps
            'windy': 400,                  # Force landing urgency (prevent hovering)
            'heavy': 800,                  # Allow full descent with 1.25× gravity
        },

        # Device
        'device': 'cpu',                   # Use 'cuda' if GPU available
    }

    return config


# Parameter count estimation
def estimate_parameters():
    """
    Estimate BRC network parameters.

    BroNet architecture:
    - Task embedding: 3 tasks × 32 dim = 96
    - Input layer: (8 + 32) × 256 + 256 = 10,496
    - Residual blocks (×3):
      - LayerNorm: 256 × 2 = 512 (per block)
      - FC1: 256 × 256 + 256 = 65,792 (per block)
      - FC2: 256 × 256 + 256 = 65,792 (per block)
      - Total per block: 132,096
      - Total 3 blocks: 396,288
    - Final LayerNorm: 512
    - Output layer: 256 × (4 actions × 51 atoms) + (4 × 51) = 256 × 204 + 204 = 52,428

    Total: ~459,820 parameters

    Comparison:
    - Independent DQN: 35,716 × 3 = 107,148 parameters
    - Shared DQN: 37,788 parameters
    - BRC: ~459,820 parameters (4.3× Independent, 12.2× Shared)

    The increased capacity is the "Bigger" part of BRC.
    """
    task_embedding = 3 * 32
    input_layer = (8 + 32) * 256 + 256
    res_blocks = 3 * (512 + 2 * (256 * 256 + 256))
    final_ln = 512
    output_layer = 256 * (4 * 51) + (4 * 51)

    total = task_embedding + input_layer + res_blocks + final_ln + output_layer

    print(f"\n📊 BRC Parameter Breakdown:")
    print(f"   Task embedding: {task_embedding:,}")
    print(f"   Input layer: {input_layer:,}")
    print(f"   Residual blocks (×3): {res_blocks:,}")
    print(f"   Final LayerNorm: {final_ln:,}")
    print(f"   Output layer: {output_layer:,}")
    print(f"   {'='*40}")
    print(f"   TOTAL: {total:,} parameters")
    print(f"\n   vs Independent DQN: {total / 107148:.1f}×")
    print(f"   vs Shared DQN: {total / 37788:.1f}×\n")

    return total


if __name__ == "__main__":
    # Print configuration
    config = get_config()

    print("\n" + "="*70)
    print("BRC CONFIGURATION")
    print("="*70)

    print("\n📋 Training Configuration:")
    for key, value in config.items():
        if key != 'max_episode_steps':
            print(f"   {key}: {value}")
        else:
            print(f"   {key}:")
            for task, steps in value.items():
                print(f"      {task}: {steps}")

    print("\n" + "="*70)

    # Estimate parameters
    estimate_parameters()
