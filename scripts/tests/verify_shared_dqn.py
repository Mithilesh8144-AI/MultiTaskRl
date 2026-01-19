"""
Quick verification script to test Shared DQN agent on each task separately.
This will help verify that the agent is actually being evaluated on the correct environments.
"""

import sys
import numpy as np
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from environments.lunar_lander_variants import make_env
from agents.shared_dqn import SharedDQNAgent


def verify_task(agent, task_name, task_id, num_episodes=20):
    """
    Manually verify agent performance on a specific task.

    Args:
        agent: SharedDQNAgent
        task_name: 'standard', 'windy', or 'heavy'
        task_id: 0, 1, or 2
        num_episodes: Number of episodes to evaluate
    """
    print(f"\n{'='*70}")
    print(f"VERIFYING: {task_name.upper()} Task (task_id={task_id})")
    print(f"{'='*70}")

    # Create environment
    env = make_env(task_name)

    # Print environment details to verify it's the right one
    print(f"Environment class: {env.__class__.__name__}")
    if hasattr(env, 'task_name'):
        print(f"Task name attribute: {env.task_name}")
    if hasattr(env, 'wind_power'):
        print(f"Wind power: {env.wind_power}")
    if hasattr(env, 'gravity_multiplier'):
        print(f"Gravity multiplier: {env.gravity_multiplier}")

    # Set task-specific timeouts (matching training config)
    max_steps = {
        'standard': 1000,
        'windy': 400,
        'heavy': 800,
    }[task_name]
    print(f"Max steps: {max_steps}")

    rewards = []
    lengths = []
    successes = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        truncated = False

        while not (done or truncated):
            # Greedy action selection
            action = agent.select_action(state, task_id, epsilon=0.0)
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            steps += 1

            if steps >= max_steps:
                truncated = True
                break

        rewards.append(episode_reward)
        lengths.append(steps)
        successes.append(1 if episode_reward >= 200 else 0)

        if (episode + 1) % 5 == 0:
            print(f"  Episode {episode + 1}/{num_episodes}: "
                  f"Reward={episode_reward:.2f}, Steps={steps}")

    env.close()

    # Print summary statistics
    print(f"\n{'='*70}")
    print(f"RESULTS FOR {task_name.upper()} TASK:")
    print(f"  Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Reward Range: [{np.min(rewards):.2f}, {np.max(rewards):.2f}]")
    print(f"  Mean Steps: {np.mean(lengths):.1f}")
    print(f"  Success Rate: {np.mean(successes)*100:.0f}% ({np.sum(successes)}/{num_episodes})")
    print(f"{'='*70}\n")

    return {
        'task': task_name,
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_steps': np.mean(lengths),
        'success_rate': np.mean(successes),
        'all_rewards': rewards
    }


def main():
    """Load Shared DQN model and verify on each task."""
    print("\n" + "="*70)
    print("SHARED DQN VERIFICATION")
    print("Testing agent on each environment separately to verify correctness")
    print("="*70)

    # Load the trained agent
    model_path = project_root / 'results' / 'shared_dqn' / 'models' / 'best.pth'

    if not model_path.exists():
        print(f"\n❌ ERROR: Model not found at {model_path}")
        print("Please train Shared DQN first using: python -m experiments.shared_dqn.train")
        return

    print(f"\n📂 Loading model from: {model_path}")

    # Create agent (same architecture as training)
    state_dim = 8
    action_dim = 4
    num_tasks = 3
    embedding_dim = 8
    hidden_dims = (256, 128)

    agent = SharedDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        num_tasks=num_tasks,
        embedding_dim=embedding_dim,
        hidden_dims=hidden_dims,
        learning_rate=5e-4,
        gamma=0.99
    )

    # Load trained weights
    checkpoint = torch.load(model_path, weights_only=False)

    # Handle both formats: direct state dict or nested checkpoint
    if 'q_network' in checkpoint:
        # Nested format (full agent checkpoint)
        agent.q_network.load_state_dict(checkpoint['q_network'])
        agent.target_network.load_state_dict(checkpoint['target_network'])
    else:
        # Direct state dict format
        agent.q_network.load_state_dict(checkpoint)
        agent.target_network.load_state_dict(checkpoint)

    agent.q_network.eval()  # Set to evaluation mode

    print(f"✅ Model loaded successfully")
    print(f"   Parameters: {sum(p.numel() for p in agent.q_network.parameters()):,}")

    # Task mapping (same as training)
    tasks = [
        ('standard', 0),
        ('windy', 1),
        ('heavy', 2),
    ]

    # Verify each task
    results = {}
    for task_name, task_id in tasks:
        results[task_name] = verify_task(agent, task_name, task_id, num_episodes=20)

    # Final comparison
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"{'Task':<12} {'Mean Reward':<15} {'Success Rate':<15} {'Mean Steps':<12}")
    print("-" * 70)
    for task_name in ['standard', 'windy', 'heavy']:
        r = results[task_name]
        print(f"{task_name:<12} {r['mean_reward']:>6.2f} ± {r['std_reward']:<5.2f}   "
              f"{r['success_rate']*100:>4.0f}% ({int(r['success_rate']*20)}/20)      "
              f"{r['mean_steps']:>6.1f}")
    print("="*70)

    # Compare with expected results from analyze_results.py
    print("\n📊 COMPARISON WITH ANALYSIS RESULTS:")
    expected = {
        'standard': 253.62,
        'windy': 151.20,
        'heavy': 189.51
    }

    print(f"{'Task':<12} {'Actual':<12} {'Expected':<12} {'Difference':<12}")
    print("-" * 70)
    for task_name in ['standard', 'windy', 'heavy']:
        actual = results[task_name]['mean_reward']
        exp = expected[task_name]
        diff = actual - exp
        print(f"{task_name:<12} {actual:>6.2f}       {exp:>6.2f}       {diff:>+6.2f}")
    print("="*70)


if __name__ == "__main__":
    main()
