"""
Evaluation Script for Independent DQN

Test trained models and visualize their performance.
"""

import sys
import numpy as np
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from environments.lunar_lander_variants import make_env
from agents.dqn import DQNAgent
from utils.metrics import count_parameters, print_experiment_summary, analyze_experiment


def evaluate_model(task_name: str, model_path: Path, num_episodes: int = 10,
                  render: bool = False, verbose: bool = True):
    """
    Evaluate a trained DQN model.

    Args:
        task_name: Name of the task ('standard', 'windy', 'heavy')
        model_path: Path to saved model checkpoint
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        verbose: Print detailed episode information

    Returns:
        Dictionary with evaluation statistics
    """
    # Create environment
    render_mode = 'human' if render else None
    env = make_env(task_name, render_mode=render_mode)

    # Create agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)

    # Load trained model
    if model_path.exists():
        agent.load(str(model_path))
        print(f"✅ Loaded model from: {model_path}")
    else:
        print(f"❌ Model not found: {model_path}")
        return None

    # Evaluate
    episode_rewards = []
    episode_lengths = []
    success_count = 0

    print(f"\n{'='*80}")
    print(f"EVALUATING {task_name.upper()} TASK")
    print(f"{'='*80}\n")

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        truncated = False

        while not (done or truncated):
            # Select action (greedy, no exploration)
            action = agent.select_action(state, epsilon=0.0)

            # Take action
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            steps += 1

            # Safety timeout
            if steps >= 1000:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)

        if episode_reward > 0:
            success_count += 1

        if verbose:
            status = "✓ SUCCESS" if episode_reward > 0 else "✗ FAILED"
            print(f"Episode {episode+1:2d}: Reward = {episode_reward:7.2f} | Steps = {steps:3d} | {status}")

    env.close()

    # Compute statistics
    stats = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'success_rate': success_count / num_episodes
    }

    # Print summary
    print(f"\n{'='*80}")
    print(f"EVALUATION SUMMARY ({num_episodes} episodes)")
    print(f"{'='*80}")
    print(f"Mean Reward:    {stats['mean_reward']:7.2f} ± {stats['std_reward']:6.2f}")
    print(f"Reward Range:   [{stats['min_reward']:7.2f}, {stats['max_reward']:7.2f}]")
    print(f"Mean Length:    {stats['mean_length']:6.1f} ± {stats['std_length']:5.1f} steps")
    print(f"Success Rate:   {stats['success_rate']*100:5.1f}%")
    print(f"{'='*80}\n")

    return stats


def compare_models(task_names: List[str]):
    """
    Compare trained models across multiple tasks.

    Args:
        task_names: List of task names to evaluate
    """
    print(f"\n{'='*80}")
    print("MULTI-TASK MODEL COMPARISON")
    print(f"{'='*80}\n")

    results = []

    for task in task_names:
        # New folder structure: results/{task}/models/best.pth
        model_path = project_root / 'results' / task / 'models' / 'best.pth'
        print(f"\n--- Evaluating {task.upper()} ---")
        stats = evaluate_model(task, model_path, num_episodes=20, verbose=False)

        if stats:
            results.append({
                'task': task,
                'mean_reward': stats['mean_reward'],
                'success_rate': stats['success_rate']
            })

    # Print comparison table
    if results:
        print(f"\n{'='*80}")
        print(f"{'Task':<15} {'Mean Reward':<15} {'Success Rate':<15}")
        print(f"{'-'*80}")
        for r in results:
            print(f"{r['task'].capitalize():<15} {r['mean_reward']:<15.2f} {r['success_rate']*100:<15.1f}%")
        print(f"{'='*80}\n")


def analyze_training_results(task_name: str):
    """
    Analyze and print training results from metrics file.

    Args:
        task_name: Name of the task
    """
    try:
        # Count parameters (for Independent DQN: state_dim=8, action_dim=4, hidden=[256, 128])
        # fc1: 8*256 + 256 = 2304
        # fc2: 256*128 + 128 = 32896
        # fc3: 128*4 + 4 = 516
        # Total: ~35,716 parameters (per task, x2 for target network = 71,432 total)
        param_count = 35716

        analysis = analyze_experiment(task_name, 'independent_dqn', param_count)
        print_experiment_summary(analysis)

    except FileNotFoundError:
        print(f"❌ No training metrics found for {task_name} task")
        print(f"   Expected file: results/logs/independent_dqn_{task_name}_metrics.json")


if __name__ == "__main__":
    """
    Usage:
        python evaluate.py                 # Analyze Heavy training results
        python evaluate.py heavy           # Evaluate Heavy model
        python evaluate.py heavy --render  # Evaluate Heavy with visualization
        python evaluate.py --all           # Compare all tasks
    """
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate Independent DQN models')
    parser.add_argument('task', nargs='?', default='heavy',
                       help='Task to evaluate (standard, windy, heavy)')
    parser.add_argument('--render', action='store_true',
                       help='Render the environment during evaluation')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to evaluate')
    parser.add_argument('--all', action='store_true',
                       help='Compare all tasks')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze training results (no model evaluation)')

    args = parser.parse_args()

    if args.all:
        # Compare all tasks
        compare_models(['standard', 'windy', 'heavy'])
    elif args.analyze_only:
        # Just analyze training metrics
        analyze_training_results(args.task)
    else:
        # Evaluate specific task
        print("\n" + "="*80)
        print("STEP 1: ANALYZE TRAINING RESULTS")
        print("="*80)
        analyze_training_results(args.task)

        print("\n" + "="*80)
        print("STEP 2: EVALUATE TRAINED MODEL")
        print("="*80)
        # New folder structure: results/{task}/models/best.pth
        model_path = project_root / 'results' / args.task / 'models' / 'best.pth'
        evaluate_model(args.task, model_path, num_episodes=args.episodes, render=args.render)
