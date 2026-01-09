"""
BRC Evaluation Script

Evaluate a trained BRC agent on all 3 Lunar Lander task variants.
Compares performance across tasks to analyze multi-task learning effectiveness.
"""

import os
import sys
import numpy as np
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from environments.lunar_lander_variants import make_env
from agents.brc import BRCAgent
from config import get_config


def evaluate_task(agent, task_name, task_id, config, num_episodes=20, render=False):
    """
    Evaluate agent on a single task.

    Args:
        agent: BRCAgent
        task_name: Task name ('standard', 'windy', 'heavy')
        task_id: Task identifier (0, 1, 2)
        config: Configuration dict with task-specific timeouts
        num_episodes: Number of evaluation episodes
        render: Whether to render the environment

    Returns:
        Dictionary with evaluation metrics
    """
    env = make_env(task_name, render_mode='human' if render else None)
    max_steps = config['max_episode_steps'][task_name]

    rewards = []
    lengths = []
    successes = []

    print(f"\n{'='*60}")
    print(f"Evaluating {task_name.upper()} Task (task_id={task_id})")
    print(f"Max steps: {max_steps}")
    print(f"{'='*60}")

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        truncated = False

        while not (done or truncated):
            # Greedy action selection (epsilon=0.0)
            action = agent.select_action(state, task_id, epsilon=0.0)
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            steps += 1

            # Task-specific timeout
            if steps >= max_steps:
                break

        rewards.append(episode_reward)
        lengths.append(steps)
        successes.append(1 if episode_reward >= 200 else 0)

        # Print progress
        if (episode + 1) % 5 == 0:
            print(f"  Episode {episode + 1}/{num_episodes}: "
                  f"Reward={episode_reward:.2f}, Steps={steps}")

    env.close()

    # Calculate statistics
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    mean_steps = np.mean(lengths)
    success_rate = np.mean(successes)

    print(f"\n{'='*60}")
    print(f"Results for {task_name.upper()}:")
    print(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"  Reward Range: [{np.min(rewards):.2f}, {np.max(rewards):.2f}]")
    print(f"  Mean Steps: {mean_steps:.1f}")
    print(f"  Success Rate: {success_rate*100:.1f}% ({int(success_rate*num_episodes)}/{num_episodes})")
    print(f"{'='*60}\n")

    return {
        'task': task_name,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_steps': mean_steps,
        'success_rate': success_rate,
        'all_rewards': rewards
    }


def evaluate_all_tasks(agent, config, num_episodes=20, render=False):
    """
    Evaluate agent on all tasks and print comparison.

    Args:
        agent: BRCAgent
        config: Configuration dict
        num_episodes: Number of episodes per task
        render: Whether to render
    """
    tasks = [
        ('standard', 0),
        ('windy', 1),
        ('heavy', 2),
    ]

    results = {}

    for task_name, task_id in tasks:
        results[task_name] = evaluate_task(agent, task_name, task_id, config,
                                          num_episodes, render)

    # Print comparison
    print(f"\n{'='*70}")
    print("COMPARISON ACROSS ALL TASKS")
    print(f"{'='*70}")
    print(f"{'Task':<12} {'Mean Reward':<15} {'Success Rate':<15} {'Mean Steps':<12}")
    print("-" * 70)

    for task_name in ['standard', 'windy', 'heavy']:
        r = results[task_name]
        print(f"{task_name:<12} {r['mean_reward']:>6.2f} ± {r['std_reward']:<5.2f}   "
              f"{r['success_rate']*100:>4.0f}% ({int(r['success_rate']*num_episodes)}/{num_episodes})      "
              f"{r['mean_steps']:>6.1f}")

    avg_reward = np.mean([results[t]['mean_reward'] for t in ['standard', 'windy', 'heavy']])
    print("-" * 70)
    print(f"{'Average':<12} {avg_reward:>6.2f}")
    print(f"{'='*70}\n")

    return results


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description='Evaluate BRC agent')
    parser.add_argument('--task', type=str, default=None,
                       help='Specific task to evaluate (standard, windy, heavy). If not provided, evaluates all.')
    parser.add_argument('--episodes', type=int, default=20,
                       help='Number of episodes to evaluate')
    parser.add_argument('--render', action='store_true',
                       help='Render the environment during evaluation')
    parser.add_argument('--model', type=str, default='best',
                       help='Model to load (best or checkpoint_epXXX)')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("BRC EVALUATION")
    print("="*70)

    # Load configuration
    config = get_config()

    # Load model
    model_path = project_root / 'results' / 'brc' / 'models' / f'{args.model}.pth'

    if not model_path.exists():
        print(f"\n❌ ERROR: Model not found at {model_path}")
        print("Please train BRC first using: python -m experiments.brc.train")
        return

    print(f"\n📂 Loading model from: {model_path}")

    # Create agent
    state_dim = 8
    action_dim = 4
    num_tasks = 3

    agent = BRCAgent(
        state_dim=state_dim,
        num_actions=action_dim,
        num_tasks=num_tasks,
        hidden_dim=config['hidden_dim'],
        num_blocks=config['num_blocks'],
        embed_dim=config['embedding_dim'],
        num_atoms=config['num_atoms'],
        v_min=config['v_min'],
        v_max=config['v_max'],
        gamma=config['gamma'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        device=config['device']
    )

    # Load trained weights
    agent.load(str(model_path))
    print(f"✅ Model loaded successfully\n")

    # Evaluate
    if args.task:
        # Single task evaluation
        task_to_id = {'standard': 0, 'windy': 1, 'heavy': 2}
        if args.task not in task_to_id:
            print(f"❌ ERROR: Invalid task '{args.task}'. Choose from: standard, windy, heavy")
            return

        task_id = task_to_id[args.task]
        evaluate_task(agent, args.task, task_id, config, args.episodes, args.render)
    else:
        # Evaluate all tasks
        evaluate_all_tasks(agent, config, args.episodes, args.render)


if __name__ == "__main__":
    main()
