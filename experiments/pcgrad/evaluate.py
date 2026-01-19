"""
PCGrad DQN Evaluation Script

Evaluate a trained PCGrad DQN agent on all 3 Lunar Lander task variants.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from environments.lunar_lander_variants import make_env
from agents.pcgrad import PCGradDQNAgent
from experiments.pcgrad.config import get_config


def evaluate_task(agent, task_name, task_id, config, num_episodes=20, render=False):
    """
    Evaluate agent on a single task.

    Args:
        agent: PCGradDQNAgent
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
            action = agent.select_action(state, task_id, epsilon=0.0)
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            steps += 1

            if steps >= max_steps:
                break

        rewards.append(episode_reward)
        lengths.append(steps)
        successes.append(1 if episode_reward >= 200 else 0)

        if (episode + 1) % 5 == 0:
            print(f"  Episode {episode + 1}/{num_episodes}: "
                  f"Reward={episode_reward:.2f}, Steps={steps}")

    env.close()

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    mean_length = np.mean(lengths)
    success_rate = np.mean(successes) * 100

    results = {
        'task': task_name,
        'task_id': task_id,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_length': mean_length,
        'success_rate': success_rate,
        'all_rewards': rewards,
        'all_lengths': lengths,
    }

    return results


def evaluate_all_tasks(model_path=None, num_episodes=20, render=False):
    """
    Evaluate trained PCGrad DQN agent on all 3 tasks.

    Args:
        model_path: Path to saved model (default: uses output_dir from config)
        num_episodes: Number of episodes per task
        render: Whether to render the environment

    Returns:
        Dictionary of {task_name: results}
    """
    config = get_config()
    use_embedding = config.get('use_task_embedding', True)
    output_dir_name = config.get('output_dir', 'pcgrad')

    if model_path is None:
        model_path = project_root / 'results' / output_dir_name / 'models' / 'best.pth'

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print(f"   Please train the PCGrad DQN model first using:")
        print(f"   python -m experiments.pcgrad.train")
        return None

    state_dim = 8
    action_dim = 4
    num_tasks = 3
    agent = PCGradDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        num_tasks=num_tasks,
        embedding_dim=config['embedding_dim'],
        hidden_dims=config['hidden_dims'],
        learning_rate=config['learning_rate'],
        gamma=config['gamma'],
        use_task_embedding=use_embedding
    )

    mode_str = "TASK-AWARE" if use_embedding else "TASK-BLIND"
    print(f"\n{'='*60}")
    print(f"PCGRAD DQN EVALUATION ({mode_str})")
    print(f"{'='*60}")
    print(f"Loading model from: {model_path}")
    agent.load(str(model_path))
    print(f"Model loaded successfully")

    tasks = ['standard', 'windy', 'heavy']
    task_to_id = {'standard': 0, 'windy': 1, 'heavy': 2}

    all_results = {}
    for task_name in tasks:
        task_id = task_to_id[task_name]
        results = evaluate_task(agent, task_name, task_id, config, num_episodes, render)
        all_results[task_name] = results

    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Episodes per task: {num_episodes}")
    print(f"\nPer-Task Results:")
    print(f"{'Task':<12} {'Mean Reward':<15} {'Std':<10} {'Success Rate':<15} {'Avg Steps':<10}")
    print(f"{'-'*60}")

    for task_name in tasks:
        results = all_results[task_name]
        print(f"{task_name.capitalize():<12} "
              f"{results['mean_reward']:>7.2f} +/- {results['std_reward']:>5.2f}   "
              f"{results['std_reward']:>6.2f}   "
              f"{results['success_rate']:>6.1f}%         "
              f"{results['mean_length']:>6.1f}")

    avg_reward = np.mean([all_results[t]['mean_reward'] for t in tasks])
    print(f"{'-'*60}")
    print(f"{'Average':<12} {avg_reward:>7.2f}")
    print(f"{'='*60}")

    return all_results


def compare_with_baselines(pcgrad_results):
    """
    Compare PCGrad results with Independent and Shared DQN baselines.

    Args:
        pcgrad_results: Dictionary of PCGrad evaluation results
    """
    independent_baselines = {
        'standard': 228.0,
        'windy': 100.0,
        'heavy': 194.0,
    }

    shared_baselines = {
        'standard': 263.0,
        'windy': 130.0,
        'heavy': 224.0,
    }

    print(f"\n{'='*70}")
    print(f"COMPARISON: PCGRAD vs BASELINES")
    print(f"{'='*70}")
    print(f"{'Task':<12} {'Independent':<12} {'Shared':<12} {'PCGrad':<12} {'vs Shared':<12}")
    print(f"{'-'*70}")

    for task_name in ['standard', 'windy', 'heavy']:
        independent = independent_baselines[task_name]
        shared = shared_baselines[task_name]
        pcgrad = pcgrad_results[task_name]['mean_reward']
        diff_shared = pcgrad - shared
        pct_change = (diff_shared / shared) * 100

        print(f"{task_name.capitalize():<12} "
              f"{independent:>7.2f}     "
              f"{shared:>7.2f}     "
              f"{pcgrad:>7.2f}     "
              f"{diff_shared:>+7.2f} ({pct_change:>+5.1f}%)")

    avg_independent = np.mean(list(independent_baselines.values()))
    avg_shared = np.mean(list(shared_baselines.values()))
    avg_pcgrad = np.mean([pcgrad_results[t]['mean_reward'] for t in ['standard', 'windy', 'heavy']])
    diff_shared = avg_pcgrad - avg_shared
    pct_change = (diff_shared / avg_shared) * 100

    print(f"{'-'*70}")
    print(f"{'Average':<12} "
          f"{avg_independent:>7.2f}     "
          f"{avg_shared:>7.2f}     "
          f"{avg_pcgrad:>7.2f}     "
          f"{diff_shared:>+7.2f} ({pct_change:>+5.1f}%)")
    print(f"{'='*70}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate PCGrad DQN on Lunar Lander tasks')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model checkpoint (default: results/pcgrad/models/best.pth)')
    parser.add_argument('--episodes', type=int, default=20,
                        help='Number of evaluation episodes per task (default: 20)')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment during evaluation')
    parser.add_argument('--task', type=str, choices=['standard', 'windy', 'heavy', 'all'], default='all',
                        help='Specific task to evaluate (default: all)')

    args = parser.parse_args()

    if args.task == 'all':
        results = evaluate_all_tasks(args.model, args.episodes, args.render)

        if results is not None:
            compare_with_baselines(results)
    else:
        config = get_config()
        use_embedding = config.get('use_task_embedding', True)
        output_dir_name = config.get('output_dir', 'pcgrad')
        task_to_id = {'standard': 0, 'windy': 1, 'heavy': 2}

        model_path = args.model if args.model else project_root / 'results' / output_dir_name / 'models' / 'best.pth'

        agent = PCGradDQNAgent(
            state_dim=8,
            action_dim=4,
            num_tasks=3,
            embedding_dim=config['embedding_dim'],
            hidden_dims=config['hidden_dims'],
            learning_rate=config['learning_rate'],
            gamma=config['gamma'],
            use_task_embedding=use_embedding
        )
        agent.load(str(model_path))

        results = evaluate_task(agent, args.task, task_to_id[args.task], config, args.episodes, args.render)

        print(f"\n{'='*60}")
        print(f"RESULTS: {args.task.upper()}")
        print(f"{'='*60}")
        print(f"Mean Reward: {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
        print(f"Success Rate: {results['success_rate']:.1f}%")
        print(f"Average Steps: {results['mean_length']:.1f}")
        print(f"{'='*60}")
