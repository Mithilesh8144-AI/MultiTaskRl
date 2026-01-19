"""
PCGrad DQN Training Script

Train a multi-task DQN agent using PCGrad (Projected Gradient) optimization.
PCGrad projects conflicting gradients to eliminate negative interference.

Key differences from Shared DQN:
    - Uses PerTaskReplayBuffer (separate buffer per task)
    - Computes per-task losses and applies gradient projection
    - Tracks and logs gradient conflict statistics
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from environments.lunar_lander_variants import make_env
from agents.pcgrad import PCGradDQNAgent, PerTaskReplayBuffer
from experiments.pcgrad.config import get_config


def save_pcgrad_metrics(all_episodes, eval_history, best_rewards, config,
                        total_env_steps, total_gradient_updates, conflict_history, output_base):
    """
    Save training metrics including PCGrad-specific conflict statistics.

    Args:
        all_episodes: List of episode dictionaries with per-episode data
        eval_history: Dict of {task: [eval_rewards]} for each task
        best_rewards: Dict of {task: best_reward} for each task
        config: Configuration dictionary
        total_env_steps: Total environment steps taken
        total_gradient_updates: Total gradient updates performed
        conflict_history: List of conflict ratios per update
        output_base: Base output directory path
    """
    output_dir = output_base / 'logs'
    output_dir.mkdir(parents=True, exist_ok=True)

    method_name = 'pcgrad' if config.get('use_task_embedding', True) else 'pcgrad_blind'
    metrics = {
        'method': method_name,
        'total_episodes': len(all_episodes),
        'episodes_per_task': config['num_episodes_per_task'],
        'parameters': 37788 if config.get('use_task_embedding', True) else 35716,  # Fewer params without embeddings
        'episodes': all_episodes,
        'eval_history': eval_history,
        'best_rewards': best_rewards,
        'total_env_steps': total_env_steps,
        'total_gradient_updates': total_gradient_updates,
        'conflict_history': conflict_history,
        'avg_conflict_ratio': np.mean(conflict_history) if conflict_history else 0.0,
        'config': config,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to {metrics_path}")


def evaluate_all_tasks(agent, envs, task_to_id, config, num_episodes=5):
    """
    Evaluate agent on all 3 tasks.

    Args:
        agent: PCGradDQNAgent
        envs: Dict of {task_name: env}
        task_to_id: Dict of {task_name: task_id}
        config: Configuration dict with task-specific timeouts
        num_episodes: Number of episodes per task

    Returns:
        Dict of {task_name: mean_reward}
    """
    results = {}

    for task_name, env in envs.items():
        task_id = task_to_id[task_name]
        task_rewards = []
        max_steps = config['max_episode_steps'][task_name]

        for _ in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            truncated = False
            steps = 0

            while not (done or truncated):
                action = agent.select_action(state, task_id, epsilon=0.0)
                state, reward, done, truncated, _ = env.step(action)
                episode_reward += reward
                steps += 1

                if steps >= max_steps:
                    break

            task_rewards.append(episode_reward)

        results[task_name] = np.mean(task_rewards)

    return results


def train():
    """Train PCGrad DQN on all 3 tasks simultaneously."""
    # Load configuration
    config = get_config()
    use_embedding = config.get('use_task_embedding', True)
    output_dir_name = config.get('output_dir', 'pcgrad')

    mode_str = "TASK-AWARE" if use_embedding else "TASK-BLIND"
    print(f"\n{'='*80}")
    print(f"PCGRAD DQN - MULTI-TASK TRAINING ({mode_str})")
    print(f"{'='*80}")
    print(f"Configuration:")
    for key, value in config.items():
        if key != 'max_episode_steps':
            print(f"  {key}: {value}")
        else:
            print(f"  {key}:")
            for task, steps in value.items():
                print(f"    {task}: {steps}")
    print(f"{'='*80}\n")

    # Create output directories
    output_base = project_root / 'results' / output_dir_name
    (output_base / 'models').mkdir(parents=True, exist_ok=True)
    (output_base / 'logs').mkdir(parents=True, exist_ok=True)
    (output_base / 'plots').mkdir(parents=True, exist_ok=True)

    # Create all 3 environments
    envs = {
        'standard': make_env('standard'),
        'windy': make_env('windy'),
        'heavy': make_env('heavy'),
    }
    task_to_id = {'standard': 0, 'windy': 1, 'heavy': 2}
    tasks = ['standard', 'windy', 'heavy']

    # Create PCGrad agent
    state_dim = 8
    action_dim = 4
    use_embedding = config.get('use_task_embedding', True)
    agent = PCGradDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        num_tasks=3,
        embedding_dim=config['embedding_dim'],
        hidden_dims=config['hidden_dims'],
        learning_rate=config['learning_rate'],
        gamma=config['gamma'],
        use_task_embedding=use_embedding
    )
    if not use_embedding:
        print("⚠️  TASK-BLIND MODE: Network has no task embeddings")

    # Create per-task replay buffers
    # Total capacity split across tasks: 100000 / 3 = ~33333 per task
    capacity_per_task = config['replay_buffer_size'] // 3
    replay_buffer = PerTaskReplayBuffer(
        num_tasks=3,
        capacity_per_task=capacity_per_task
    )

    # Training statistics
    all_episodes = []
    eval_history = {'standard': [], 'windy': [], 'heavy': []}
    best_rewards = {'standard': -np.inf, 'windy': -np.inf, 'heavy': -np.inf}

    # Sample efficiency tracking
    total_env_steps = 0
    total_gradient_updates = 0

    # PCGrad-specific tracking
    conflict_history = []
    recent_conflicts = []

    # Calculate total episodes
    total_episodes = config['num_episodes_per_task'] * len(tasks)

    # Start training
    start_time = time.time()
    print(f"\nStarting PCGrad multi-task training...")
    print(f"Total episodes: {total_episodes} ({config['num_episodes_per_task']} per task)")
    print(f"Tasks: {', '.join(tasks)}")
    print("Progress bar will update below.\n")

    # Progress bar
    pbar = tqdm(
        total=total_episodes,
        desc="Training PCGrad DQN",
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    )

    # ========================================================================
    # TRAINING LOOP (Round-robin across tasks)
    # ========================================================================
    for episode in range(total_episodes):
        # Round-robin task selection
        task_name = tasks[episode % len(tasks)]
        task_id = task_to_id[task_name]
        env = envs[task_name]

        # Get task-specific timeout
        max_steps = config['max_episode_steps'][task_name]

        # Episode state
        state, _ = env.reset()
        episode_reward = 0
        episode_losses = []
        episode_conflicts = []
        done = False
        truncated = False
        steps = 0

        # Play one episode
        while not (done or truncated):
            # Select action (with task conditioning)
            action = agent.select_action(state, task_id)

            # Take action
            next_state, reward, done, truncated, _ = env.step(action)

            # Store transition in per-task buffer
            replay_buffer.push(state, action, reward, next_state, done, task_id)

            # Update agent if ALL task buffers have enough samples
            # This ensures clean per-task gradient computation
            if replay_buffer.can_sample_all(config['batch_size']):
                loss, conflict_stats = agent.update(replay_buffer, config['batch_size'])
                episode_losses.append(loss)
                episode_conflicts.append(conflict_stats['conflict_ratio'])
                total_gradient_updates += 1

                # Track conflict history
                conflict_history.append(conflict_stats['conflict_ratio'])

            state = next_state
            episode_reward += reward
            steps += 1
            total_env_steps += 1

            # Task-specific timeout
            if steps >= max_steps:
                truncated = True

        # Store episode data (including conflict info)
        avg_conflict = np.mean(episode_conflicts) if episode_conflicts else 0.0
        recent_conflicts.append(avg_conflict)
        if len(recent_conflicts) > 100:
            recent_conflicts.pop(0)

        episode_data = {
            'episode': episode,
            'task': task_name,
            'task_id': task_id,
            'reward': episode_reward,
            'steps': steps,
            'loss': np.mean(episode_losses) if episode_losses else 0.0,
            'epsilon': agent.epsilon,
            'conflict_ratio': avg_conflict
        }
        all_episodes.append(episode_data)

        # Decay epsilon
        agent.decay_epsilon()

        # Update target network
        if episode % config['target_update_freq'] == 0 and episode > 0:
            agent.update_target_network()

        # Evaluation (every eval_freq episodes)
        if episode % config['eval_freq'] == 0 and episode > 0:
            eval_results = evaluate_all_tasks(agent, envs, task_to_id, config,
                                              num_episodes=config['eval_episodes'])

            # Store eval results
            for task, reward in eval_results.items():
                eval_history[task].append(reward)

                if reward > best_rewards[task]:
                    best_rewards[task] = reward

            # Log eval results and conflict stats
            avg_recent_conflict = np.mean(recent_conflicts) if recent_conflicts else 0.0
            pbar.write(f"\nEpisode {episode} Evaluation:")
            for task, reward in eval_results.items():
                pbar.write(f"   {task.capitalize()}: {reward:7.2f}")
            pbar.write(f"   Avg Conflict Ratio: {avg_recent_conflict:.3f}")

        # Log conflict stats periodically
        if episode % config['gradient_log_freq'] == 0 and episode > 0 and recent_conflicts:
            avg_recent_conflict = np.mean(recent_conflicts)
            # Just update progress bar postfix - don't spam output

        # Save checkpoint
        if episode % config['save_freq'] == 0 and episode > 0:
            checkpoint_path = output_base / 'models' / f'checkpoint_ep{episode}.pth'
            agent.save(str(checkpoint_path))

        # Update progress bar
        pbar.update(1)
        avg_conf = np.mean(recent_conflicts[-10:]) if recent_conflicts else 0.0
        pbar.set_postfix({
            'Task': task_name[:3].upper(),
            'Reward': f'{episode_reward:.1f}',
            'Eps': f'{agent.epsilon:.3f}',
            'Conf': f'{avg_conf:.2f}'
        })

    pbar.close()

    # ========================================================================
    # TRAINING COMPLETE
    # ========================================================================
    training_time = time.time() - start_time

    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Training time: {training_time/60:.2f} minutes ({training_time:.1f} seconds)")
    print(f"\nBest Evaluation Rewards:")
    for task, reward in best_rewards.items():
        print(f"  {task.capitalize()}: {reward:.2f}")
    print(f"\nPCGrad Statistics:")
    if conflict_history:
        print(f"  Average conflict ratio: {np.mean(conflict_history):.3f}")
        print(f"  Max conflict ratio: {np.max(conflict_history):.3f}")
        print(f"  Min conflict ratio: {np.min(conflict_history):.3f}")
    print(f"\nSample Efficiency:")
    print(f"  Total environment steps: {total_env_steps:,}")
    print(f"  Total gradient updates: {total_gradient_updates:,}")
    print(f"{'='*80}\n")

    # Save final model
    final_model_path = output_base / 'models' / 'best.pth'
    agent.save(str(final_model_path))
    print(f"Final model saved to {final_model_path}")

    # Save metrics
    save_pcgrad_metrics(
        all_episodes, eval_history, best_rewards, config,
        total_env_steps, total_gradient_updates, conflict_history, output_base
    )

    # Close environments
    for env in envs.values():
        env.close()

    print("\nTraining pipeline complete!")


if __name__ == "__main__":
    train()
