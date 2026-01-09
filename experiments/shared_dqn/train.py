"""
Shared DQN Training Script

Train a single DQN agent on all 3 Lunar Lander task variants simultaneously.
Episodes cycle through tasks in round-robin fashion (standard → windy → heavy).

Expected: ~60% performance degradation vs Independent DQN due to gradient conflicts.
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
from agents.shared_dqn import SharedDQNAgent, MultiTaskReplayBuffer
from config import get_config


def save_shared_metrics(all_episodes, eval_history, best_rewards, config,
                       total_env_steps, total_gradient_updates):
    """
    Save training metrics in format compatible with analyze_results.py.

    Args:
        all_episodes: List of episode dictionaries with per-episode data
        eval_history: Dict of {task: [eval_rewards]} for each task
        best_rewards: Dict of {task: best_reward} for each task
        config: Configuration dictionary
        total_env_steps: Total environment steps taken
        total_gradient_updates: Total gradient updates performed
    """
    output_dir = project_root / 'results' / 'shared_dqn' / 'logs'
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        'method': 'shared_dqn',
        'total_episodes': len(all_episodes),
        'episodes_per_task': config['num_episodes_per_task'],
        'parameters': 37788,  # Shared network parameters
        'episodes': all_episodes,
        'eval_history': eval_history,
        'best_rewards': best_rewards,
        'total_env_steps': total_env_steps,
        'total_gradient_updates': total_gradient_updates,
        'config': config,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"💾 Metrics saved to {metrics_path}")


def evaluate_all_tasks(agent, envs, task_to_id, config, num_episodes=5):
    """
    Evaluate agent on all 3 tasks.

    Args:
        agent: SharedDQNAgent
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
        max_steps = config['max_episode_steps'][task_name]  # Task-specific timeout

        for _ in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            truncated = False
            steps = 0

            while not (done or truncated):
                action = agent.select_action(state, task_id, epsilon=0.0)  # Greedy
                state, reward, done, truncated, _ = env.step(action)
                episode_reward += reward
                steps += 1

                # Use task-specific timeout (Standard: 1000, Windy: 400, Heavy: 800)
                if steps >= max_steps:
                    break

            task_rewards.append(episode_reward)

        results[task_name] = np.mean(task_rewards)

    return results


def train():
    """Train Shared DQN on all 3 tasks simultaneously."""
    # Load configuration
    config = get_config()
    print(f"\n{'='*80}")
    print(f"SHARED DQN - MULTI-TASK TRAINING")
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
    output_base = project_root / 'results' / 'shared_dqn'
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

    # Create shared agent
    state_dim = 8  # Lunar Lander state dimension
    action_dim = 4  # Lunar Lander action space
    agent = SharedDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        num_tasks=3,
        embedding_dim=config['embedding_dim'],
        hidden_dims=config['hidden_dims'],
        learning_rate=config['learning_rate'],
        gamma=config['gamma']
    )

    # Create shared replay buffer
    replay_buffer = MultiTaskReplayBuffer(config['replay_buffer_size'])

    # Training statistics
    all_episodes = []  # Store all episode data
    eval_history = {'standard': [], 'windy': [], 'heavy': []}
    best_rewards = {'standard': -np.inf, 'windy': -np.inf, 'heavy': -np.inf}

    # Sample efficiency tracking
    total_env_steps = 0
    total_gradient_updates = 0

    # Calculate total episodes (500 per task × 3 tasks)
    total_episodes = config['num_episodes_per_task'] * len(tasks)

    # Start training
    start_time = time.time()
    print(f"\n🚀 Starting multi-task training...")
    print(f"📊 Total episodes: {total_episodes} ({config['num_episodes_per_task']} per task)")
    print(f"🔄 Tasks: {', '.join(tasks)}")
    print("💡 Progress bar will update below.\n")

    # Progress bar
    pbar = tqdm(
        total=total_episodes,
        desc="Training Shared DQN",
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
        done = False
        truncated = False
        steps = 0

        # Play one episode
        while not (done or truncated):
            # Select action (with task conditioning)
            action = agent.select_action(state, task_id)

            # Take action
            next_state, reward, done, truncated, _ = env.step(action)

            # Store transition in shared buffer (with task_id)
            replay_buffer.push(state, action, reward, next_state, done, task_id)

            # Update agent if enough samples
            if len(replay_buffer) > config['min_replay_size']:
                batch = replay_buffer.sample(config['batch_size'])
                loss = agent.update(*batch)
                episode_losses.append(loss)
                total_gradient_updates += 1

            state = next_state
            episode_reward += reward
            steps += 1
            total_env_steps += 1

            # Task-specific timeout
            if steps >= max_steps:
                truncated = True

        # Store episode data
        episode_data = {
            'episode': episode,
            'task': task_name,
            'task_id': task_id,
            'reward': episode_reward,
            'steps': steps,
            'loss': np.mean(episode_losses) if episode_losses else 0.0,
            'epsilon': agent.epsilon
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

                # Track best model per task
                if reward > best_rewards[task]:
                    best_rewards[task] = reward

            # Log eval results
            pbar.write(f"\n📊 Episode {episode} Evaluation:")
            for task, reward in eval_results.items():
                pbar.write(f"   {task.capitalize()}: {reward:7.2f}")

        # Save checkpoint
        if episode % config['save_freq'] == 0 and episode > 0:
            checkpoint_path = output_base / 'models' / f'checkpoint_ep{episode}.pth'
            agent.save(str(checkpoint_path))

        # Update progress bar
        pbar.update(1)
        pbar.set_postfix({
            'Task': task_name[:3].upper(),
            'Reward': f'{episode_reward:.1f}',
            'ε': f'{agent.epsilon:.3f}'
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
    print(f"\nSample Efficiency:")
    print(f"  Total environment steps: {total_env_steps:,}")
    print(f"  Total gradient updates: {total_gradient_updates:,}")
    print(f"{'='*80}\n")

    # Save final model
    final_model_path = output_base / 'models' / 'best.pth'
    agent.save(str(final_model_path))
    print(f"💾 Final model saved to {final_model_path}")

    # Save metrics
    save_shared_metrics(
        all_episodes, eval_history, best_rewards, config,
        total_env_steps, total_gradient_updates
    )

    # Close environments
    for env in envs.values():
        env.close()

    print("\n✅ Training pipeline complete!")


if __name__ == "__main__":
    train()
