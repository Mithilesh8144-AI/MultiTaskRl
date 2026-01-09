"""
Independent DQN Training Script

Train a separate DQN agent on a single Lunar Lander task variant.
This provides the upper bound on performance (no multi-task interference).
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
from agents.dqn import DQNAgent
from utils.replay_buffer import ReplayBuffer
from config import get_config


# ============================================================================
# CONFIGURATION - Change this to switch tasks
# ============================================================================
TASK_NAME = 'standard'  # Options: 'standard', 'windy', 'heavy'


def save_progress_checkpoint(episode_rewards, episode_losses, eval_rewards, eval_episodes, task_name,
                           total_env_steps=None, total_gradient_updates=None, performance_thresholds=None):
    """Save training progress to JSON file."""
    # Task-specific folder structure: results/{task_name}/logs/metrics.json
    output_dir = project_root / 'results' / task_name / 'logs'
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'episode_rewards': episode_rewards,
        'episode_losses': episode_losses,
        'eval_rewards': eval_rewards,
        'eval_episodes': eval_episodes,
        'total_env_steps': total_env_steps,
        'total_gradient_updates': total_gradient_updates,
        'performance_thresholds': performance_thresholds or {},
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    checkpoint_path = output_dir / 'metrics.json'
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)

    print(f"💾 Progress saved to {checkpoint_path}")


def train(task_name):
    """
    Train Independent DQN on a single task.

    Args:
        task_name (str): One of 'standard', 'windy', 'heavy'
    """
    # Load configuration
    config = get_config(task_name)
    print(f"\n{'='*80}")
    print(f"INDEPENDENT DQN - {task_name.upper()} TASK")
    print(f"{'='*80}")
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"{'='*80}\n")

    # Create task-specific output directories: results/{task_name}/{logs,models,plots}
    output_base = project_root / 'results' / task_name
    (output_base / 'models').mkdir(parents=True, exist_ok=True)
    (output_base / 'logs').mkdir(parents=True, exist_ok=True)
    (output_base / 'plots').mkdir(parents=True, exist_ok=True)

    # Create environment
    env = make_env(task_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Create agent
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=config['learning_rate'],
        gamma=config['gamma'],
        epsilon_start=config['epsilon_start'],
        epsilon_end=config['epsilon_end'],
        epsilon_decay=config['epsilon_decay'],
        target_update_freq=config['target_update_freq']
    )

    # Create replay buffer
    replay_buffer = ReplayBuffer(config['replay_buffer_size'])

    # Training statistics
    episode_rewards = []
    episode_losses = []
    eval_rewards = []
    eval_episodes = []

    # Sample efficiency metrics
    total_env_steps = 0
    total_gradient_updates = 0
    performance_thresholds = {50: None, 100: None, 150: None, 200: None}

    # Best model tracking
    best_eval_reward = -np.inf

    # Safety: Maximum steps per episode
    MAX_EPISODE_STEPS = config['max_episode_steps']

    # Start training
    start_time = time.time()
    print(f"\n🚀 Starting training on {task_name} task...")
    print("💡 Progress bar will update below.\n")

    # Progress bar
    pbar = tqdm(
        total=config['num_episodes'],
        desc=f"Training {task_name}",
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    )

    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    for episode in range(config['num_episodes']):
        state, info = env.reset()
        episode_reward = 0
        episode_loss = []
        done = False
        truncated = False
        steps = 0

        # Play one episode
        while not (done or truncated):
            # Select action
            action = agent.select_action(state)

            # Take action
            next_state, reward, done, truncated, info = env.step(action)

            # Store transition
            replay_buffer.push(state, action, reward, next_state, done)

            # Train if we have enough experiences
            if len(replay_buffer) >= config['min_replay_size']:
                batch = replay_buffer.sample(config['batch_size'])
                loss = agent.update(batch)
                episode_loss.append(loss)
                total_gradient_updates += 1

            episode_reward += reward
            state = next_state
            steps += 1
            total_env_steps += 1

            # Safety check: prevent infinite episodes
            if steps >= MAX_EPISODE_STEPS:
                pbar.write(f"⚠️  Training episode {episode+1} exceeded {MAX_EPISODE_STEPS} steps, forcing termination")
                break

        # Decay epsilon
        agent.decay_epsilon()
        agent.episodes += 1

        # Update target network
        if episode % agent.target_update_freq == 0:
            agent.update_target_network()

        # Store statistics
        episode_rewards.append(episode_reward)
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        episode_losses.append(avg_loss)
        avg_reward_100 = np.mean(episode_rewards[-100:]) if episode_rewards else 0

        # Check thresholds
        for threshold, first_episode in performance_thresholds.items():
            if first_episode is None and avg_reward_100 >= threshold:
                performance_thresholds[threshold] = {
                    'episode': episode + 1,
                    'total_steps': total_env_steps,
                    'gradient_updates': total_gradient_updates
                }
                pbar.write(f"🎯 Threshold {threshold} reached at episode {episode+1} (steps: {total_env_steps:,})")

        # Update progress bar
        pbar.set_postfix({
            'reward': f'{episode_reward:.1f}',
            'avg_100': f'{avg_reward_100:.1f}',
            'ε': f'{agent.epsilon:.3f}',
            'loss': f'{avg_loss:.2f}'
        })
        pbar.update(1)

        # Print summary every 100 episodes
        if (episode + 1) % 100 == 0:
            pbar.write(f"\n[Episode {episode+1:4d}] Reward: {episode_reward:7.2f} | Avg(100): {avg_reward_100:7.2f} | Loss: {avg_loss:6.4f}")
            pbar.write(f"               Steps: {total_env_steps:,} | Updates: {total_gradient_updates:,}")

        # Save progress checkpoint every 50 episodes
        if (episode + 1) % 50 == 0:
            save_progress_checkpoint(episode_rewards, episode_losses, eval_rewards, eval_episodes, task_name,
                                   total_env_steps, total_gradient_updates, performance_thresholds)

        # ====================================================================
        # EVALUATION
        # ====================================================================
        if episode % config['eval_freq'] == 0 and episode > 0:
            eval_reward_mean = 0
            eval_reward_list = []

            for _ in range(config['eval_episodes']):
                eval_state, _ = env.reset()
                eval_reward = 0
                eval_done = False
                eval_truncated = False
                eval_steps = 0

                while not (eval_done or eval_truncated):
                    eval_action = agent.select_action(eval_state, epsilon=0.0)
                    eval_state, r, eval_done, eval_truncated, _ = env.step(eval_action)
                    eval_reward += r

                    eval_steps += 1
                    if eval_steps >= MAX_EPISODE_STEPS:
                        pbar.write(f"⚠️  Eval episode exceeded {MAX_EPISODE_STEPS} steps, forcing termination")
                        break

                eval_reward_list.append(eval_reward)
                eval_reward_mean += eval_reward

            eval_reward_mean /= config['eval_episodes']
            eval_reward_std = np.std(eval_reward_list)
            eval_rewards.append(eval_reward_mean)
            eval_episodes.append(episode)

            pbar.write("-" * 80)
            if eval_reward_mean > best_eval_reward:
                best_eval_reward = eval_reward_mean
                model_path = output_base / 'models' / 'best.pth'
                agent.save(str(model_path))
                pbar.write(f"[EVAL @ Episode {episode+1}] Mean: {eval_reward_mean:7.2f} (±{eval_reward_std:5.2f}) ⭐ NEW BEST!")
            else:
                pbar.write(f"[EVAL @ Episode {episode+1}] Mean: {eval_reward_mean:7.2f} (±{eval_reward_std:5.2f})")
            pbar.write("-" * 80)

        # Checkpoint
        if episode % config['save_freq'] == 0 and episode > 0:
            checkpoint_path = output_base / 'models' / f'checkpoint_ep{episode}.pth'
            agent.save(str(checkpoint_path))
            pbar.write(f"[CHECKPOINT] Saved to {checkpoint_path}")

    # Close progress bar
    pbar.close()

    # Training complete
    training_time = time.time() - start_time

    print("\n" + "=" * 80)
    print(f"✅ Training on {task_name.upper()} complete!")
    print("=" * 80)
    print(f"Training time: {training_time/60:.2f} minutes ({training_time:.1f} seconds)")
    print(f"Best eval reward: {best_eval_reward:.2f}")
    print(f"Final avg reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
    print()
    print("SAMPLE EFFICIENCY METRICS:")
    print(f"  Total environment steps: {total_env_steps:,}")
    print(f"  Total gradient updates: {total_gradient_updates:,}")
    print(f"  Steps per episode (avg): {total_env_steps / len(episode_rewards):.1f}")
    print()
    print("PERFORMANCE THRESHOLDS REACHED:")
    for threshold in sorted(performance_thresholds.keys()):
        milestone = performance_thresholds[threshold]
        if milestone:
            print(f"  Reward ≥ {threshold:3d}: Episode {milestone['episode']:4d} | Steps: {milestone['total_steps']:,} | Updates: {milestone['gradient_updates']:,}")
        else:
            print(f"  Reward ≥ {threshold:3d}: Not reached")
    print("=" * 80)

    env.close()

    # Save final metrics
    save_progress_checkpoint(episode_rewards, episode_losses, eval_rewards, eval_episodes, task_name,
                           total_env_steps, total_gradient_updates, performance_thresholds)


if __name__ == "__main__":
    train(TASK_NAME)
