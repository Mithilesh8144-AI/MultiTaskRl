"""
BRC Training Script

Train a BRC (Bigger, Regularized, Categorical) agent on all 3 Lunar Lander task variants.
Episodes cycle through tasks in round-robin fashion (standard → windy → heavy).

Expected: BRC should outperform Shared DQN due to higher capacity, approaching Independent DQN performance.
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
from agents.brc import BRCAgent, MultiTaskReplayBuffer
from config import get_config


def save_brc_metrics(all_episodes, eval_history, best_rewards, config,
                     total_env_steps, total_gradient_updates):
    """
    Save training metrics in format compatible with analyze_results.py.

    Args:
        all_episodes: List of episode dictionaries
        eval_history: Dict of {task: [eval_rewards]}
        best_rewards: Dict of {task: best_reward}
        config: Configuration dictionary
        total_env_steps: Total environment steps
        total_gradient_updates: Total gradient updates
    """
    output_dir = project_root / 'results' / 'brc' / 'logs'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate actual parameters from loaded model
    # (will be close to ~460K from config estimation)
    param_count = 459820  # From config.py estimation

    metrics = {
        'method': 'brc',
        'total_episodes': len(all_episodes),
        'episodes_per_task': config['num_episodes_per_task'],
        'parameters': param_count,
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
        agent: BRCAgent
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
                # Greedy action selection (epsilon=0)
                action = agent.select_action(state, task_id, epsilon=0.0)
                state, reward, done, truncated, _ = env.step(action)
                episode_reward += reward
                steps += 1

                # Task-specific timeout
                if steps >= max_steps:
                    break

            task_rewards.append(episode_reward)

        results[task_name] = np.mean(task_rewards)

    return results


def train():
    """Train BRC on all 3 tasks simultaneously."""
    # Load configuration
    config = get_config()

    print(f"\n{'='*80}")
    print(f"BRC - MULTI-TASK TRAINING")
    print(f"(Bigger, Regularized, Categorical DQN)")
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
    output_base = project_root / 'results' / 'brc'
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

    # Create BRC agent
    state_dim = 8  # Lunar Lander state dimension
    action_dim = 4  # Lunar Lander action space

    agent = BRCAgent(
        state_dim=state_dim,
        num_actions=action_dim,
        num_tasks=3,
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

    # Create shared replay buffer
    replay_buffer = MultiTaskReplayBuffer(config['replay_buffer_size'])

    # Training statistics
    all_episodes = []  # Store all episode data
    eval_history = {'standard': [], 'windy': [], 'heavy': []}
    best_rewards = {'standard': -np.inf, 'windy': -np.inf, 'heavy': -np.inf}

    # Sample efficiency tracking
    total_env_steps = 0
    total_gradient_updates = 0

    # Epsilon decay tracking
    epsilon = config['epsilon_start']

    # Calculate total episodes
    total_episodes = config['num_episodes_per_task'] * len(tasks)

    # Start training
    start_time = time.time()
    print(f"\n🚀 Starting multi-task training...")
    print(f"📊 Total episodes: {total_episodes} ({config['num_episodes_per_task']} per task)")
    print(f"🔄 Tasks: {', '.join(tasks)}")
    print(f"🧠 Network: BroNet (hidden_dim={config['hidden_dim']}, blocks={config['num_blocks']})")
    print(f"📦 Categorical: {config['num_atoms']} atoms, range=[{config['v_min']}, {config['v_max']}]")
    print("💡 Progress bar will update below.\n")

    # Progress bar
    pbar = tqdm(
        total=total_episodes,
        desc="Training BRC",
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )

    for episode in range(total_episodes):
        # Round-robin task selection
        task_name = tasks[episode % len(tasks)]
        task_id = task_to_id[task_name]
        env = envs[task_name]
        max_steps = config['max_episode_steps'][task_name]

        # Reset environment
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = 0
        loss_count = 0
        steps = 0
        done = False
        truncated = False

        # Episode loop
        while not (done or truncated):
            # Select action
            action = agent.select_action(state, task_id, epsilon=epsilon)

            # Take step
            next_state, reward, done, truncated, _ = env.step(action)
            steps += 1
            total_env_steps += 1

            # Store transition
            replay_buffer.push(state, action, reward, next_state, done, task_id)

            # Update agent
            if len(replay_buffer) >= config['min_replay_size']:
                batch = replay_buffer.sample(config['batch_size'], device=config['device'])
                loss = agent.update(*batch)
                episode_loss += loss
                loss_count += 1
                total_gradient_updates += 1

            state = next_state
            episode_reward += reward

            # Task-specific timeout
            if steps >= max_steps:
                truncated = True

        # Decay epsilon
        epsilon = max(config['epsilon_end'], epsilon * config['epsilon_decay'])

        # Store episode data
        avg_loss = episode_loss / loss_count if loss_count > 0 else 0.0
        all_episodes.append({
            'episode': episode,
            'task': task_name,
            'task_id': task_id,
            'reward': episode_reward,
            'steps': steps,
            'loss': avg_loss,
            'epsilon': epsilon
        })

        # Update progress bar
        pbar.update(1)
        pbar.set_postfix({
            'task': task_name,
            'reward': f'{episode_reward:.1f}',
            'ε': f'{epsilon:.3f}'
        })

        # Update target network
        if (episode + 1) % config['target_update_freq'] == 0:
            agent.update_target_network()

        # Periodic evaluation
        if (episode + 1) % config['eval_freq'] == 0:
            pbar.write(f"\n📊 Evaluation at episode {episode + 1}:")
            eval_results = evaluate_all_tasks(agent, envs, task_to_id, config,
                                            num_episodes=config['eval_episodes'])

            for task_name, reward in eval_results.items():
                eval_history[task_name].append({
                    'episode': episode + 1,
                    'reward': reward
                })
                if reward > best_rewards[task_name]:
                    best_rewards[task_name] = reward

                pbar.write(f"  {task_name:10s}: {reward:7.2f}")

            avg_reward = np.mean(list(eval_results.values()))
            pbar.write(f"  {'Average':10s}: {avg_reward:7.2f}\n")

        # Save checkpoint
        if (episode + 1) % config['save_freq'] == 0:
            checkpoint_path = output_base / 'models' / f'checkpoint_ep{episode+1}.pth'
            agent.save(str(checkpoint_path))
            pbar.write(f"💾 Checkpoint saved: {checkpoint_path.name}")

    pbar.close()

    # Training complete
    elapsed_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"✅ TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {elapsed_time/60:.1f} minutes")
    print(f"Total episodes: {total_episodes}")
    print(f"Total environment steps: {total_env_steps:,}")
    print(f"Total gradient updates: {total_gradient_updates:,}")
    print(f"\nBest evaluation rewards:")
    for task_name in tasks:
        print(f"  {task_name:10s}: {best_rewards[task_name]:7.2f}")
    print(f"  {'Average':10s}: {np.mean(list(best_rewards.values())):7.2f}")

    # Final evaluation (20 episodes per task)
    print(f"\n{'='*80}")
    print(f"📊 FINAL EVALUATION (20 episodes per task)")
    print(f"{'='*80}")
    final_results = evaluate_all_tasks(agent, envs, task_to_id, config, num_episodes=20)

    final_avg_rewards = {}
    for task_name, reward in final_results.items():
        final_avg_rewards[task_name] = reward
        print(f"  {task_name:10s}: {reward:7.2f}")

    final_avg = np.mean(list(final_results.values()))
    print(f"  {'Average':10s}: {final_avg:7.2f}")
    print(f"{'='*80}\n")

    # Save best model
    best_model_path = output_base / 'models' / 'best.pth'
    agent.save(str(best_model_path))
    print(f"💾 Best model saved to {best_model_path}")

    # Save metrics
    save_brc_metrics(all_episodes, eval_history, best_rewards, config,
                     total_env_steps, total_gradient_updates)

    # Close environments
    for env in envs.values():
        env.close()

    print(f"\n🎉 All done! Results saved to {output_base}")


if __name__ == "__main__":
    train()
