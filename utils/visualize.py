"""
Visualization utilities for comparing RL experiments.

Creates publication-quality plots for:
1. Training curves (rewards over episodes)
2. Sample efficiency (reward vs environment steps)
3. Parameter efficiency (performance vs model size)
4. Multi-task comparison
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


def smooth_curve(data: List[float], window: int = 10) -> np.ndarray:
    """
    Smooth a noisy curve using moving average.

    Args:
        data: List of values to smooth
        window: Window size for moving average

    Returns:
        Smoothed numpy array
    """
    if len(data) < window:
        return np.array(data)

    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def plot_training_curve(task_name: str, experiment_name: str = 'independent_dqn',
                        smooth_window: int = 10, save_path: Optional[Path] = None):
    """
    Plot training curve for a single experiment.

    Args:
        task_name: Name of the task
        experiment_name: Name of the experiment (independent_dqn, shared_dqn, brc, pcgrad)
        smooth_window: Window for smoothing noisy rewards
        save_path: Path to save figure (optional)
    """
    project_root = Path(__file__).parent.parent

    # Multi-task methods store all tasks in one file
    multitask_methods = ['shared_dqn', 'brc', 'pcgrad', 'shared_dqn_blind', 'pcgrad_blind']

    if experiment_name in multitask_methods:
        # Load from multi-task format: results/{method}/logs/metrics.json
        metrics_path = project_root / 'results' / experiment_name / 'logs' / 'metrics.json'
        with open(metrics_path, 'r') as f:
            data = json.load(f)

        # Extract task-specific episodes
        episode_rewards = [ep['reward'] for ep in data['episodes'] if ep['task'] == task_name]

        # Handle eval_history - could be list of floats or list of dicts
        raw_eval = data.get('eval_history', {}).get(task_name, [])
        if raw_eval and isinstance(raw_eval[0], dict):
            # BRC format: list of {'episode': x, 'reward': y}
            eval_rewards = [e['reward'] for e in raw_eval]
            eval_episodes = [e['episode'] // 3 for e in raw_eval]  # Convert global to per-task episodes
        else:
            # Shared DQN format: list of floats
            eval_rewards = raw_eval
            eval_freq = data.get('config', {}).get('eval_freq', 50)
            eval_episodes = list(range(eval_freq // 3, len(episode_rewards) + 1, eval_freq // 3))[:len(eval_rewards)]
    else:
        # Load metrics from Independent DQN format: results/{task_name}/logs/metrics.json
        metrics_path = project_root / 'results' / task_name / 'logs' / 'metrics.json'
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        episode_rewards = metrics['episode_rewards']
        eval_rewards = metrics.get('eval_rewards', [])
        eval_episodes = metrics.get('eval_episodes', [])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Episode rewards (raw + smoothed)
    episodes = np.arange(1, len(episode_rewards) + 1)
    ax1.plot(episodes, episode_rewards, alpha=0.3, label='Raw rewards', color='blue')

    if len(episode_rewards) >= smooth_window:
        smoothed = smooth_curve(episode_rewards, smooth_window)
        smooth_episodes = episodes[:len(smoothed)]
        ax1.plot(smooth_episodes, smoothed, label=f'Smoothed (window={smooth_window})', color='blue', linewidth=2)

    # Plot eval rewards
    if eval_rewards:
        ax1.scatter(eval_episodes, eval_rewards, color='red', s=50, zorder=5, label='Eval rewards', marker='D')

    # Threshold lines
    thresholds = [50, 100, 150, 200]
    for threshold in thresholds:
        ax1.axhline(y=threshold, color='green', linestyle='--', alpha=0.3, linewidth=1)
        ax1.text(len(episodes) * 0.98, threshold, f'{threshold}', ha='right', va='bottom', fontsize=9, color='green')

    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.set_title(f'{experiment_name.upper()} on {task_name.upper()} Task - Training Rewards', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Moving average (last 100 episodes)
    if len(episode_rewards) >= 100:
        moving_avg = [np.mean(episode_rewards[max(0, i-99):i+1]) for i in range(len(episode_rewards))]
        ax2.plot(episodes, moving_avg, label='Avg reward (last 100 episodes)', color='purple', linewidth=2)

        # Threshold lines
        for threshold in thresholds:
            ax2.axhline(y=threshold, color='green', linestyle='--', alpha=0.3, linewidth=1)

        ax2.set_xlabel('Episode', fontsize=12)
        ax2.set_ylabel('Average Reward', fontsize=12)
        ax2.set_title('Moving Average (100 episodes)', fontsize=12, fontweight='bold')
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

    return fig


def plot_sample_efficiency(experiments: List[Tuple[str, str]], save_path: Optional[Path] = None):
    """
    Plot sample efficiency comparison: Reward vs Environment Steps.

    Args:
        experiments: List of (task_name, experiment_name) tuples
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    project_root = Path(__file__).parent.parent

    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    multitask_methods = ['shared_dqn', 'brc', 'pcgrad', 'shared_dqn_blind', 'pcgrad_blind']

    # Cache loaded multi-task data to avoid reloading
    multitask_cache = {}

    for (task_name, experiment_name), color in zip(experiments, colors):
        if experiment_name in multitask_methods:
            # Load from multi-task format (with caching)
            if experiment_name not in multitask_cache:
                metrics_path = project_root / 'results' / experiment_name / 'logs' / 'metrics.json'
                with open(metrics_path, 'r') as f:
                    multitask_cache[experiment_name] = json.load(f)

            data = multitask_cache[experiment_name]
            episode_rewards = [ep['reward'] for ep in data['episodes'] if ep['task'] == task_name]
            total_steps = data.get('total_env_steps', None)

            if total_steps is None or not episode_rewards:
                print(f"Warning: No data found for {experiment_name} on {task_name}")
                continue

            # Approximate steps per task episode
            steps_per_episode = total_steps / len(data['episodes'])
            cumulative_steps = np.arange(1, len(episode_rewards) + 1) * steps_per_episode * 3
        else:
            # Load metrics from Independent DQN format: results/{task_name}/logs/metrics.json
            metrics_path = project_root / 'results' / task_name / 'logs' / 'metrics.json'
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)

            episode_rewards = metrics['episode_rewards']
            total_steps = metrics.get('total_env_steps', None)

            if total_steps is None:
                print(f"Warning: No total_env_steps found for {experiment_name} on {task_name}")
                continue

            # Compute cumulative steps per episode (approximate)
            steps_per_episode = total_steps / len(episode_rewards)
            cumulative_steps = np.arange(1, len(episode_rewards) + 1) * steps_per_episode

        # Smooth rewards
        smoothed_rewards = smooth_curve(episode_rewards, window=20)
        cumulative_steps_smooth = cumulative_steps[:len(smoothed_rewards)]

        ax.plot(cumulative_steps_smooth, smoothed_rewards,
               label=f'{experiment_name} - {task_name}', color=color, linewidth=2)

    # Threshold lines
    thresholds = [50, 100, 150, 200]
    for threshold in thresholds:
        ax.axhline(y=threshold, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        ax.text(ax.get_xlim()[1] * 0.02, threshold, f'{threshold}', fontsize=9, color='gray')

    ax.set_xlabel('Total Environment Steps', fontsize=12)
    ax.set_ylabel('Average Reward (smoothed)', fontsize=12)
    ax.set_title('Sample Efficiency Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

    return fig


def calculate_sample_efficiency(tasks: List[str], thresholds: List[int] = [50, 100, 150, 200],
                                experiment_name: str = 'independent_dqn') -> Dict:
    """
    Calculate sample efficiency: episodes/steps required to reach performance thresholds.

    Args:
        tasks: List of task names
        thresholds: List of reward thresholds to measure
        experiment_name: Name of experiment (independent_dqn, shared_dqn, brc, pcgrad)

    Returns:
        Dictionary with efficiency metrics per task and threshold
    """
    efficiency_data = {}
    project_root = Path(__file__).parent.parent

    # Multi-task methods store all tasks in one file
    multitask_methods = ['shared_dqn', 'brc', 'pcgrad', 'shared_dqn_blind', 'pcgrad_blind']

    if experiment_name in multitask_methods:
        # Load from multi-task format
        metrics_path = project_root / 'results' / experiment_name / 'logs' / 'metrics.json'
        try:
            with open(metrics_path, 'r') as f:
                data = json.load(f)

            total_steps = data.get('total_env_steps', None)

            for task in tasks:
                episode_rewards = [ep['reward'] for ep in data['episodes'] if ep['task'] == task]
                if not episode_rewards:
                    efficiency_data[task] = {th: {'episodes': None, 'steps': None, 'reached': False}
                                            for th in thresholds}
                    continue

                # Calculate moving average (last 100 episodes)
                moving_avg = [np.mean(episode_rewards[max(0, i-99):i+1])
                             for i in range(len(episode_rewards))]

                task_efficiency = {}
                for threshold in thresholds:
                    episodes_to_threshold = None
                    steps_to_threshold = None

                    for i, avg in enumerate(moving_avg):
                        if avg >= threshold:
                            episodes_to_threshold = i + 1
                            if total_steps is not None:
                                # Approximate steps per task episode
                                steps_per_episode = total_steps / len(data['episodes'])
                                steps_to_threshold = int(episodes_to_threshold * steps_per_episode * 3)  # 3 tasks
                            break

                    task_efficiency[threshold] = {
                        'episodes': episodes_to_threshold,
                        'steps': steps_to_threshold,
                        'reached': episodes_to_threshold is not None
                    }

                efficiency_data[task] = task_efficiency

        except FileNotFoundError:
            print(f"Warning: No metrics found for {experiment_name}")
            for task in tasks:
                efficiency_data[task] = {th: {'episodes': None, 'steps': None, 'reached': False}
                                        for th in thresholds}

        return efficiency_data

    # Independent DQN format
    for task in tasks:
        metrics_path = project_root / 'results' / task / 'logs' / 'metrics.json'
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)

            episode_rewards = metrics['episode_rewards']
            total_steps = metrics.get('total_env_steps', None)

            # Calculate moving average (last 100 episodes)
            moving_avg = [np.mean(episode_rewards[max(0, i-99):i+1])
                         for i in range(len(episode_rewards))]

            task_efficiency = {}

            for threshold in thresholds:
                # Find first episode where moving avg exceeds threshold
                episodes_to_threshold = None
                steps_to_threshold = None

                for i, avg in enumerate(moving_avg):
                    if avg >= threshold:
                        episodes_to_threshold = i + 1
                        if total_steps is not None:
                            steps_per_episode = total_steps / len(episode_rewards)
                            steps_to_threshold = int(episodes_to_threshold * steps_per_episode)
                        break

                task_efficiency[threshold] = {
                    'episodes': episodes_to_threshold,
                    'steps': steps_to_threshold,
                    'reached': episodes_to_threshold is not None
                }

            efficiency_data[task] = task_efficiency

        except FileNotFoundError:
            print(f"Warning: No metrics found for {task}")
            efficiency_data[task] = {th: {'episodes': None, 'steps': None, 'reached': False}
                                    for th in thresholds}

    return efficiency_data


def plot_sample_efficiency_table(tasks: List[str], thresholds: List[int] = [50, 100, 150, 200],
                                 experiment_name: str = 'independent_dqn',
                                 save_path: Optional[Path] = None):
    """
    Create a table visualization of sample efficiency metrics.

    Args:
        tasks: List of task names
        thresholds: List of reward thresholds
        experiment_name: Name of experiment
        save_path: Path to save figure (optional)
    """
    efficiency_data = calculate_sample_efficiency(tasks, thresholds, experiment_name)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')

    # Build table data
    headers = ['Task'] + [f'{th} Reward' for th in thresholds]
    table_data = []

    for task in tasks:
        row = [task.capitalize()]
        for threshold in thresholds:
            data = efficiency_data[task][threshold]
            if data['reached']:
                episodes = data['episodes']
                steps = data['steps']
                if steps:
                    cell_text = f"{episodes} eps\n({steps/1000:.1f}k steps)"
                else:
                    cell_text = f"{episodes} eps"
            else:
                cell_text = "Not reached"
            row.append(cell_text)
        table_data.append(row)

    table = ax.table(cellText=table_data, colLabels=headers, loc='center',
                    cellLoc='center', colColours=['lightgray'] * len(headers))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.title(f'{experiment_name.upper()} - Sample Efficiency (Episodes/Steps to Threshold)',
             fontsize=14, fontweight='bold', pad=20)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Table saved to: {save_path}")
    else:
        plt.show()

    return fig, efficiency_data


def plot_parameter_efficiency(experiment_results: List[Dict], save_path: Optional[Path] = None):
    """
    Plot parameter efficiency: Performance vs Model Size.

    Args:
        experiment_results: List of dicts with keys:
            - 'name': Experiment name
            - 'task': Task name
            - 'params': Parameter count
            - 'performance': Final average reward
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Extract data
    names = [r['name'] for r in experiment_results]
    params = [r['params'] for r in experiment_results]
    performance = [r['performance'] for r in experiment_results]

    # Scatter plot
    colors = plt.cm.viridis(np.linspace(0, 1, len(experiment_results)))
    scatter = ax.scatter(params, performance, c=colors, s=200, alpha=0.7, edgecolors='black', linewidth=1.5)

    # Annotate points
    for i, name in enumerate(names):
        ax.annotate(name, (params[i], performance[i]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.3))

    ax.set_xlabel('Total Parameters', fontsize=12)
    ax.set_ylabel('Final Average Reward', fontsize=12)
    ax.set_title('Parameter Efficiency: Performance vs Model Size', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Log scale for parameters if range is large
    if max(params) / min(params) > 10:
        ax.set_xscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

    return fig


def plot_conflict_robustness(tasks: List[str], experiment_name: str = 'independent_dqn',
                            save_path: Optional[Path] = None):
    """
    Plot conflict robustness: Per-task rewards + average across tasks.

    This visualization shows whether one task is being sacrificed for another,
    which is critical for identifying gradient conflicts in multi-task learning.

    Args:
        tasks: List of task names (e.g., ['standard', 'windy', 'heavy'])
        experiment_name: Name of the experiment (independent_dqn, shared_dqn, brc, pcgrad)
        save_path: Path to save figure (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    project_root = Path(__file__).parent.parent

    # Collect per-task data
    task_rewards = {}
    max_episodes = 0

    # Multi-task methods store all tasks in one file
    multitask_methods = ['shared_dqn', 'brc', 'pcgrad', 'shared_dqn_blind', 'pcgrad_blind']

    if experiment_name in multitask_methods:
        # Load from multi-task format
        metrics_path = project_root / 'results' / experiment_name / 'logs' / 'metrics.json'
        try:
            with open(metrics_path, 'r') as f:
                data = json.load(f)

            for task in tasks:
                episode_rewards = [ep['reward'] for ep in data['episodes'] if ep['task'] == task]
                if episode_rewards:
                    task_rewards[task] = episode_rewards
                    max_episodes = max(max_episodes, len(episode_rewards))
        except FileNotFoundError:
            print(f"Warning: No metrics found for {experiment_name}")
    else:
        # Independent DQN format
        for task in tasks:
            metrics_path = project_root / 'results' / task / 'logs' / 'metrics.json'
            try:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                episode_rewards = metrics['episode_rewards']
                task_rewards[task] = episode_rewards
                max_episodes = max(max_episodes, len(episode_rewards))
            except FileNotFoundError:
                print(f"Warning: No metrics found for {task}")
                continue

    if not task_rewards:
        print("Error: No task data found!")
        return None

    # Plot 1: Per-Task Rewards (smoothed)
    colors = {'standard': 'blue', 'windy': 'orange', 'heavy': 'green'}

    for task, rewards in task_rewards.items():
        episodes = np.arange(1, len(rewards) + 1)
        smoothed = smooth_curve(rewards, window=20)
        smooth_episodes = episodes[:len(smoothed)]

        color = colors.get(task, 'gray')
        ax1.plot(smooth_episodes, smoothed, label=f'{task.capitalize()}',
                color=color, linewidth=2)

    ax1.axhline(y=200, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Target (200)')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Reward (smoothed)', fontsize=12)
    ax1.set_title('Per-Task Performance', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Final Average Reward Comparison
    final_avgs = []
    final_stds = []
    task_labels = []

    for task in tasks:
        if task in task_rewards:
            final_100 = task_rewards[task][-100:]
            final_avgs.append(np.mean(final_100))
            final_stds.append(np.std(final_100))
            task_labels.append(task.capitalize())

    # Add overall average
    if task_rewards:
        all_final_rewards = [np.mean(rewards[-100:]) for rewards in task_rewards.values()]
        overall_avg = np.mean(all_final_rewards)
        overall_std = np.std(all_final_rewards)
        final_avgs.append(overall_avg)
        final_stds.append(overall_std)
        task_labels.append('Average')

    x = np.arange(len(task_labels))
    bars = ax2.bar(x, final_avgs, yerr=final_stds, capsize=5, alpha=0.7,
                   color=['blue', 'orange', 'green', 'purple'][:len(task_labels)])

    # Color the average bar differently
    if len(bars) > len(tasks):
        bars[-1].set_color('purple')
        bars[-1].set_alpha(0.9)

    ax2.axhline(y=200, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Target (200)')
    ax2.set_xlabel('Task', fontsize=12)
    ax2.set_ylabel('Final Avg Reward (last 100 eps)', fontsize=12)
    ax2.set_title('Conflict Robustness: Per-Task vs Average', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(task_labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'{experiment_name.upper()} - Conflict Robustness Analysis',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

    return fig


def plot_multi_task_comparison(tasks: List[str], experiment_names: List[str],
                               save_path: Optional[Path] = None):
    """
    Plot multi-task performance comparison across experiments.

    Args:
        tasks: List of task names
        experiment_names: List of experiment names
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Collect data
    data = {exp: [] for exp in experiment_names}

    for exp_name in experiment_names:
        for task in tasks:
            # Load from new folder structure: results/{task}/logs/metrics.json
            metrics_path = Path(__file__).parent.parent / 'results' / task / 'logs' / 'metrics.json'
            try:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                episode_rewards = metrics['episode_rewards']
                final_avg = np.mean(episode_rewards[-100:])
                data[exp_name].append(final_avg)
            except FileNotFoundError:
                data[exp_name].append(0)
                print(f"Warning: No metrics for {exp_name} on {task}")

    # Bar plot
    x = np.arange(len(tasks))
    width = 0.8 / len(experiment_names)

    for i, exp_name in enumerate(experiment_names):
        offset = (i - len(experiment_names)/2 + 0.5) * width
        ax.bar(x + offset, data[exp_name], width, label=exp_name, alpha=0.8)

    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('Final Average Reward', fontsize=12)
    ax.set_title('Multi-Task Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

    return fig


if __name__ == "__main__":
    """Example usage: Plot Heavy task training curve"""
    import sys

    if len(sys.argv) > 1:
        task = sys.argv[1]
    else:
        task = 'heavy'

    print(f"Plotting training curve for {task} task...")
    plot_training_curve(task, save_path=Path(f'results/plots/training_curve_{task}.png'))
