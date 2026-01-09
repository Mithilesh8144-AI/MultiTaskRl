"""
Metrics Utilities for Reinforcement Learning Experiments

Provides functions for:
1. Computing sample efficiency metrics
2. Computing parameter efficiency metrics
3. Analyzing training results
4. Comparing multiple experiments
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """
    Count total trainable parameters in a PyTorch model.

    Args:
        model: PyTorch neural network module

    Returns:
        Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_training_metrics(task_name: str, experiment_name: str = 'independent_dqn') -> Dict:
    """
    Load training metrics from JSON file.

    Args:
        task_name: Name of the task (e.g., 'standard', 'windy', 'heavy')
        experiment_name: Name of the experiment folder (currently unused, for future multi-method support)

    Returns:
        Dictionary containing training metrics
    """
    # New folder structure: results/{task_name}/logs/metrics.json
    metrics_path = Path(__file__).parent.parent / 'results' / task_name / 'logs' / 'metrics.json'

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    with open(metrics_path, 'r') as f:
        return json.load(f)


def compute_sample_efficiency(metrics: Dict) -> Dict:
    """
    Compute sample efficiency statistics from training metrics.

    Sample efficiency measures how quickly the agent learns (fewer samples = more efficient).

    Args:
        metrics: Dictionary with training metrics

    Returns:
        Dictionary with sample efficiency statistics:
        - total_env_steps: Total environment interactions
        - total_gradient_updates: Total learning updates
        - steps_per_episode: Average steps per episode
        - threshold_achievements: Episodes/steps when reaching performance thresholds
    """
    episode_rewards = metrics.get('episode_rewards', [])
    total_env_steps = metrics.get('total_env_steps', None)
    total_gradient_updates = metrics.get('total_gradient_updates', None)
    performance_thresholds = metrics.get('performance_thresholds', {})

    # Compute average steps per episode
    num_episodes = len(episode_rewards)
    avg_steps_per_episode = total_env_steps / num_episodes if total_env_steps else None

    return {
        'total_env_steps': total_env_steps,
        'total_gradient_updates': total_gradient_updates,
        'steps_per_episode': avg_steps_per_episode,
        'num_episodes': num_episodes,
        'threshold_achievements': performance_thresholds
    }


def compute_final_performance(metrics: Dict, window: int = 100) -> Dict:
    """
    Compute final performance statistics.

    Args:
        metrics: Dictionary with training metrics
        window: Window size for computing average (default: last 100 episodes)

    Returns:
        Dictionary with performance statistics
    """
    episode_rewards = metrics.get('episode_rewards', [])
    eval_rewards = metrics.get('eval_rewards', [])

    if not episode_rewards:
        return {}

    # Final performance (last N episodes)
    final_rewards = episode_rewards[-window:]
    final_avg = np.mean(final_rewards)
    final_std = np.std(final_rewards)
    final_min = np.min(final_rewards)
    final_max = np.max(final_rewards)

    # Best eval performance
    best_eval = max(eval_rewards) if eval_rewards else None
    final_eval = eval_rewards[-1] if eval_rewards else None

    # Success rate (episodes with positive reward)
    success_rate = sum(1 for r in final_rewards if r > 0) / len(final_rewards)

    return {
        'final_avg_reward': final_avg,
        'final_std_reward': final_std,
        'final_min_reward': final_min,
        'final_max_reward': final_max,
        'best_eval_reward': best_eval,
        'final_eval_reward': final_eval,
        'success_rate': success_rate,
        'window_size': window
    }


def analyze_experiment(task_name: str, experiment_name: str = 'independent_dqn',
                      parameter_count: Optional[int] = None) -> Dict:
    """
    Full analysis of a single experiment.

    Args:
        task_name: Name of the task
        experiment_name: Name of the experiment
        parameter_count: Number of model parameters (if available)

    Returns:
        Comprehensive analysis dictionary
    """
    metrics = load_training_metrics(task_name, experiment_name)
    sample_efficiency = compute_sample_efficiency(metrics)
    performance = compute_final_performance(metrics)

    analysis = {
        'task_name': task_name,
        'experiment_name': experiment_name,
        'parameter_count': parameter_count,
        'sample_efficiency': sample_efficiency,
        'performance': performance,
        'raw_metrics': metrics
    }

    return analysis


def compare_experiments(experiments: List[Tuple[str, str, int]]) -> Dict:
    """
    Compare multiple experiments across sample and parameter efficiency.

    Args:
        experiments: List of (task_name, experiment_name, parameter_count) tuples

    Returns:
        Comparison dictionary with rankings and statistics
    """
    results = []

    for task_name, experiment_name, param_count in experiments:
        try:
            analysis = analyze_experiment(task_name, experiment_name, param_count)
            results.append(analysis)
        except FileNotFoundError:
            print(f"Warning: No metrics found for {experiment_name} on {task_name}")
            continue

    if not results:
        return {}

    # Create comparison table
    comparison = {
        'experiments': results,
        'rankings': {
            'by_performance': sorted(results, key=lambda x: x['performance'].get('final_avg_reward', -np.inf), reverse=True),
            'by_sample_efficiency': sorted(results, key=lambda x: x['sample_efficiency'].get('total_env_steps', np.inf)),
            'by_parameter_efficiency': sorted(results, key=lambda x: (x['parameter_count'] or np.inf, -x['performance'].get('final_avg_reward', -np.inf)))
        }
    }

    return comparison


def print_experiment_summary(analysis: Dict):
    """
    Pretty print experiment analysis.

    Args:
        analysis: Analysis dictionary from analyze_experiment()
    """
    print(f"\n{'='*80}")
    print(f"EXPERIMENT SUMMARY: {analysis['experiment_name'].upper()} - {analysis['task_name'].upper()}")
    print(f"{'='*80}")

    # Performance
    perf = analysis['performance']
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"  Final Avg Reward (last 100): {perf.get('final_avg_reward', 'N/A'):.2f} ± {perf.get('final_std_reward', 0):.2f}")
    print(f"  Best Eval Reward: {perf.get('best_eval_reward', 'N/A'):.2f}")
    print(f"  Success Rate: {perf.get('success_rate', 0)*100:.1f}%")

    # Sample Efficiency
    sample_eff = analysis['sample_efficiency']
    print(f"\n⚡ SAMPLE EFFICIENCY:")
    print(f"  Total Environment Steps: {sample_eff.get('total_env_steps', 'N/A'):,}")
    print(f"  Total Gradient Updates: {sample_eff.get('total_gradient_updates', 'N/A'):,}")
    print(f"  Steps per Episode (avg): {sample_eff.get('steps_per_episode', 0):.1f}")

    # Thresholds
    thresholds = sample_eff.get('threshold_achievements', {})
    if thresholds:
        print(f"\n🎯 PERFORMANCE THRESHOLDS:")
        for threshold in sorted([int(k) for k in thresholds.keys()]):
            milestone = thresholds[str(threshold)]
            if milestone:
                print(f"  Reward ≥ {threshold:3d}: Episode {milestone.get('episode', 'N/A'):4d} | "
                      f"Steps: {milestone.get('total_steps', 'N/A'):,} | "
                      f"Updates: {milestone.get('gradient_updates', 'N/A'):,}")
            else:
                print(f"  Reward ≥ {threshold:3d}: Not reached")

    # Parameter Efficiency
    if analysis['parameter_count']:
        print(f"\n🔢 PARAMETER EFFICIENCY:")
        print(f"  Total Parameters: {analysis['parameter_count']:,}")
        print(f"  Performance per 1K params: {perf.get('final_avg_reward', 0) / (analysis['parameter_count'] / 1000):.4f}")

    print(f"\n{'='*80}\n")


def save_analysis(analysis: Dict, output_path: Path):
    """
    Save analysis results to JSON file.

    Args:
        analysis: Analysis dictionary
        output_path: Path to save JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to JSON-serializable format
    def make_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        return obj

    serializable_analysis = make_serializable(analysis)

    with open(output_path, 'w') as f:
        json.dump(serializable_analysis, f, indent=2)

    print(f"Analysis saved to: {output_path}")


# Alias for backward compatibility
load_metrics = load_training_metrics
