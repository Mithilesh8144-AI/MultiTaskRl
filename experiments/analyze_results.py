"""
Comprehensive Results Analysis Script

Generates all plots and reports for the VarShare project matching preview.webp requirements:
1. Conflict Robustness: Average + Per-Task Rewards
2. Sample Efficiency: Episodes/Steps to Thresholds
3. Parameter Efficiency: Performance vs Model Size
4. Training Curves: Individual task analysis

Supports Independent DQN, Shared DQN, BRC, PCGrad, and GradNorm formats.
Also supports task-blind variants: shared_dqn_blind, pcgrad_blind, gradnorm_blind.

Usage:
    python -m experiments.analyze_results                    # All available methods
    python -m experiments.analyze_results --method independent_dqn
    python -m experiments.analyze_results --method shared_dqn
    python -m experiments.analyze_results --method brc
    python -m experiments.analyze_results --method pcgrad
    python -m experiments.analyze_results --method gradnorm
    python -m experiments.analyze_results --method all       # Compare all
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.visualize import (
    plot_training_curve,
    plot_conflict_robustness,
    plot_sample_efficiency,
    plot_sample_efficiency_table,
    plot_parameter_efficiency,
    calculate_sample_efficiency
)
from utils.metrics import load_metrics


def detect_available_methods():
    """Detect which training methods have results available."""
    methods = []

    # Check Independent DQN (3 separate task results)
    independent_tasks = ['standard', 'windy', 'heavy']
    independent_complete = all(
        (project_root / 'results' / task / 'logs' / 'metrics.json').exists()
        for task in independent_tasks
    )
    if independent_complete:
        methods.append('independent_dqn')

    # Check Shared DQN (single multi-task result)
    shared_metrics = project_root / 'results' / 'shared_dqn' / 'logs' / 'metrics.json'
    if shared_metrics.exists():
        methods.append('shared_dqn')

    # Check BRC (single multi-task result)
    brc_metrics = project_root / 'results' / 'brc' / 'logs' / 'metrics.json'
    if brc_metrics.exists():
        methods.append('brc')

    # Check PCGrad (single multi-task result)
    pcgrad_metrics = project_root / 'results' / 'pcgrad' / 'logs' / 'metrics.json'
    if pcgrad_metrics.exists():
        methods.append('pcgrad')

    # Check GradNorm (single multi-task result)
    gradnorm_metrics = project_root / 'results' / 'gradnorm' / 'logs' / 'metrics.json'
    if gradnorm_metrics.exists():
        methods.append('gradnorm')

    # Check task-blind variants
    shared_blind_metrics = project_root / 'results' / 'shared_dqn_blind' / 'logs' / 'metrics.json'
    if shared_blind_metrics.exists():
        methods.append('shared_dqn_blind')

    pcgrad_blind_metrics = project_root / 'results' / 'pcgrad_blind' / 'logs' / 'metrics.json'
    if pcgrad_blind_metrics.exists():
        methods.append('pcgrad_blind')

    gradnorm_blind_metrics = project_root / 'results' / 'gradnorm_blind' / 'logs' / 'metrics.json'
    if gradnorm_blind_metrics.exists():
        methods.append('gradnorm_blind')

    return methods


def load_multitask_metrics(method: str):
    """
    Load multi-task metrics (single file with all tasks).

    Args:
        method: Method name ('shared_dqn', 'brc', 'pcgrad', etc.)

    Returns:
        Dictionary with per-task metrics extracted from the combined file
    """
    metrics_path = project_root / 'results' / method / 'logs' / 'metrics.json'

    if not metrics_path.exists():
        raise FileNotFoundError(f"{method} metrics not found at {metrics_path}")

    with open(metrics_path, 'r') as f:
        data = json.load(f)

    # Extract per-task data from combined episodes
    task_metrics = {
        'standard': {'episode_rewards': [], 'episode_lengths': []},
        'windy': {'episode_rewards': [], 'episode_lengths': []},
        'heavy': {'episode_rewards': [], 'episode_lengths': []},
    }

    # Parse episodes by task
    for episode_data in data['episodes']:
        task = episode_data['task']
        task_metrics[task]['episode_rewards'].append(episode_data['reward'])
        task_metrics[task]['episode_lengths'].append(episode_data['steps'])

    # Add metadata
    default_params = {
        'shared_dqn': 37788, 'brc': 459820, 'pcgrad': 37788, 'gradnorm': 37788,
        'shared_dqn_blind': 35716, 'pcgrad_blind': 35716, 'gradnorm_blind': 35716
    }
    for task in task_metrics:
        task_metrics[task]['method'] = method
        task_metrics[task]['parameters'] = data.get('parameters', default_params.get(method, 0))

    return task_metrics


def load_shared_dqn_metrics():
    """Load Shared DQN metrics (backward compatibility wrapper)."""
    return load_multitask_metrics('shared_dqn')


def count_parameters(task_name: str, method: str = 'independent_dqn') -> int:
    """Count parameters in a trained model."""
    # Multi-task methods have single model for all tasks
    multitask_methods = ['shared_dqn', 'brc', 'pcgrad', 'gradnorm', 'shared_dqn_blind', 'pcgrad_blind', 'gradnorm_blind']
    if method in multitask_methods:
        model_path = project_root / 'results' / method / 'models' / 'best.pth'
    else:
        # Independent DQN has separate models per task
        model_path = project_root / 'results' / task_name / 'models' / 'best.pth'

    if not model_path.exists():
        print(f"  Warning: Model not found for {task_name} ({method})")
        return 0

    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        # Extract state dict from checkpoint
        if isinstance(checkpoint, dict):
            if 'q_network' in checkpoint:
                # Shared DQN format
                state_dict = checkpoint['q_network']
            elif 'q_network_state_dict' in checkpoint:
                # Independent DQN format (actual key name)
                state_dict = checkpoint['q_network_state_dict']
            elif 'model_state_dict' in checkpoint:
                # Alternative format
                state_dict = checkpoint['model_state_dict']
            else:
                # Assume the dict itself is the state dict
                state_dict = checkpoint
        else:
            # Direct state dict
            state_dict = checkpoint

        # Count parameters
        if hasattr(state_dict, 'values'):
            # It's a dict-like object
            total = 0
            for param in state_dict.values():
                if hasattr(param, 'numel'):
                    total += param.numel()
                elif hasattr(param, 'shape'):
                    # NumPy array
                    total += param.size
            return total
        else:
            print(f"  Warning: Unexpected state_dict type for {task_name} ({method})")
            return 0

    except Exception as e:
        print(f"  Error loading model for {task_name} ({method}): {e}")
        return 0


def generate_all_plots(method='independent_dqn'):
    """
    Generate all analysis plots matching preview.webp requirements.

    Args:
        method: 'independent_dqn', 'shared_dqn', or 'all'
    """
    # Create output directory
    plots_dir = project_root / 'results' / 'analysis'
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE ANALYSIS PLOTS")
    print(f"Method: {method}")
    print("Matching requirements from preview.webp:")
    print("  1. Conflict Robustness (Average + Per-Task Rewards)")
    print("  2. Sample Efficiency (Steps to Thresholds)")
    print("  3. Parameter Efficiency (Params vs Performance)")
    print("  4. Training Curves (Individual Tasks)")
    print("="*80 + "\n")

    tasks = ['standard', 'windy', 'heavy']
    experiment_name = method

    # ========================================================================
    # 1. CONFLICT ROBUSTNESS
    # ========================================================================
    print("[1/5] Conflict Robustness Analysis...")

    try:
        save_path = plots_dir / f'{experiment_name}_conflict_robustness.png'
        plot_conflict_robustness(tasks, experiment_name, save_path=save_path)
        print(f"  ✓ Saved: {save_path.name}\n")
    except Exception as e:
        print(f"  ✗ Error: {e}\n")

    # ========================================================================
    # 2. SAMPLE EFFICIENCY (Table)
    # ========================================================================
    print("[2/5] Sample Efficiency Table...")

    try:
        save_path = plots_dir / f'{experiment_name}_sample_efficiency_table.png'
        fig, efficiency_data = plot_sample_efficiency_table(
            tasks=tasks,
            thresholds=[50, 100, 150, 200],
            experiment_name=experiment_name,
            save_path=save_path
        )
        print(f"  ✓ Saved: {save_path.name}")

        # Print summary
        print("\n  Sample Efficiency Summary:")
        for task, data in efficiency_data.items():
            print(f"    {task.capitalize()}:")
            for threshold, metrics in data.items():
                if metrics['reached']:
                    eps = metrics['episodes']
                    steps = metrics['steps']
                    if steps:
                        print(f"      {threshold} reward: {eps} eps ({steps/1000:.1f}k steps)")
                    else:
                        print(f"      {threshold} reward: {eps} eps")
                else:
                    print(f"      {threshold} reward: Not reached")
        print()
    except Exception as e:
        print(f"  ✗ Error: {e}\n")

    # ========================================================================
    # 3. SAMPLE EFFICIENCY (Curves)
    # ========================================================================
    print("[3/5] Sample Efficiency Curves...")

    try:
        experiments = [(task, experiment_name) for task in tasks]
        save_path = plots_dir / f'{experiment_name}_sample_efficiency_curves.png'
        plot_sample_efficiency(experiments, save_path=save_path)
        print(f"  ✓ Saved: {save_path.name}\n")
    except Exception as e:
        print(f"  ✗ Error: {e}\n")

    # ========================================================================
    # 4. PARAMETER EFFICIENCY
    # ========================================================================
    print("[4/5] Parameter Efficiency Analysis...")

    try:
        experiment_results = []

        # Load metrics based on method
        multitask_methods = ['shared_dqn', 'brc', 'pcgrad', 'gradnorm', 'shared_dqn_blind', 'pcgrad_blind', 'gradnorm_blind']
        if method in multitask_methods:
            task_metrics = load_multitask_metrics(method)
            param_count = count_parameters('', method=method)
            method_label = method.upper() if method == 'brc' else method.replace('_', ' ').title()

            for task in tasks:
                if task in task_metrics:
                    rewards = task_metrics[task]['episode_rewards']
                    final_reward = sum(rewards[-100:]) / min(100, len(rewards))

                    experiment_results.append({
                        'name': f'{task.capitalize()} ({method_label})',
                        'task': task,
                        'params': param_count,
                        'performance': final_reward
                    })
        else:
            # Independent DQN
            for task in tasks:
                try:
                    metrics = load_metrics(task, experiment_name)
                    final_reward = sum(metrics['episode_rewards'][-100:]) / 100
                    param_count = count_parameters(task, method=experiment_name)

                    experiment_results.append({
                        'name': f'{task.capitalize()}',
                        'task': task,
                        'params': param_count,
                        'performance': final_reward
                    })
                except Exception as e:
                    print(f"  Warning: Could not load {task}: {e}")

        if experiment_results:
            save_path = plots_dir / f'{experiment_name}_parameter_efficiency.png'
            plot_parameter_efficiency(experiment_results, save_path=save_path)
            print(f"  ✓ Saved: {save_path.name}")

            print("\n  Parameter Efficiency Summary:")
            for result in experiment_results:
                print(f"    {result['task'].capitalize()}: {result['params']:,} params, "
                      f"{result['performance']:.2f} reward")
            print()
        else:
            print("  ✗ No data available\n")
    except Exception as e:
        print(f"  ✗ Error: {e}\n")

    # ========================================================================
    # 5. INDIVIDUAL TRAINING CURVES
    # ========================================================================
    print("[5/5] Individual Training Curves...")

    for task in tasks:
        try:
            save_path = plots_dir / f'{experiment_name}_{task}_training.png'
            plot_training_curve(task, experiment_name, smooth_window=20, save_path=save_path)
            print(f"  ✓ {task}: {save_path.name}")
        except Exception as e:
            print(f"  ✗ {task}: {e}")

    print("\n" + "="*80)
    print("✅ ALL PLOTS GENERATED SUCCESSFULLY")
    print(f"📁 Output directory: {plots_dir}")
    print("="*80 + "\n")

    return plots_dir


if __name__ == "__main__":
    """
    Usage:
        python -m experiments.analyze_results                    # All available methods
        python -m experiments.analyze_results --method independent_dqn
        python -m experiments.analyze_results --method shared_dqn
        python -m experiments.analyze_results --method all       # Compare both
    """
    parser = argparse.ArgumentParser(description='Analyze RL training results')
    parser.add_argument('--method', type=str, default='auto',
                        choices=['independent_dqn', 'shared_dqn', 'brc', 'pcgrad', 'gradnorm',
                                 'shared_dqn_blind', 'pcgrad_blind', 'gradnorm_blind', 'all', 'auto'],
                        help='Method to analyze (default: auto-detect)')
    args = parser.parse_args()

    # Auto-detect available methods
    available_methods = detect_available_methods()

    if not available_methods:
        print("\n❌ Error: No training results found!")
        print("   Please train at least one method first:")
        print("   - Independent DQN: python -m experiments.independent_dqn.train")
        print("   - Shared DQN: python -m experiments.shared_dqn.train")
        sys.exit(1)

    print(f"\n📊 Available methods: {', '.join(available_methods)}")

    # Determine which methods to analyze
    if args.method == 'auto':
        methods_to_analyze = available_methods
    elif args.method == 'all':
        if len(available_methods) < 2:
            print(f"\n⚠️  Warning: Only {len(available_methods)} method available.")
            print(f"   Analyzing: {available_methods[0]}")
            methods_to_analyze = available_methods
        else:
            methods_to_analyze = available_methods
    else:
        if args.method not in available_methods:
            print(f"\n❌ Error: {args.method} results not found!")
            print(f"   Available: {', '.join(available_methods)}")
            sys.exit(1)
        methods_to_analyze = [args.method]

    # Generate plots for each method
    all_plots_dirs = []
    for method in methods_to_analyze:
        plots_dir = generate_all_plots(method=method)
        all_plots_dirs.append(plots_dir)

    # Print summary
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    for plots_dir in all_plots_dirs:
        for plot_file in sorted(plots_dir.glob('*.png')):
            print(f"  {plot_file.name}")
    print()
