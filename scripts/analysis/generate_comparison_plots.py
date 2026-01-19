"""
Generate side-by-side comparison plots for Independent DQN vs Shared DQN.
Creates publication-ready visualizations highlighting key differences.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

# Colors
COLOR_INDEPENDENT = '#2E86AB'  # Blue
COLOR_SHARED = '#A23B72'       # Purple/Magenta
COLOR_STANDARD = '#06A77D'     # Green
COLOR_WINDY = '#F18F01'        # Orange
COLOR_HEAVY = '#C73E1D'        # Red


def load_data():
    """Load all experimental data."""
    project_root = Path(__file__).parent

    # Independent DQN data
    independent_data = {}
    for task in ['standard', 'windy', 'heavy']:
        try:
            with open(project_root / 'results' / task / 'logs' / 'metrics.json', 'r') as f:
                data = json.load(f)
                # Handle both old format (episode_rewards list) and new format (episodes list)
                if 'episode_rewards' in data:
                    # Old format
                    independent_data[task] = {
                        'rewards': data['episode_rewards'],
                        'eval': data.get('final_evaluation', {}),
                        'parameters': data.get('parameters', 35716)
                    }
                else:
                    # New format (shouldn't happen for Independent DQN, but just in case)
                    independent_data[task] = {
                        'rewards': [ep['reward'] for ep in data['episodes']],
                        'eval': data.get('final_evaluation', {}),
                        'parameters': data.get('parameters', 35716)
                    }
        except Exception as e:
            print(f"Warning: Could not load {task} data: {e}")

    # Shared DQN data
    shared_data = None
    try:
        with open(project_root / 'results' / 'shared_dqn' / 'logs' / 'metrics.json', 'r') as f:
            shared_data = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load shared_dqn data: {e}")

    return independent_data, shared_data


def plot_1_performance_comparison(independent_data, shared_data, output_dir):
    """
    Plot 1: Bar chart comparing final performance across methods and tasks.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    tasks = ['standard', 'windy', 'heavy']
    task_labels = ['Standard', 'Windy', 'Heavy Weight']

    # Get final performance (last 100 episodes)
    independent_scores = []
    shared_scores = []

    for task in tasks:
        # Independent DQN
        if task in independent_data:
            rewards = independent_data[task]['rewards']
            last_100 = rewards[-100:] if len(rewards) >= 100 else rewards
            independent_scores.append(np.mean(last_100))
        else:
            independent_scores.append(0)

        # Shared DQN
        if shared_data:
            task_episodes = [ep for ep in shared_data['episodes'] if ep['task'] == task]
            last_100 = task_episodes[-100:]
            shared_scores.append(np.mean([ep['reward'] for ep in last_100]))
        else:
            shared_scores.append(0)

    # Create bars
    x = np.arange(len(tasks))
    width = 0.35

    bars1 = ax.bar(x - width/2, independent_scores, width, label='Independent DQN',
                   color=COLOR_INDEPENDENT, alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, shared_scores, width, label='Shared DQN',
                   color=COLOR_SHARED, alpha=0.8, edgecolor='black', linewidth=1)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9)

    # Add improvement percentages
    for i, (ind_score, shared_score) in enumerate(zip(independent_scores, shared_scores)):
        improvement = ((shared_score - ind_score) / ind_score) * 100
        y_pos = max(ind_score, shared_score) + 10
        ax.text(i, y_pos, f'+{improvement:.1f}%',
               ha='center', fontsize=9, color='green', fontweight='bold')

    ax.set_xlabel('Task', fontweight='bold')
    ax.set_ylabel('Average Reward (Last 100 Episodes)', fontweight='bold')
    ax.set_title('Performance Comparison: Independent DQN vs Shared DQN',
                fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=200, color='gray', linestyle='--', alpha=0.5, label='Success Threshold (200)')

    plt.tight_layout()
    output_path = output_dir / 'comparison_1_performance.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Saved: {output_path.name}")
    plt.close()


def plot_2_parameter_efficiency(independent_data, shared_data, output_dir):
    """
    Plot 2: Scatter plot showing parameter efficiency (params vs performance).
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    tasks = ['standard', 'windy', 'heavy']
    task_colors = {'standard': COLOR_STANDARD, 'windy': COLOR_WINDY, 'heavy': COLOR_HEAVY}

    # Independent DQN points
    for task in tasks:
        if task in independent_data:
            rewards = independent_data[task]['rewards']
            last_100 = rewards[-100:] if len(rewards) >= 100 else rewards
            avg_reward = np.mean(last_100)
            params = independent_data[task]['parameters']

            ax.scatter(params, avg_reward, s=200, color=task_colors[task],
                      marker='o', alpha=0.7, edgecolor='black', linewidth=2,
                      label=f'Independent - {task.title()}')
            ax.annotate(f'Ind-{task.title()}\n{avg_reward:.1f}',
                       (params, avg_reward), xytext=(10, 10),
                       textcoords='offset points', fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=task_colors[task], alpha=0.3))

    # Shared DQN points
    if shared_data:
        params = shared_data['parameters']
        for task in tasks:
            task_episodes = [ep for ep in shared_data['episodes'] if ep['task'] == task]
            last_100 = task_episodes[-100:]
            avg_reward = np.mean([ep['reward'] for ep in last_100])

            ax.scatter(params, avg_reward, s=200, color=task_colors[task],
                      marker='s', alpha=0.7, edgecolor='black', linewidth=2,
                      label=f'Shared - {task.title()}')
            ax.annotate(f'Shared-{task.title()}\n{avg_reward:.1f}',
                       (params, avg_reward), xytext=(-60, -15),
                       textcoords='offset points', fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=task_colors[task], alpha=0.3))

    # Add efficiency lines (reward per 1K params)
    x_range = np.linspace(35000, 110000, 100)
    for efficiency in [1, 2, 3, 4, 5]:
        y = (x_range / 1000) * efficiency
        ax.plot(x_range, y, '--', alpha=0.2, color='gray', linewidth=1)
        ax.text(105000, (105000/1000)*efficiency, f'{efficiency}r/1Kp',
               fontsize=7, alpha=0.5)

    ax.set_xlabel('Total Parameters', fontweight='bold')
    ax.set_ylabel('Average Reward', fontweight='bold')
    ax.set_title('Parameter Efficiency: Performance vs Model Size',
                fontweight='bold', pad=15)
    ax.grid(alpha=0.3)

    # Custom legend
    legend_elements = [
        mpatches.Patch(color=COLOR_INDEPENDENT, label='Independent DQN (35.7K × 3 = 107K params)'),
        mpatches.Patch(color=COLOR_SHARED, label='Shared DQN (37.8K params, 65% reduction)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)

    plt.tight_layout()
    output_path = output_dir / 'comparison_2_parameter_efficiency.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Saved: {output_path.name}")
    plt.close()


def plot_3_training_curves_comparison(independent_data, shared_data, output_dir):
    """
    Plot 3: Training curves comparison (3 subplots for 3 tasks).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    tasks = ['standard', 'windy', 'heavy']
    task_labels = ['Standard', 'Windy', 'Heavy Weight']

    for idx, (task, task_label) in enumerate(zip(tasks, task_labels)):
        ax = axes[idx]

        # Independent DQN
        if task in independent_data:
            rewards = independent_data[task]['rewards']
            # Smooth with rolling average
            window = 50
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            episodes = np.arange(len(smoothed))

            ax.plot(episodes, smoothed, label='Independent DQN',
                   color=COLOR_INDEPENDENT, linewidth=2, alpha=0.8)

        # Shared DQN
        if shared_data:
            task_episodes = [ep for ep in shared_data['episodes'] if ep['task'] == task]
            rewards = [ep['reward'] for ep in task_episodes]

            # Smooth with rolling average
            window = 50
            if len(rewards) >= window:
                smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                episodes = np.arange(len(smoothed))

                ax.plot(episodes, smoothed, label='Shared DQN',
                       color=COLOR_SHARED, linewidth=2, alpha=0.8)

        ax.set_xlabel('Episode', fontweight='bold')
        ax.set_ylabel('Reward (50-ep MA)', fontweight='bold')
        ax.set_title(f'{task_label} Task', fontweight='bold')
        ax.legend(loc='lower right', framealpha=0.9)
        ax.grid(alpha=0.3)
        ax.axhline(y=200, color='green', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(0.02, 0.98, 'Success: 200', transform=ax.transAxes,
               verticalalignment='top', fontsize=8, color='green')

    plt.suptitle('Training Progress Comparison', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    output_path = output_dir / 'comparison_3_training_curves.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Saved: {output_path.name}")
    plt.close()


def plot_4_comprehensive_summary(independent_data, shared_data, output_dir):
    """
    Plot 4: Comprehensive 2x2 summary dashboard.
    """
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    tasks = ['standard', 'windy', 'heavy']
    task_labels = ['Standard', 'Windy', 'Heavy']

    # ------ Subplot 1: Performance Bar Chart ------
    ax1 = fig.add_subplot(gs[0, 0])

    independent_scores = []
    shared_scores = []

    for task in tasks:
        if task in independent_data:
            rewards = independent_data[task]['rewards']
            last_100 = rewards[-100:] if len(rewards) >= 100 else rewards
            independent_scores.append(np.mean(last_100))
        else:
            independent_scores.append(0)

        if shared_data:
            task_episodes = [ep for ep in shared_data['episodes'] if ep['task'] == task]
            last_100 = task_episodes[-100:]
            shared_scores.append(np.mean([ep['reward'] for ep in last_100]))
        else:
            shared_scores.append(0)

    x = np.arange(len(tasks))
    width = 0.35

    ax1.bar(x - width/2, independent_scores, width, label='Independent',
           color=COLOR_INDEPENDENT, alpha=0.8, edgecolor='black')
    ax1.bar(x + width/2, shared_scores, width, label='Shared',
           color=COLOR_SHARED, alpha=0.8, edgecolor='black')

    ax1.set_ylabel('Avg Reward', fontweight='bold')
    ax1.set_title('A) Final Performance Comparison', fontweight='bold', loc='left')
    ax1.set_xticks(x)
    ax1.set_xticklabels(task_labels)
    ax1.legend(framealpha=0.9)
    ax1.grid(axis='y', alpha=0.3)

    # ------ Subplot 2: Parameter Count ------
    ax2 = fig.add_subplot(gs[0, 1])

    methods = ['Independent\nDQN', 'Shared\nDQN']
    params = [107148, 37788]
    colors_bar = [COLOR_INDEPENDENT, COLOR_SHARED]

    bars = ax2.bar(methods, params, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=1.5)

    for bar, param in zip(bars, params):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{param/1000:.1f}K',
                ha='center', va='bottom', fontweight='bold')

    # Add efficiency annotation
    ax2.text(0.5, 80000, '65% Reduction', ha='center', fontsize=11,
            color='green', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3))

    ax2.set_ylabel('Total Parameters', fontweight='bold')
    ax2.set_title('B) Model Size Comparison', fontweight='bold', loc='left')
    ax2.grid(axis='y', alpha=0.3)

    # ------ Subplot 3: Average Reward Across Tasks ------
    ax3 = fig.add_subplot(gs[1, 0])

    avg_independent = np.mean(independent_scores)
    avg_shared = np.mean(shared_scores)

    methods = ['Independent\nDQN', 'Shared\nDQN']
    avg_rewards = [avg_independent, avg_shared]

    bars = ax3.bar(methods, avg_rewards, color=[COLOR_INDEPENDENT, COLOR_SHARED],
                   alpha=0.8, edgecolor='black', linewidth=1.5)

    for bar, reward in zip(bars, avg_rewards):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{reward:.1f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Add improvement
    improvement = ((avg_shared - avg_independent) / avg_independent) * 100
    ax3.text(0.5, max(avg_rewards) - 20, f'+{improvement:.1f}%', ha='center', fontsize=12,
            color='green', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3))

    ax3.set_ylabel('Average Reward', fontweight='bold')
    ax3.set_title('C) Overall Performance', fontweight='bold', loc='left')
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim([0, max(avg_rewards) * 1.2])

    # ------ Subplot 4: Parameter Efficiency ------
    ax4 = fig.add_subplot(gs[1, 1])

    # Calculate reward per 1K params
    ind_efficiency = avg_independent / (107148 / 1000)
    shared_efficiency = avg_shared / (37788 / 1000)

    methods = ['Independent\nDQN', 'Shared\nDQN']
    efficiencies = [ind_efficiency, shared_efficiency]

    bars = ax4.bar(methods, efficiencies, color=[COLOR_INDEPENDENT, COLOR_SHARED],
                   alpha=0.8, edgecolor='black', linewidth=1.5)

    for bar, eff in zip(bars, efficiencies):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{eff:.2f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Add ratio
    ratio = shared_efficiency / ind_efficiency
    ax4.text(0.5, max(efficiencies) - 0.5, f'{ratio:.1f}× Better', ha='center', fontsize=12,
            color='green', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3))

    ax4.set_ylabel('Reward per 1K Parameters', fontweight='bold')
    ax4.set_title('D) Parameter Efficiency', fontweight='bold', loc='left')
    ax4.grid(axis='y', alpha=0.3)
    ax4.set_ylim([0, max(efficiencies) * 1.3])

    plt.suptitle('Multi-Task RL: Independent vs Shared DQN Summary',
                fontweight='bold', fontsize=16, y=0.98)

    plt.tight_layout()
    output_path = output_dir / 'comparison_4_comprehensive_summary.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Saved: {output_path.name}")
    plt.close()


def main():
    """Generate all comparison plots."""
    print("\n" + "="*70)
    print("GENERATING COMPARISON PLOTS: INDEPENDENT DQN vs SHARED DQN")
    print("="*70 + "\n")

    # Load data
    print("📂 Loading experimental data...")
    independent_data, shared_data = load_data()

    if not independent_data:
        print("❌ ERROR: No Independent DQN data found")
        return

    if not shared_data:
        print("❌ ERROR: No Shared DQN data found")
        return

    print(f"✓ Loaded Independent DQN: {len(independent_data)} tasks")
    print(f"✓ Loaded Shared DQN: {len(shared_data['episodes'])} episodes\n")

    # Create output directory
    output_dir = Path(__file__).parent / 'results' / 'analysis' / 'comparisons'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("🎨 Generating comparison plots...\n")

    print("[1/4] Performance comparison...")
    plot_1_performance_comparison(independent_data, shared_data, output_dir)

    print("[2/4] Parameter efficiency...")
    plot_2_parameter_efficiency(independent_data, shared_data, output_dir)

    print("[3/4] Training curves...")
    plot_3_training_curves_comparison(independent_data, shared_data, output_dir)

    print("[4/4] Comprehensive summary dashboard...")
    plot_4_comprehensive_summary(independent_data, shared_data, output_dir)

    print("\n" + "="*70)
    print("✅ ALL COMPARISON PLOTS GENERATED")
    print(f"📁 Output directory: {output_dir}")
    print("="*70 + "\n")

    # List generated files
    print("Generated files:")
    for file in sorted(output_dir.glob('comparison_*.png')):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()
