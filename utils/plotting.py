"""
Plotting Utilities for Live Training Visualization

This module provides helpers for creating live, updating plots during training.
Perfect for Jupyter notebooks where you want to see the agent's progress in real-time.

Key Features:
=============
- Live updating plots (rewards, loss, epsilon)
- Smoothing for noisy RL curves
- Save plots to disk
- Configurable styling
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Tuple
from IPython import display


class LivePlotter:
    """
    Create and update plots in real-time during training.

    Designed for use in Jupyter notebooks with %matplotlib inline or %matplotlib notebook.
    """

    def __init__(self, figsize: Tuple[int, int] = (15, 5), num_plots: int = 3):
        """
        Initialize the live plotter.

        Args:
            figsize: Figure size (width, height) in inches
            num_plots: Number of subplots to create
        """
        self.fig, self.axes = plt.subplots(1, num_plots, figsize=figsize)
        if num_plots == 1:
            self.axes = [self.axes]  # Make it a list for consistency

        # Storage for plot data
        self.data = {i: {'x': [], 'y': []} for i in range(num_plots)}

        # Configure plot aesthetics
        plt.style.use('seaborn-v0_8-darkgrid')
        self.fig.tight_layout(pad=3.0)

    def update(
        self,
        plot_idx: int,
        x: float,
        y: float,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        color: str = 'blue',
        smooth: bool = False,
        smooth_window: int = 10
    ) -> None:
        """
        Update a specific subplot with new data point.

        Args:
            plot_idx: Index of subplot to update (0, 1, 2, ...)
            x: X-axis value (e.g., episode number)
            y: Y-axis value (e.g., reward)
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            color: Line color
            smooth: Whether to plot smoothed curve
            smooth_window: Window size for moving average
        """
        # Append new data
        self.data[plot_idx]['x'].append(x)
        self.data[plot_idx]['y'].append(y)

        # Get the axis
        ax = self.axes[plot_idx]
        ax.clear()

        # Plot raw data
        ax.plot(
            self.data[plot_idx]['x'],
            self.data[plot_idx]['y'],
            color=color,
            alpha=0.3,
            linewidth=1,
            label='Raw'
        )

        # Plot smoothed data if requested
        if smooth and len(self.data[plot_idx]['y']) >= smooth_window:
            y_smooth = self._moving_average(
                self.data[plot_idx]['y'],
                window=smooth_window
            )
            x_smooth = self.data[plot_idx]['x'][smooth_window-1:]
            ax.plot(
                x_smooth,
                y_smooth,
                color=color,
                linewidth=2,
                label=f'Smooth (window={smooth_window})'
            )
            ax.legend(loc='best')

        # Labels
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3)

        # Update display (for Jupyter)
        display.clear_output(wait=True)
        display.display(self.fig)

    def update_multiple(
        self,
        episode: int,
        reward: float,
        loss: Optional[float] = None,
        epsilon: Optional[float] = None,
        smooth: bool = True
    ) -> None:
        """
        Convenience method to update all three standard plots at once.

        Args:
            episode: Current episode number
            reward: Episode reward
            loss: Training loss (optional)
            epsilon: Exploration epsilon (optional)
            smooth: Whether to show smoothed curves
        """
        # Plot 1: Rewards
        self.update(
            0, episode, reward,
            title='Episode Rewards',
            xlabel='Episode',
            ylabel='Reward',
            color='green',
            smooth=smooth,
            smooth_window=20
        )

        # Plot 2: Loss (if provided)
        if loss is not None:
            self.update(
                1, episode, loss,
                title='Training Loss',
                xlabel='Episode',
                ylabel='Loss',
                color='red',
                smooth=smooth,
                smooth_window=20
            )

        # Plot 3: Epsilon (if provided)
        if epsilon is not None:
            self.update(
                2, episode, epsilon,
                title='Exploration (Epsilon)',
                xlabel='Episode',
                ylabel='Epsilon',
                color='blue',
                smooth=False  # Epsilon doesn't need smoothing
            )

    @staticmethod
    def _moving_average(data: List[float], window: int = 10) -> np.ndarray:
        """
        Compute moving average for smoothing.

        Args:
            data: List of values
            window: Window size for averaging

        Returns:
            Smoothed array
        """
        weights = np.ones(window) / window
        return np.convolve(data, weights, mode='valid')

    def save(self, filepath: str) -> None:
        """
        Save current figure to disk.

        Args:
            filepath: Path to save file (e.g., 'results/plots/training.png')
        """
        self.fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {filepath}")

    def close(self) -> None:
        """Close the figure."""
        plt.close(self.fig)


def plot_comparison(
    results_dict: dict,
    metric: str = 'reward',
    title: str = 'Method Comparison',
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Plot comparison of multiple methods on same axes.

    Useful for comparing Independent DQN vs Shared DQN vs PCGrad, etc.

    Args:
        results_dict: Dictionary mapping method names to reward lists
            Example: {'Independent DQN': [rewards...], 'Shared DQN': [rewards...]}
        metric: What metric is being plotted (for ylabel)
        title: Plot title
        figsize: Figure size

    Example:
        >>> results = {
        ...     'Independent': independent_rewards,
        ...     'Shared': shared_rewards,
        ...     'PCGrad': pcgrad_rewards
        ... }
        >>> plot_comparison(results, title='Multi-Task Baselines')
    """
    plt.figure(figsize=figsize)

    colors = ['blue', 'red', 'green', 'purple', 'orange']

    for idx, (method, values) in enumerate(results_dict.items()):
        episodes = list(range(len(values)))
        plt.plot(
            episodes,
            values,
            label=method,
            color=colors[idx % len(colors)],
            linewidth=2,
            alpha=0.7
        )

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel(metric.capitalize(), fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_per_task_comparison(
    task_names: List[str],
    method_results: dict,
    title: str = 'Per-Task Performance',
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Bar chart comparing different methods on each task.

    Args:
        task_names: List of task names ['Standard', 'Windy', 'Heavy']
        method_results: Dict mapping method name to list of per-task rewards
            Example: {'Independent': [200, 180, 150], 'Shared': [150, 100, 50]}
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(task_names))
    width = 0.25
    multiplier = 0

    for method, rewards in method_results.items():
        offset = width * multiplier
        ax.bar(x + offset, rewards, width, label=method, alpha=0.8)
        multiplier += 1

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('Final Reward', fontsize=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels(task_names)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    print("Plotting utilities loaded successfully!")
    print("\nExample usage in a Jupyter notebook:")
    print("""
    from utils.plotting import LivePlotter

    # Create live plotter
    plotter = LivePlotter(figsize=(15, 5), num_plots=3)

    # In training loop:
    for episode in range(1000):
        reward = train_one_episode(...)
        loss = ...
        epsilon = ...

        # Update plots
        plotter.update_multiple(episode, reward, loss, epsilon)

    # Save final plot
    plotter.save('results/plots/training_curves.png')
    """)
