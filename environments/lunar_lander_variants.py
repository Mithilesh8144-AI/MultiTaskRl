"""
Lunar Lander Environment Variants for Multi-Task RL

Three task variants:
1. Standard - Unchanged LunarLander-v2 baseline
2. Windy - Random lateral wind force applied each step
3. HeavyWeight - Increased gravity/mass making lander heavier
"""

import numpy as np
import gymnasium as gym
from gymnasium.envs.box2d.lunar_lander import LunarLander


class StandardLunarLander(LunarLander):
    """
    Standard Lunar Lander - unchanged baseline.
    Wrapper around the original LunarLander-v2 for consistency.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task_name = "Standard"

    def reset(self, **kwargs):
        return super().reset(**kwargs)

    def step(self, action):
        return super().step(action)


class WindyLunarLander(LunarLander):
    """
    Windy Lunar Lander - adds random lateral wind force each step.

    The wind applies a horizontal force to the lander, making the task
    more challenging and requiring the agent to compensate for drift.
    """

    def __init__(self, wind_power=20.0, **kwargs):
        """
        Args:
            wind_power (float): Maximum magnitude of wind force (symmetric ±wind_power)
        """
        super().__init__(**kwargs)
        self.wind_power = wind_power
        self.task_name = "Windy"

    def reset(self, **kwargs):
        return super().reset(**kwargs)

    def step(self, action):
        # Apply random lateral wind force before taking the step
        if self.lander is not None:
            wind_force_x = np.random.uniform(-self.wind_power, self.wind_power)
            # Apply force to center of lander (force, wake=True)
            self.lander.ApplyForceToCenter((wind_force_x, 0.0), True)

        # Execute normal step
        return super().step(action)


class HeavyWeightLunarLander(LunarLander):
    """
    Heavy Weight Lunar Lander - increased gravity making lander heavier.

    This variant increases the gravitational pull, requiring the agent
    to use more thrust to control descent and landing.
    """

    def __init__(self, gravity_multiplier=1.5, **kwargs):
        """
        Args:
            gravity_multiplier (float): Multiplier for gravity (standard is -10.0)
        """
        self.gravity_multiplier = gravity_multiplier
        super().__init__(**kwargs)
        self.task_name = "HeavyWeight"

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        # Increase gravity after environment is initialized
        if self.world is not None:
            # Standard gravity is (0, -10), we multiply the y-component
            self.world.gravity = (0, -10.0 * self.gravity_multiplier)
        return result

    def step(self, action):
        return super().step(action)


def make_env(task_name, **kwargs):
    """
    Factory function to create environment by name.

    Args:
        task_name (str): One of 'standard', 'windy', 'heavy'
        **kwargs: Additional arguments passed to environment constructor

    Returns:
        LunarLander environment instance
    """
    task_name = task_name.lower()

    if task_name == 'standard':
        return StandardLunarLander(**kwargs)
    elif task_name == 'windy':
        return WindyLunarLander(**kwargs)
    elif task_name in ['heavy', 'heavyweight', 'heavy_weight']:
        return HeavyWeightLunarLander(**kwargs)
    else:
        raise ValueError(f"Unknown task name: {task_name}. Choose from: standard, windy, heavy")


def get_all_tasks():
    """
    Returns a list of all available task names.
    """
    return ['standard', 'windy', 'heavy']


def test_environment(env, num_episodes=5):
    """
    Test an environment with random actions to verify it works.

    Args:
        env: Gymnasium environment
        num_episodes (int): Number of episodes to run

    Returns:
        dict: Statistics about the test runs
    """
    print(f"\n{'='*60}")
    print(f"Testing {env.task_name} Lunar Lander")
    print(f"{'='*60}")

    episode_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0
        steps = 0
        done = False
        truncated = False

        while not (done or truncated):
            action = env.action_space.sample()  # Random action
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if steps > 1000:  # Safety limit
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Steps = {steps}")

    stats = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards)
    }

    print(f"\nSummary Statistics:")
    print(f"  Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"  Reward Range: [{stats['min_reward']:.2f}, {stats['max_reward']:.2f}]")
    print(f"  Mean Episode Length: {stats['mean_length']:.1f} steps")

    return stats


if __name__ == "__main__":
    """
    Test all three environment variants to ensure they work correctly.
    """
    print("\n" + "="*60)
    print("LUNAR LANDER VARIANTS - ENVIRONMENT TEST")
    print("="*60)

    # Test each environment variant
    tasks = get_all_tasks()
    all_stats = {}

    for task_name in tasks:
        env = make_env(task_name, render_mode=None)
        stats = test_environment(env, num_episodes=5)
        all_stats[task_name] = stats
        env.close()

    # Print comparative summary
    print("\n" + "="*60)
    print("COMPARATIVE SUMMARY")
    print("="*60)
    print(f"{'Task':<15} {'Mean Reward':<15} {'Std Reward':<15} {'Mean Steps':<15}")
    print("-"*60)
    for task_name in tasks:
        stats = all_stats[task_name]
        print(f"{task_name.capitalize():<15} "
              f"{stats['mean_reward']:<15.2f} "
              f"{stats['std_reward']:<15.2f} "
              f"{stats['mean_length']:<15.1f}")

    print("\n" + "="*60)
    print("All environments tested successfully!")
    print("="*60)
