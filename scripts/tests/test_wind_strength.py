"""Quick test to find reasonable wind power for Windy LunarLander."""

import numpy as np
from environments.lunar_lander_variants import WindyLunarLander

def test_wind_power(wind_power, num_episodes=10):
    """Test a specific wind power with random agent."""
    env = WindyLunarLander(wind_power=wind_power, render_mode=None)

    episode_lengths = []
    episode_rewards = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        steps = 0
        reward_sum = 0

        while not (done or truncated) and steps < 500:  # 500 step limit
            action = env.action_space.sample()  # Random actions
            state, reward, done, truncated, _ = env.step(action)
            steps += 1
            reward_sum += reward

        episode_lengths.append(steps)
        episode_rewards.append(reward_sum)

    env.close()

    return {
        'wind_power': wind_power,
        'mean_length': np.mean(episode_lengths),
        'mean_reward': np.mean(episode_rewards),
        'timeout_rate': sum(1 for l in episode_lengths if l >= 500) / num_episodes
    }


if __name__ == "__main__":
    print("="*70)
    print("TESTING DIFFERENT WIND STRENGTHS (Random Agent)")
    print("="*70)
    print("\nThis tests how different wind powers affect episode length/rewards")
    print("with a RANDOM agent (no learning). Lower wind = episodes end faster.\n")

    # Test different wind powers
    wind_powers = [5.0, 10.0, 15.0, 20.0, 30.0]

    results = []
    for wind_power in wind_powers:
        print(f"Testing wind_power={wind_power}...", end=" ")
        result = test_wind_power(wind_power, num_episodes=10)
        results.append(result)
        print(f"Done. Avg steps: {result['mean_length']:.0f}, Timeout: {result['timeout_rate']*100:.0f}%")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Wind Power':<15} {'Avg Steps':<15} {'Avg Reward':<15} {'Timeout %':<15}")
    print("-"*70)
    for r in results:
        print(f"{r['wind_power']:<15.1f} {r['mean_length']:<15.0f} "
              f"{r['mean_reward']:<15.1f} {r['timeout_rate']*100:<15.0f}")

    print("\n" + "="*70)
    print("RECOMMENDATION:")
    print("="*70)
    print("If timeout rate is HIGH with random agent, it will be MUCH WORSE")
    print("with a trained agent that learns to avoid crashing.")
    print()
    print("Suggested wind_power range: 10-15 (challenging but learnable)")
    print("="*70)
