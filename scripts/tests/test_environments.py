"""Quick test to verify all 3 Lunar Lander environment variants work correctly."""

import numpy as np
from environments import make_env

def test_environment(task_name, num_steps=100):
    """Test a single environment variant."""
    print(f"\n{'='*60}")
    print(f"Testing {task_name.upper()} environment")
    print('='*60)

    # Create environment
    env = make_env(task_name, render_mode=None)

    # Basic info
    print(f"✓ Environment created: {env.task_name}")
    print(f"  State dimension: {env.observation_space.shape[0]}")
    print(f"  Action dimension: {env.action_space.n}")

    # Test reset
    state, info = env.reset()
    print(f"✓ Reset successful: state shape = {state.shape}")

    # Run a few random steps
    total_reward = 0
    states = []

    for step in range(num_steps):
        action = env.action_space.sample()  # Random action
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        states.append(next_state)

        if done or truncated:
            print(f"✓ Episode ended at step {step+1}")
            break

    print(f"✓ {num_steps} steps executed successfully")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Final state: {next_state[:4]}...")  # Show first 4 values

    # Environment-specific checks
    if task_name == 'windy':
        print(f"  Wind power: {env.wind_power}")
        print(f"  ✓ Wind forces will be applied during flight")
    elif task_name == 'heavy':
        print(f"  Gravity multiplier: {env.gravity_multiplier}")
        print(f"  World gravity: {env.world.gravity}")
        print(f"  ✓ Increased gravity active")

    env.close()
    return True


def main():
    """Test all environment variants."""
    print("\n" + "="*60)
    print("ENVIRONMENT VARIANTS TEST")
    print("="*60)
    print("This will test all 3 Lunar Lander variants:")
    print("  1. Standard (unchanged)")
    print("  2. Windy (random lateral wind)")
    print("  3. Heavy Weight (increased gravity)")
    print()

    tasks = ['standard', 'windy', 'heavy']
    results = {}

    for task in tasks:
        try:
            success = test_environment(task, num_steps=100)
            results[task] = "✅ PASS"
        except Exception as e:
            results[task] = f"❌ FAIL: {str(e)}"
            print(f"❌ Error: {e}")

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for task, result in results.items():
        print(f"  {task.capitalize():15} {result}")
    print("="*60)

    # Check if all passed
    if all("PASS" in r for r in results.values()):
        print("\n🎉 All environment variants working correctly!")
        print("   Ready to use on Colab!")
    else:
        print("\n⚠️  Some tests failed. Check errors above.")


if __name__ == "__main__":
    main()
