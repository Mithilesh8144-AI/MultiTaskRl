"""Quick test script to verify BRC implementation works correctly."""

import sys
from pathlib import Path
import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from environments.lunar_lander_variants import make_env
from agents.brc import BRCAgent, MultiTaskReplayBuffer

print("\n" + "="*70)
print("BRC IMPLEMENTATION TEST")
print("="*70)

# Test 1: Create agent
print("\n[1/5] Creating BRC agent...")
agent = BRCAgent(
    state_dim=8,
    num_actions=4,
    num_tasks=3,
    hidden_dim=256,
    num_blocks=3,
    embed_dim=32,
    num_atoms=51,
    v_min=-100.0,
    v_max=300.0,
    gamma=0.99,
    learning_rate=3e-4,
    weight_decay=1e-4,
    device='cpu'
)
print("✅ Agent created successfully")

# Test 2: Forward pass
print("\n[2/5] Testing forward pass...")
state = torch.randn(4, 8)  # Batch of 4 states
task_ids = torch.LongTensor([0, 1, 2, 0])  # Mix of tasks

logits = agent.q_network(state, task_ids)
q_values = agent.q_network.get_q_values(state, task_ids, agent.support)

print(f"✅ Forward pass successful")
print(f"   Logits shape: {logits.shape} (should be [4, 4, 51])")
print(f"   Q-values shape: {q_values.shape} (should be [4, 4])")
assert logits.shape == (4, 4, 51), "Incorrect logits shape"
assert q_values.shape == (4, 4), "Incorrect Q-values shape"

# Test 3: Action selection
print("\n[3/5] Testing action selection...")
state_np = np.random.randn(8)
action = agent.select_action(state_np, task_id=0, epsilon=0.0)
print(f"✅ Action selection works: action={action}")
assert 0 <= action < 4, "Invalid action"

# Test 4: Replay buffer
print("\n[4/5] Testing replay buffer...")
buffer = MultiTaskReplayBuffer(capacity=1000)

for i in range(100):
    state = np.random.randn(8)
    action = np.random.randint(0, 4)
    reward = np.random.randn()
    next_state = np.random.randn(8)
    done = False
    task_id = i % 3
    buffer.push(state, action, reward, next_state, done, task_id)

print(f"✅ Buffer populated: {len(buffer)} transitions")

batch = buffer.sample(32, device='cpu')
states, actions, rewards, next_states, dones, task_ids = batch
print(f"   Batch shapes:")
print(f"      states: {states.shape}")
print(f"      actions: {actions.shape}")
print(f"      task_ids: {task_ids.shape}")

# Test 5: Update step
print("\n[5/5] Testing update step...")
loss = agent.update(states, actions, rewards, next_states, dones, task_ids)
print(f"✅ Update successful: loss={loss:.4f}")
assert loss >= 0, "Loss should be non-negative"

# Test 6: Environment interaction
print("\n[6/6] Testing environment interaction (10 steps)...")
env = make_env('standard')
state, _ = env.reset()

for step in range(10):
    action = agent.select_action(state, task_id=0, epsilon=0.5)
    next_state, reward, done, truncated, _ = env.step(action)

    buffer.push(state, action, reward, next_state, done, task_id=0)

    if len(buffer) >= 32:
        batch = buffer.sample(32, device='cpu')
        loss = agent.update(*batch)

    state = next_state
    if done or truncated:
        state, _ = env.reset()

print(f"✅ Environment interaction works")

env.close()

# Test 7: Save/Load
print("\n[7/7] Testing save/load...")
test_path = project_root / 'test_brc_checkpoint.pth'
agent.save(str(test_path))
print(f"✅ Model saved to {test_path}")

# Create new agent and load
agent2 = BRCAgent(
    state_dim=8,
    num_actions=4,
    num_tasks=3,
    hidden_dim=256,
    num_blocks=3,
    embed_dim=32,
    num_atoms=51,
    v_min=-100.0,
    v_max=300.0,
    device='cpu'
)
agent2.load(str(test_path))
print(f"✅ Model loaded successfully")

# Verify loaded model produces same output
with torch.no_grad():
    test_state = torch.randn(1, 8)
    test_task = torch.LongTensor([0])

    q1 = agent.q_network.get_q_values(test_state, test_task, agent.support)
    q2 = agent2.q_network.get_q_values(test_state, test_task, agent2.support)

    assert torch.allclose(q1, q2, atol=1e-5), "Loaded model produces different output!"

print(f"✅ Save/load verification passed")

# Clean up test file
test_path.unlink()

print("\n" + "="*70)
print("✅ ALL TESTS PASSED!")
print("="*70)
print("\nBRC implementation is ready for training.")
print("Run: python -m experiments.brc.train")
print("="*70 + "\n")
