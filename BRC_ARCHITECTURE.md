# BRC Architecture Reference

**Bigger, Regularized, Categorical DQN for Multi-Task Reinforcement Learning**

**Created:** 2026-01-07
**Purpose:** Detailed technical reference for BRC implementation

---

## Table of Contents
1. [Overview](#overview)
2. [Architecture Components](#architecture-components)
3. [Categorical DQN Mechanics](#categorical-dqn-mechanics)
4. [Training Procedure](#training-procedure)
5. [Parameter Breakdown](#parameter-breakdown)
6. [Design Choices](#design-choices)
7. [Code Examples](#code-examples)

---

## Overview

### What is BRC?

BRC is a multi-task reinforcement learning baseline that tests the hypothesis:

> **"Can large networks absorb gradient conflicts without special gradient manipulation techniques?"**

The name BRC stands for:
- **B**igger: 459,820 parameters (12.2× Shared DQN, 4.3× Independent DQN)
- **R**egularized: AdamW optimizer with weight decay (L2 penalty)
- **C**ategorical: C51-style distributional RL with 51 atoms

### Key Innovation

Instead of using gradient manipulation (PCGrad, GradNorm, CAGrad), BRC uses:
1. **High-capacity networks** to learn task-specific features in different subspaces
2. **Residual connections** to prevent vanishing gradients in deep networks
3. **Distributional RL** for more stable learning (cross-entropy loss vs MSE)
4. **Regularization** to prevent overfitting to any single task

---

## Architecture Components

### 1. ResidualBlock

**Purpose:** Building block for deep networks with skip connections

```
Input (256-dim)
    |
    ├─────────────────────┐ (skip connection)
    |                     |
LayerNorm                 |
    |                     |
Linear(256 → 256)         |
    |                     |
ReLU                      |
    |                     |
Linear(256 → 256)         |
    |                     |
    +─────────────────────┘ (element-wise addition)
    |
Output (256-dim)
```

**Code:**
```python
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.ln(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x + residual  # Skip connection prevents vanishing gradients
```

**Why Skip Connections?**
- Allow gradients to flow directly backward through the network
- Enable training of very deep networks (we use 3 blocks)
- Help preserve both shared and task-specific features

**Why LayerNorm (not BatchNorm)?**
- RL data is non-stationary (distribution changes during training)
- BatchNorm assumes i.i.d. data (violates RL assumptions)
- LayerNorm normalizes per-sample, more stable for RL

---

### 2. BroNet (The Full Network)

**Purpose:** High-capacity multi-task Q-network

```
State (8-dim) ──┐
                ├─> Concatenate ──> Linear(40 → 256) ──> ReLU
Task ID (0/1/2) ─> Embedding(3 → 32-dim)                  |
                                                           |
                                                    ResidualBlock 1
                                                           |
                                                    ResidualBlock 2
                                                           |
                                                    ResidualBlock 3
                                                           |
                                                      LayerNorm
                                                           |
                                              Linear(256 → 204)  [4 actions × 51 atoms]
                                                           |
                                                    Reshape to (4, 51)
                                                           |
                                                      Softmax (dim=-1)
                                                           |
                                          Output: Distribution over returns per action
```

**Code:**
```python
class BroNet(nn.Module):
    def __init__(self, state_dim=8, num_actions=4, num_tasks=3,
                 embed_dim=32, hidden_dim=256, num_blocks=3, num_atoms=51):
        super().__init__()

        # Task embedding: 3 tasks × 32-dim = 96 parameters
        self.task_embedding = nn.Embedding(num_tasks, embed_dim)

        # Input layer: (8 + 32) × 256 + 256 = 10,496 parameters
        self.input_layer = nn.Linear(state_dim + embed_dim, hidden_dim)

        # Residual blocks: 3 × 132,096 = 396,288 parameters
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])

        # Final normalization: 512 parameters
        self.final_ln = nn.LayerNorm(hidden_dim)

        # Output layer: 256 × 204 + 204 = 52,428 parameters
        self.output_layer = nn.Linear(hidden_dim, num_actions * num_atoms)

        self.num_actions = num_actions
        self.num_atoms = num_atoms

    def forward(self, state: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        # Get task embedding: [batch_size, 32]
        task_emb = self.task_embedding(task_ids)

        # Concatenate state + task: [batch_size, 40]
        x = torch.cat([state, task_emb], dim=-1)

        # Input projection: [batch_size, 256]
        x = F.relu(self.input_layer(x))

        # Residual blocks (3 blocks)
        for block in self.res_blocks:
            x = block(x)

        # Final normalization
        x = self.final_ln(x)

        # Output layer: [batch_size, 204]
        logits = self.output_layer(x)

        # Reshape to [batch_size, 4 actions, 51 atoms]
        logits = logits.view(-1, self.num_actions, self.num_atoms)

        # Softmax over atoms (ensures valid probability distribution)
        return F.log_softmax(logits, dim=-1)
```

**Key Points:**
- Task embedding is learned (not fixed)
- Early fusion (concatenate state + task before processing)
- Deep processing through 3 residual blocks
- Output is log-probabilities for numerical stability

---

## Categorical DQN Mechanics

### What is Categorical DQN?

Instead of predicting a single Q-value, categorical DQN predicts a **distribution** over possible returns.

**Standard DQN:**
```
Q(s, a) = 150.5  (single scalar)
```

**Categorical DQN:**
```
Q(s, a) = Distribution over 51 atoms:
  [-100, -92, -84, ..., 0, ..., 292, 300]
    0.0   0.0   0.01  ... 0.3 ... 0.15  0.0
```

### Support Atoms

We discretize the return range into 51 fixed values:

```python
v_min = -100.0  # Worst possible return (crash)
v_max = 300.0   # Best possible return (perfect landing)
num_atoms = 51

support = [-100.0, -92.16, -84.31, ..., 0.0, ..., 292.16, 300.0]
         └─────────────────────────────────────────────────────┘
                            51 atoms
```

**Why 51 atoms?**
- Standard C51 paper uses 51
- Good balance: fine-grained enough to capture variance, not too many parameters
- Delta between atoms: (300 - (-100)) / 50 = 8.0 reward units

### Computing Q-values from Distribution

Given distribution probabilities `p(z)` over atoms `z`:

```
Q(s, a) = Σ z_i * p(z_i)  (expected value)
          i=1..51
```

**Code:**
```python
def get_q_values(self, state, task_ids, support):
    # Get log-probabilities: [batch, actions, atoms]
    log_probs = self.forward(state, task_ids)

    # Convert to probabilities
    probs = log_probs.exp()  # [batch, actions, atoms]

    # Compute expected value: Q = Σ z_i * p(z_i)
    q_values = torch.sum(probs * support, dim=-1)  # [batch, actions]

    return q_values
```

### Loss Function: Cross-Entropy

**Standard DQN uses MSE:**
```
Loss = (Q(s, a) - target)²
```

**Categorical DQN uses Cross-Entropy:**
```
Loss = -Σ p_target(z_i) * log(p_predicted(z_i))
       i=1..51
```

**Why Cross-Entropy?**
- More stable for large networks (no gradient explosion from squared terms)
- Preserves distributional information (not just mean)
- Better uncertainty quantification

---

## Training Procedure

### 1. Distributional Bellman Projection

**The Challenge:** Target distribution doesn't align with our fixed support atoms.

**Example:**
```
Current support: [-100, -92, -84, ..., 300]
Target values:    r + γ * z = [10, 18, 26, ...]  (doesn't match!)
```

**Solution:** Project target onto nearest support atoms using linear interpolation.

**Algorithm:**
```python
def distributional_bellman_projection(
    rewards, next_states, dones, task_ids, gamma, support, delta_z
):
    batch_size = rewards.shape[0]
    num_atoms = support.shape[0]

    # 1. Get next state distribution (target network)
    with torch.no_grad():
        # Get best action from online network (Double DQN)
        next_q_values = q_network.get_q_values(next_states, task_ids, support)
        next_actions = next_q_values.argmax(dim=1)  # [batch_size]

        # Get distribution for best action (target network)
        next_log_probs = target_network(next_states, task_ids)  # [batch, actions, atoms]
        next_probs = next_log_probs.exp()
        next_dist = next_probs[range(batch_size), next_actions]  # [batch, atoms]

        # 2. Compute Bellman update for each atom
        # T_z_j = r + γ * z_j (for each atom)
        Tz = rewards.unsqueeze(1) + gamma * support.unsqueeze(0) * (1 - dones.unsqueeze(1))
        # Shape: [batch_size, num_atoms]

        # 3. Clip to support range
        Tz = Tz.clamp(min=support[0], max=support[-1])

        # 4. Compute projection indices
        # Find which support atom is closest to each Tz value
        b = (Tz - support[0]) / delta_z  # Float indices [0, 50]
        l = b.floor().long()  # Lower bound index
        u = b.ceil().long()   # Upper bound index

        # 5. Distribute probability mass
        m = torch.zeros(batch_size, num_atoms, device=rewards.device)

        for i in range(batch_size):
            for j in range(num_atoms):
                l_idx = l[i, j]
                u_idx = u[i, j]
                prob = next_dist[i, j]

                # Linear interpolation weights
                if l_idx == u_idx:
                    m[i, l_idx] += prob
                else:
                    # Split probability between lower and upper
                    m[i, l_idx] += prob * (u_idx - b[i, j])
                    m[i, u_idx] += prob * (b[i, j] - l_idx)

        return m  # Target distribution [batch_size, num_atoms]
```

**Visual Example:**
```
Support atoms:    [-100, -92, -84, -76, ...]
                     ↑              ↑
Target value: -88 ──┴──────────────┘
              (60% weight to -92, 40% weight to -84)
```

### 2. BRCAgent Update Step

```python
def update(self, states, actions, rewards, next_states, dones, task_ids):
    # 1. Get current distribution for taken actions
    current_log_probs = self.q_network(states, task_ids)  # [batch, actions, atoms]
    current_dist = current_log_probs[range(batch_size), actions]  # [batch, atoms]

    # 2. Compute target distribution (Bellman projection)
    target_dist = distributional_bellman_projection(
        rewards, next_states, dones, task_ids,
        self.gamma, self.support, self.delta_z
    )

    # 3. Cross-entropy loss
    loss = -torch.sum(target_dist * current_dist, dim=-1).mean()

    # 4. Backpropagation
    self.optimizer.zero_grad()
    loss.backward()

    # 5. Gradient clipping (prevent exploding gradients)
    torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)

    # 6. Optimizer step (AdamW with weight decay)
    self.optimizer.step()

    return loss.item()
```

### 3. Multi-Task Training Loop

```python
total_episodes = 1500  # 500 per task
tasks = ['standard', 'windy', 'heavy']

for episode in range(total_episodes):
    # Round-robin task selection
    task_name = tasks[episode % 3]
    task_id = task_to_id[task_name]
    env = envs[task_name]
    max_steps = config['max_episode_steps'][task_name]

    # Episode rollout
    state, _ = env.reset()
    while not (done or truncated):
        action = agent.select_action(state, task_id, epsilon)
        next_state, reward, done, truncated, _ = env.step(action)

        # Store transition with task ID
        buffer.push(state, action, reward, next_state, done, task_id)

        # Update if buffer has enough samples
        if len(buffer) >= min_replay_size:
            batch = buffer.sample(batch_size)
            loss = agent.update(*batch)

        state = next_state

        # Task-specific timeout
        if steps >= max_steps:
            truncated = True

    # Update target network every 10 episodes
    if (episode + 1) % 10 == 0:
        agent.update_target_network()
```

---

## Parameter Breakdown

### Total: 459,820 Parameters

| Component | Calculation | Parameters |
|-----------|-------------|------------|
| **Task Embedding** | 3 tasks × 32 dim | 96 |
| **Input Layer** | (8 + 32) × 256 + 256 | 10,496 |
| **Residual Block 1** | LayerNorm(256) + 2 × Linear(256×256 + 256) | 132,096 |
| **Residual Block 2** | LayerNorm(256) + 2 × Linear(256×256 + 256) | 132,096 |
| **Residual Block 3** | LayerNorm(256) + 2 × Linear(256×256 + 256) | 132,096 |
| **Final LayerNorm** | 2 × 256 (weight + bias) | 512 |
| **Output Layer** | 256 × (4 × 51) + (4 × 51) | 52,428 |
| **TOTAL** | | **459,820** |

### Comparison with Baselines

| Method | Parameters | Multiplier vs BRC |
|--------|------------|-------------------|
| Independent DQN | 107,148 (3 × 35,716) | 0.23× |
| Shared DQN | 37,788 | 0.08× |
| **BRC** | **459,820** | **1.0×** |

**Key Insight:**
- BRC has 12.2× more parameters than Shared DQN
- BRC has 4.3× more parameters than Independent DQN (all 3 networks combined!)
- Hypothesis: Extra capacity allows network to learn task-specific features in different subspaces without explicit gradient manipulation

---

## Design Choices

### 1. Why 32-dim Task Embeddings (vs Shared DQN's 8-dim)?

**Shared DQN:** 8-dim embeddings (minimal capacity)
```python
task_embedding = nn.Embedding(3, 8)  # 24 parameters
```

**BRC:** 32-dim embeddings (high capacity)
```python
task_embedding = nn.Embedding(3, 32)  # 96 parameters
```

**Rationale:**
- Larger embeddings can capture more task-specific nuances
- 32-dim allows network to separate tasks in high-dimensional space
- Each task gets ~10× more "room" to encode differences

### 2. Why 3 Residual Blocks?

**Options:**
- 1 block: Too shallow, limited representation power
- 3 blocks: Sweet spot (verified in BRC paper)
- 5+ blocks: Diminishing returns, harder to train

**Trade-off:**
- More blocks = more capacity (can learn complex features)
- More blocks = more parameters (4.3× Independent DQN is acceptable)
- More blocks = harder optimization (residuals mitigate this)

### 3. Why AdamW with Weight Decay (vs Adam)?

**Adam:** No L2 regularization
```python
optimizer = optim.Adam(params, lr=3e-4)
```

**AdamW:** Decoupled weight decay (the "Regularized" part)
```python
optimizer = optim.AdamW(params, lr=3e-4, weight_decay=1e-4)
```

**Effect:**
- Weight decay = L2 penalty on parameters
- Prevents network from overfitting to any single task
- Encourages shared representations (small weights = simpler model)

### 4. Why 51 Atoms (vs Standard DQN's Single Value)?

**Standard DQN:** Single Q-value per action
```
Q(s, a) = 150.5  (point estimate)
```

**C51 (Categorical DQN):** Distribution over 51 values
```
Q(s, a) ~ Distribution([-100, -92, ..., 300])
```

**Benefits:**
- Captures uncertainty (low-confidence states have wide distributions)
- More stable learning (cross-entropy vs MSE)
- Better for multi-task (different tasks have different return variances)

### 5. Why Task-Specific Timeouts?

**Standard:** 1000 steps (episodes finish naturally in 100-300)
**Windy:** 400 steps (prevents hovering behavior)
**Heavy:** 800 steps (allows full descent with 1.25× gravity)

**Rationale:**
- Different physics = different episode lengths
- Timeout creates "landing urgency" (prevents local optima)
- Fair comparison: each task gets appropriate timeout

---

## Code Examples

### Complete Forward Pass Example

```python
import torch
from agents.brc import BRCAgent

# Create agent
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
    device='cpu'
)

# Example batch
states = torch.randn(32, 8)        # 32 states
task_ids = torch.randint(0, 3, (32,))  # Mix of all tasks

# Get distributions
log_probs = agent.q_network(states, task_ids)
print(f"Log-probs shape: {log_probs.shape}")  # [32, 4, 51]

# Get Q-values
q_values = agent.q_network.get_q_values(states, task_ids, agent.support)
print(f"Q-values shape: {q_values.shape}")  # [32, 4]

# Select actions
actions = q_values.argmax(dim=1)
print(f"Actions: {actions[:5]}")  # [2, 0, 3, 1, 2]
```

### Training Update Example

```python
# Sample batch from replay buffer
batch = replay_buffer.sample(64, device='cpu')
states, actions, rewards, next_states, dones, task_ids = batch

# Update agent
loss = agent.update(states, actions, rewards, next_states, dones, task_ids)
print(f"Loss: {loss:.4f}")  # e.g., 0.0523
```

### Evaluation Example

```python
# Evaluate on specific task
env = make_env('heavy')
state, _ = env.reset()
episode_reward = 0

while not done:
    action = agent.select_action(state, task_id=2, epsilon=0.0)  # Greedy
    state, reward, done, truncated, _ = env.step(action)
    episode_reward += reward

print(f"Heavy task reward: {episode_reward:.2f}")
```

---

## Expected Results

### Hypothesis

**BRC should outperform Shared DQN due to higher capacity:**
- Shared DQN: ~60% degradation vs Independent (gradient conflicts dominate)
- BRC: ~20% degradation vs Independent (capacity absorbs conflicts)

### Performance Prediction

| Method | Avg Reward | Parameters |
|--------|------------|------------|
| Independent DQN | 174.38 | 107,148 |
| Shared DQN | 205.61 | 37,788 |
| **BRC (Expected)** | **~195-200** | **459,820** |

**Key Test:**
- If BRC ≈ Shared DQN: Capacity doesn't help (need PCGrad)
- If BRC ≈ Independent: Capacity solves conflicts (PCGrad unnecessary!)
- If BRC between: Partial success (PCGrad may still help)

---

## Summary

### The BRC Recipe

1. **Bigger Networks**
   - 256 hidden units (vs 128)
   - 3 residual blocks (vs 2 linear layers)
   - 32-dim task embeddings (vs 8-dim)
   - Total: 459,820 parameters (12.2× Shared DQN)

2. **Regularization**
   - AdamW optimizer with weight_decay=1e-4
   - LayerNorm throughout (not BatchNorm)
   - Gradient clipping (max_norm=10.0)
   - Lower learning rate (3e-4 vs 5e-4)

3. **Categorical DQN**
   - 51 atoms over [-100, 300]
   - Distributional Bellman projection
   - Cross-entropy loss (vs MSE)
   - Better uncertainty quantification

4. **Multi-Task Training**
   - Single shared buffer (all tasks mixed)
   - Round-robin task cycling (deterministic)
   - Task-specific timeouts (400/800/1000)
   - Target network updates every 10 episodes

### Files Reference

| File | Lines | Purpose |
|------|-------|---------|
| `agents/brc.py` | 459 | Core implementation (ResidualBlock, BroNet, BRCAgent) |
| `experiments/brc/config.py` | 140 | Hyperparameters and parameter estimation |
| `experiments/brc/train.py` | 347 | Multi-task training loop |
| `experiments/brc/evaluate.py` | 219 | Evaluation script |
| `test_brc.py` | 153 | Test suite (7 tests, all passed ✅) |

### Next Steps

1. Train BRC: `python -m experiments.brc.train` (~3-4 hours)
2. Evaluate: `python -m experiments.brc.evaluate --episodes 20`
3. Analyze: `python -m experiments.analyze_results --method brc`
4. Compare: Update `generate_comparison_plots.py` with BRC results

---

**End of Document**
