# Implementation Details: PCGrad and GradNorm

**Date:** January 19, 2026

---

## Replay Buffer Setup

### IMPORTANT CLARIFICATION

**Both PCGrad and GradNorm use the SAME replay buffer setup:**
- **Per-Task Replay Buffers** for both blind and non-blind versions
- Each task (Standard, Windy, Heavy) has its own separate buffer
- Total capacity: 100K transitions split as ~33K per task
- Minimum size before training: 2K transitions per task

**Why Per-Task Buffers?**
1. **Fair Sampling:** Ensures equal gradient contribution from all tasks
2. **Balanced Training:** Prevents harder tasks (Windy) from dominating the buffer
3. **Gradient Quality:** PCGrad and GradNorm both need clean per-task gradients

**What's Different Between Blind and Non-Blind?**
- **Task Embeddings:** Blind = no embeddings, Non-blind = 8-dim embeddings
- **Network Input:** Blind = state only (8-dim), Non-blind = state + embedding (16-dim)
- **Replay Buffers:** SAME for both (per-task separation)

---

## PCGrad Implementation Summary

### Core Concept
Project conflicting gradients so they don't interfere with each other.

### Key Components

**1. Per-Task Replay Buffer** (`PerTaskReplayBuffer`)
```python
buffers = {
    task_0: deque(maxlen=33333),  # Standard
    task_1: deque(maxlen=33333),  # Windy
    task_2: deque(maxlen=33333),  # Heavy
}
```

**2. PCGrad Optimizer** (`PCGradOptimizer`)
- Wraps a standard Adam optimizer
- Tracks conflict statistics (how often gradients conflict)
- Implements gradient projection algorithm

**3. Gradient Projection**
```
For each pair of tasks (i, j):
  if dot(g_i, g_j) < 0:  # Conflicting gradients
    g_i' = g_i - (g_i · g_j / ||g_j||²) * g_j  # Project away
```
- Random task ordering for fairness
- Only projects when gradients oppose each other (negative dot product)

**4. Update Process**
```python
# 1. Sample batch from each task
batch_0 = replay_buffer.sample(task_id=0, batch_size=64)
batch_1 = replay_buffer.sample(task_id=1, batch_size=64)
batch_2 = replay_buffer.sample(task_id=2, batch_size=64)

# 2. Compute separate loss for each task
loss_0 = compute_task_loss(batch_0, task_id=0)
loss_1 = compute_task_loss(batch_1, task_id=1)
loss_2 = compute_task_loss(batch_2, task_id=2)

# 3. Extract gradients per task (via backward on each loss)
grads_0 = compute_grads(loss_0)
grads_1 = compute_grads(loss_1)
grads_2 = compute_grads(loss_2)

# 4. Project conflicting gradients
projected_grads = pcgrad_project([grads_0, grads_1, grads_2])

# 5. Average projected gradients
final_grad = mean(projected_grads)

# 6. Apply single optimizer step
optimizer.step()
```

**5. Task-Blind vs Task-Aware**
- **Both versions:** Use per-task replay buffers
- **Blind:** Network input = state only (8-dim)
- **Aware:** Network input = state + task_embedding (8 + 8 = 16-dim)
- **Result:** Blind was more stable, Aware catastrophically failed on Heavy task

---

## GradNorm Implementation Summary

### Core Concept
Learn task weights dynamically by balancing gradient magnitudes, accounting for training rates.

### Key Components

**1. Per-Task Replay Buffer** (`PerTaskReplayBuffer`)
```python
# SAME as PCGrad - per-task separation
buffers = {
    task_0: deque(maxlen=33333),
    task_1: deque(maxlen=33333),
    task_2: deque(maxlen=33333),
}
```

**2. Learnable Task Weights** (`GradNormLossWeighter`)
```python
log_weights = nn.Parameter(torch.zeros(3))  # One per task
weights = exp(log_weights) / sum(exp(log_weights))  # Softmax
```
- Stored in log-space for numerical stability
- Normalized via softmax to sum to ~3.0
- Updated via gradient descent

**3. GradNorm Algorithm**
```
1. Compute gradient norms: G_i = ||∇_W L_i|| for each task
2. Compute training rates: r_i = L_i(t) / L_i(0)
3. Compute relative rates: r̃_i = r_i / avg(r)
4. Compute target norms: target_i = avg(G) * (r̃_i)^α
5. Minimize: GradNorm_loss = Σ |G_i - target_i|
6. Update weights to minimize GradNorm_loss
```

**4. The Alpha Parameter**
- α = 1.5 (recommended value)
- Controls asymmetry: slower-learning tasks get higher weights
- α = 0 → equal weighting
- α > 0 → prioritize struggling tasks

**5. Shared Layer Selection**
- Use second-to-last hidden layer (`fc2`)
- This is the "shared representation" layer
- GradNorm balances gradient magnitudes at this layer

**6. Update Process**
```python
# Sample from each task
batch_0 = replay_buffer.sample(task_id=0, batch_size=64)
batch_1 = replay_buffer.sample(task_id=1, batch_size=64)
batch_2 = replay_buffer.sample(task_id=2, batch_size=64)

# Compute per-task losses
loss_0 = compute_task_loss(batch_0, task_id=0)
loss_1 = compute_task_loss(batch_1, task_id=1)
loss_2 = compute_task_loss(batch_2, task_id=2)

# Step A: Update task weights using GradNorm
gradnorm_stats = loss_weighter.update_weights(
    [loss_0, loss_1, loss_2],
    shared_layer=q_network.fc2,
    optimizer=optimizer
)

# Step B: Recompute losses (fresh forward pass)
loss_0 = compute_task_loss(batch_0, task_id=0)
loss_1 = compute_task_loss(batch_1, task_id=1)
loss_2 = compute_task_loss(batch_2, task_id=2)

# Step C: Apply learned weights and update main network
weighted_loss = w_0 * loss_0 + w_1 * loss_1 + w_2 * loss_2
weighted_loss.backward()
optimizer.step()
```

**7. Why Two Forward Passes?**
- First pass: Compute gradients for GradNorm weight update
- Second pass: Fresh computation graph for main network update
- Necessary because GradNorm update frees the computation graph

**8. Task-Blind vs Task-Aware**
- **Both versions:** Use per-task replay buffers (SAME)
- **Blind:** No task embeddings, single shared representation
- **Aware:** 8-dim task embeddings concatenated with state
- **Result:** Blind won decisively (+220 vs -68 average)

---

## Key Differences: PCGrad vs GradNorm

| Aspect | PCGrad | GradNorm |
|--------|--------|----------|
| **Approach** | Geometric (gradient projection) | Optimization (loss weighting) |
| **What it modifies** | Gradient directions | Loss contribution weights |
| **Adaptation** | Static (based on current batch) | Dynamic (learns over time) |
| **Hyperparameters** | None (automatic) | α = 1.5, weight_lr = 0.01 |
| **Computational cost** | O(T²) pairwise projections | 2x forward passes per update |
| **Buffer strategy** | Per-task (separate) | Per-task (separate) |
| **Conflict detection** | Dot product < 0 | Gradient magnitude imbalance |
| **Learned parameters** | None | Task weights (3 scalars) |

---

## Why Task Embeddings Failed

### For PCGrad
- **Too many degrees of freedom:** Network could use embeddings to "route around" gradient conflicts
- **Instability:** Heavy task gradients conflicted severely, causing divergence
- **Optimization landscape:** Gradient projection + task-specific parameters = complex, unstable

### For GradNorm
- **Weight imbalance:** With embeddings, weights went haywire (Standard: 2.66, Windy: 0.009)
- **Feedback loop:** Dominant task got more weight → learned better → got even more weight
- **Specialization:** Task embeddings allowed overfitting to Standard while ignoring others

### Why Task-Blind Worked
- **Forced generalization:** Network had to learn robust shared features
- **Simpler optimization:** No task-specific escape hatches
- **Better balancing:** GradNorm's weights stayed reasonable (1.64, 1.21, 0.15)

---

## Architecture Comparison

### Shared DQN (Baseline)
```
[State (8) + Task Embedding (8)] → Linear(256) → ReLU → Linear(128) → ReLU → Linear(4)
```
- Parameters: 37,788 (with embeddings), 35,716 (without)

### PCGrad DQN
```
Same architecture as Shared DQN
+ PCGrad optimizer wrapper
+ Per-task replay buffers
```
- Parameters: Same as Shared DQN
- Extra: Gradient projection algorithm

### GradNorm DQN
```
Same architecture as Shared DQN
+ GradNorm loss weighter
+ 3 learnable task weights
+ Per-task replay buffers
```
- Parameters: Same as Shared DQN + 3 weights
- Extra: Weight adaptation algorithm

### BRC (Failed)
```
[State (8) + Task Embedding (32)] → Linear(256) → ResBlock → ResBlock → Linear(84)
```
- Parameters: 296,884 (8× Shared DQN)
- Categorical: 4 actions × 21 atoms = 84 outputs
- Failed due to capacity overkill and catastrophic forgetting

---

## Training Configuration

### Common Settings (All Methods)
```python
num_episodes_per_task = 500  # 1500 total
batch_size = 64
replay_buffer_size = 100000  # ~33K per task
min_replay_size = 2000       # Per task
learning_rate = 5e-4
gamma = 0.99
epsilon_decay = 0.995
target_update_freq = 10
```

### Task-Specific Timeouts
```python
max_episode_steps = {
    'standard': 1000,  # Easiest task
    'windy': 400,      # Tight timeout (prevents hovering)
    'heavy': 800,      # Moderate timeout
}
```

### Method-Specific Settings

**PCGrad:**
- No extra hyperparameters
- Automatic conflict detection and projection

**GradNorm:**
- α (alpha) = 1.5  # Asymmetry parameter
- weight_lr = 0.01  # Learning rate for task weights

**BRC:**
- learning_rate = 2.5e-4  # Lower for stability
- weight_decay = 1e-4     # Regularization
- gradient_clip = 10.0    # Prevent explosion
- num_atoms = 21          # Categorical distribution
- embedding_dim = 32      # Larger task embeddings

---

## Final Recommendations

### What Worked
1. **Shared DQN with embeddings:** Strong baseline (202.9 avg)
2. **GradNorm Blind:** Best overall (220.3 avg, +8.6%)
3. **Per-task replay buffers:** Essential for both PCGrad and GradNorm

### What Failed
1. **BRC:** Too much capacity for simple tasks (-67.2%)
2. **PCGrad with embeddings:** Catastrophic failure on Heavy (-117.3%)
3. **GradNorm with embeddings:** Weight imbalance caused collapse (-133.4%)
4. **PCGrad Blind:** Still underperformed baselines (-28.0%)

### Key Insights
1. **Task embeddings help simple methods but hurt complex ones**
2. **Per-task buffers are critical for gradient-based methods**
3. **Network capacity is not a substitute for proper multi-task learning**
4. **GradNorm's dynamic balancing works best with task-blind networks**

---

*Last updated: 2026-01-19*
