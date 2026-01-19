# PCGrad (Projected Gradient) - Experiment Notes

**Date:** 2026-01-18
**Status:** Complete
**Method:** PCGrad - Gradient Surgery for Multi-Task Learning

---

## Overview

I implemented "Gradient Surgery for Multi-Task Learning" (Yu et al., 2020). When gradients from different tasks conflict (negative dot product), PCGrad projects them to eliminate the conflicting component.

**Key Insight:** PCGrad works on gradients, not network architecture. I can use it with or without task embeddings.

---

## The Gradient Conflict Problem

In multi-task learning, different tasks may want to push shared parameters in opposite directions:

```
Task A gradient: →→→→→
Task B gradient: ←←←←←
Combined (sum):  ~0    ← Gradients cancel out!
```

This is called **negative transfer** - tasks interfere with each other.

---

## How I Implemented PCGrad

### Step 1: Compute Per-Task Gradients

```python
# Separate replay buffers per task
batch_A = buffer_A.sample(batch_size)
batch_B = buffer_B.sample(batch_size)
batch_C = buffer_C.sample(batch_size)

# Compute losses
loss_A = compute_td_loss(batch_A, task_id=0)
loss_B = compute_td_loss(batch_B, task_id=1)
loss_C = compute_td_loss(batch_C, task_id=2)

# Get gradients
grad_A = compute_gradient(loss_A)
grad_B = compute_gradient(loss_B)
grad_C = compute_gradient(loss_C)
```

### Step 2: Detect Conflicts

Two gradients conflict if their dot product is negative:

```python
conflict = dot(grad_A, grad_B) < 0
```

Visual:
```
No conflict (angle < 90°):        Conflict (angle > 90°):
    grad_A                            grad_A
       ↗                                 ↗
      /                                 /
     /                                 /
    ●───────→ grad_B                  ●───────→
                                             \
                                              \
                                               ↘ grad_B
```

### Step 3: Project Conflicting Gradients

For conflicting gradients, I remove the component that conflicts:

```python
if dot(grad_A, grad_B) < 0:
    # Project grad_A onto plane perpendicular to grad_B
    grad_A_new = grad_A - (dot(grad_A, grad_B) / norm(grad_B)²) * grad_B
```

Visual:
```
Before projection:           After projection:
                                  grad_A'
    grad_A                          ↑
       ↗                           |
      /                            |
     /                             |
    ●───────→ grad_B              ●───────→ grad_B

grad_A' is perpendicular to grad_B (no conflict!)
```

### Step 4: Average and Apply

```python
# Average projected gradients
final_grad = (grad_A_new + grad_B_new + grad_C_new) / 3

# Apply to network
optimizer.step()
```

---

## PCGrad Algorithm (Pseudocode)

```python
def pc_backward(losses, parameters):
    # 1. Compute per-task gradients
    grads = [compute_gradient(loss) for loss in losses]

    # 2. Project conflicting pairs
    projected = []
    for i, grad_i in enumerate(grads):
        for j, grad_j in enumerate(grads):
            if i != j:
                dot_product = dot(grad_i, grad_j)
                if dot_product < 0:  # Conflict!
                    # Remove conflicting component
                    grad_i = grad_i - (dot_product / norm(grad_j)²) * grad_j
        projected.append(grad_i)

    # 3. Average and apply
    final_grad = mean(projected)
    set_gradients(parameters, final_grad)
```

---

## Why PCGrad Works Without Task Embeddings

Task embeddings and PCGrad are **orthogonal**:

| Component | What it does | Depends on embeddings? |
|-----------|--------------|------------------------|
| Task embeddings | Network knows which task | N/A |
| PCGrad | Projects conflicting gradients | **NO** |

PCGrad computes per-task gradients from **separate replay buffers**, not from the network architecture:

```
My Task-Blind PCGrad Flow:
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Task A Buffer   │     │ Task B Buffer   │     │ Task C Buffer   │
│ (different data)│     │ (different data)│     │ (different data)│
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         ▼                       ▼                       ▼
   ┌───────────┐           ┌───────────┐           ┌───────────┐
   │ Loss_A    │           │ Loss_B    │           │ Loss_C    │
   └─────┬─────┘           └─────┬─────┘           └─────┬─────┘
         │                       │                       │
         ▼                       ▼                       ▼
   ┌───────────┐           ┌───────────┐           ┌───────────┐
   │ Grad_A    │           │ Grad_B    │           │ Grad_C    │
   └─────┬─────┘           └─────┬─────┘           └─────┬─────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                          ┌─────────────┐
                          │ PCGrad      │ ← Projects if conflicts
                          │ Projection  │
                          └─────────────┘
```

Even a task-blind network produces different gradients per task because the buffers contain different experiences.

---

## Architecture

I used the same SharedQNetwork as Shared DQN:

```
State (8) + Task Embedding (8) → Linear(256) → ReLU → Linear(128) → ReLU → Linear(4)
```

Or in task-blind mode:
```
State (8) → Linear(256) → ReLU → Linear(128) → ReLU → Linear(4)
```

**Parameters:** 37,788 (task-aware) or 35,716 (task-blind)

---

## Key Difference from Shared DQN

| Aspect | Shared DQN | My PCGrad |
|--------|-----------|--------|
| Replay Buffer | Single shared buffer | **Separate per-task buffers** |
| Gradient Computation | One loss from mixed batch | **Separate loss per task** |
| Gradient Handling | Simple sum | **Project conflicting gradients** |
| Compute Cost | 1 backward pass | **3 backward passes** (one per task) |

---

## Configuration

```python
{
    'num_episodes_per_task': 500,     # 1500 total
    'batch_size': 64,
    'replay_buffer_size': 100000,     # Shared across tasks (~33K per task)
    'min_replay_size': 2000,          # Per-task minimum
    'learning_rate': 5e-4,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    'target_update_freq': 10,

    # Architecture
    'embedding_dim': 8,
    'hidden_dims': (256, 128),
    'use_task_embedding': True,       # False for task-blind

    # PCGrad-specific
    'gradient_log_freq': 10,          # Log conflict stats

    # Output
    'output_dir': 'pcgrad',           # or 'pcgrad_blind'
}
```

---

## Results

| Method | Standard | Windy | Heavy | Avg | vs Shared DQN |
|--------|----------|-------|-------|-----|---------------|
| Shared DQN (baseline) | 263 | 130 | 224 | **206** | - |
| **PCGrad (task-aware)** | 224 | 30 | -359 | **-35** | **-117%** |
| **PCGrad (task-blind)** | 189 | 70 | 178 | **146** | **-29%** |

---

## What Went Wrong

### PCGrad with Task Embeddings (Catastrophic Failure)

**Heavy task completely failed (-359 reward)**
- Heavy task consistently crashed or terminated early
- Standard task was okay (224)
- Windy task performed poorly (30)

**My Analysis:** Task embeddings combined with gradient projection created instability. Looking at my training curves, the Heavy task's gradient conflicts with the other tasks caused the network to diverge. The gradient projection might have been too aggressive, preventing the network from finding a good multi-task solution.

### PCGrad Task-Blind (Stable but Underperformed)

After the failure with task embeddings, I removed them and trained PCGrad in task-blind mode where the network doesn't know which task it's currently training on.

**Results:**
- Much more stable than the task-aware version
- Standard: 189 (decent)
- Windy: 70 (struggled)
- Heavy: 178 (recovered from catastrophic failure)

**My Analysis:** Removing task embeddings fixed the instability, but PCGrad still underperformed the simple Shared DQN baseline. The gradient projection seems to be overly conservative, preventing the network from learning effective shared representations. Without task conditioning, PCGrad couldn't leverage task-specific information effectively, and the gradient conflicts it was designed to solve weren't actually the main problem in this domain.

---

## Experiments I Ran

### Experiment 1: PCGrad (Task-Aware)
```bash
# Config: use_task_embedding: True, output_dir: 'pcgrad'
python -m experiments.pcgrad.train
python -m experiments.pcgrad.evaluate --episodes 20
python -m experiments.analyze_results --method pcgrad
```

### Experiment 2: PCGrad (Task-Blind)
```bash
# Config: use_task_embedding: False, output_dir: 'pcgrad_blind'
python -m experiments.pcgrad.train
python -m experiments.pcgrad.evaluate --episodes 20
python -m experiments.analyze_results --method pcgrad_blind
```

---

## Conflict Statistics

PCGrad tracks conflict metrics:

```python
stats = {
    'conflict_count': 5,      # Number of conflicting gradient pairs
    'total_pairs': 6,         # Total pairs checked (3 choose 2 = 3, but check both directions)
    'conflict_ratio': 0.83,   # 5/6 = 83% of pairs conflicted
}
```

High conflict ratio → PCGrad is doing more work
Low conflict ratio → Gradients already aligned (PCGrad less useful)

---

## Files

| File | Purpose |
|------|---------|
| `agents/pcgrad.py` | PCGradOptimizer, PerTaskReplayBuffer, PCGradDQNAgent |
| `experiments/pcgrad/config.py` | Hyperparameters |
| `experiments/pcgrad/train.py` | Training with separate per-task buffers |
| `experiments/pcgrad/evaluate.py` | Evaluation |
| `results/pcgrad/` | Task-aware results |
| `results/pcgrad_blind/` | Task-blind results |

---

## Key Takeaways

1. **PCGrad is orthogonal to task embeddings** - works with or without
2. **Requires separate per-task buffers** - more memory than Shared DQN
3. **3× more backward passes** - more compute than Shared DQN
4. **Task embeddings + PCGrad = catastrophic failure** for my Lunar Lander tasks
5. **Task-blind helped stability** but PCGrad's core mechanism (conflict resolution) might not be beneficial for these similar tasks
6. **Gradient conflicts may not be the problem** - maybe they're actually providing beneficial regularization

---

## References

- "Gradient Surgery for Multi-Task Learning" (Yu et al., 2020)
- Paper: https://arxiv.org/abs/2001.06782
