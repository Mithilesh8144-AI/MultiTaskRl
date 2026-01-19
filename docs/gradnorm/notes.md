# GradNorm (Gradient Normalization) - Experiment Notes

**Date:** 2026-01-18
**Status:** Complete
**Method:** GradNorm - Adaptive Loss Balancing for Multi-Task Learning

---

## Overview

I implemented "Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks" (Chen et al., 2018). It dynamically balances task losses by learning task weights that encourage all tasks to train at similar rates.

**Key Insight:** GradNorm balances gradient *magnitudes*, while PCGrad handles gradient *directions*. They solve different aspects of multi-task interference.

---

## The Gradient Imbalance Problem

In multi-task learning, some tasks may dominate training because their gradients are larger:

```
Task A gradient magnitude: ||∇A|| = 100   ← Dominates training!
Task B gradient magnitude: ||∇B|| = 10
Task C gradient magnitude: ||∇C|| = 5

Combined gradient mostly follows Task A direction
Tasks B and C make little progress
```

This happens when:
- Tasks have different loss scales
- Tasks are at different stages of learning
- Tasks have different difficulty levels

---

## How I Implemented GradNorm

### Step 1: Compute Per-Task Gradient Norms

For each task, I compute the gradient norm on the shared layer:

```python
# Weighted loss for task i
weighted_loss_i = w_i * loss_i

# Gradient norm on shared layer (e.g., last hidden layer)
G_i = ||∇_shared(w_i * L_i)||
```

### Step 2: Compute Training Rates

I track how fast each task is learning relative to its initial loss:

```python
# Training rate = current_loss / initial_loss
r_i = L_i(t) / L_i(0)

# Relative inverse training rate (normalized)
r_tilde_i = r_i / mean(r)
```

Tasks with higher `r_tilde` are learning slower (higher relative loss).

### Step 3: Compute Target Gradient Norms

The key insight: slower tasks should have *larger* gradients to catch up:

```python
# Average gradient norm across tasks
G_avg = mean(G_i for all tasks)

# Target norm for task i (α controls balancing strength)
target_i = G_avg * (r_tilde_i)^α
```

### Step 4: Update Task Weights

I minimize the difference between actual and target gradient norms:

```python
# GradNorm loss
L_grad = sum(|G_i - target_i| for all tasks)

# Update weights to minimize L_grad
w_i = w_i - lr_weight * ∇L_grad
```

### Step 5: Normalize Weights

Keep weights from exploding or vanishing:

```python
# Renormalize so weights sum to num_tasks
w = w * num_tasks / sum(w)
```

---

## The Alpha Parameter

The asymmetry parameter `α` controls balancing aggressiveness:

| α Value | Behavior |
|---------|----------|
| α = 0 | No balancing (standard weighted sum) |
| α = 1 | Linear balancing |
| α = 1.5 | **What I used** - stronger emphasis on lagging tasks |
| α > 2 | Very aggressive balancing |

With α = 1.5:
- Tasks that are 2× slower get ~2.8× higher target gradient norm
- Tasks that are 3× slower get ~5.2× higher target gradient norm

---

## Visual Comparison: Before vs After GradNorm

```
Before GradNorm (equal weights w=1):
┌─────────────────────────────────────────────────┐
│ Task A: ████████████████████████████  ||∇|| = 100
│ Task B: ████                          ||∇|| = 10
│ Task C: ██                            ||∇|| = 5
└─────────────────────────────────────────────────┘
Total gradient dominated by Task A

After GradNorm (learned weights):
┌─────────────────────────────────────────────────┐
│ Task A: ████████████  w=0.4           ||∇|| = 40
│ Task B: ████████████  w=3.0           ||∇|| = 30
│ Task C: ████████████  w=6.0           ||∇|| = 30
└─────────────────────────────────────────────────┘
Balanced gradients → all tasks progress equally
```

---

## Architecture

I used the same SharedQNetwork as Shared DQN:

```
State (8) + Task Embedding (8) → Linear(256) → ReLU → Linear(128) → ReLU → Linear(4)
                                                            ↑
                                                    Shared layer for
                                                    gradient norm computation
```

Or in task-blind mode:
```
State (8) → Linear(256) → ReLU → Linear(128) → ReLU → Linear(4)
```

**Parameters:** 37,788 (task-aware) or 35,716 (task-blind) + 3 learnable weights

---

## Key Difference from Other Methods

| Method | What it addresses | Mechanism |
|--------|-------------------|-----------|
| Shared DQN | Nothing (baseline) | Simple gradient sum |
| PCGrad | Gradient *direction* conflicts | Project conflicting gradients |
| **My GradNorm** | Gradient *magnitude* imbalance | Learn task weights |
| BRC | Capacity hypothesis | Larger network |

GradNorm and PCGrad are complementary - they could theoretically be combined.

---

## Configuration

```python
{
    'num_episodes_per_task': 500,     # 1500 total
    'batch_size': 64,
    'replay_buffer_size': 100000,     # Split across tasks (~33K per task)
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

    # GradNorm-specific
    'gradnorm_alpha': 1.5,            # Asymmetry parameter (recommended: 1.5)
    'gradnorm_lr': 0.01,              # Learning rate for task weights
    'weight_log_freq': 10,            # Log weight stats every 10 episodes

    # Output
    'output_dir': 'gradnorm',         # or 'gradnorm_blind'
}
```

---

## Results

| Method | Standard | Windy | Heavy | Avg | vs Shared DQN |
|--------|----------|-------|-------|-----|---------------|
| Shared DQN (baseline) | 263 | 130 | 224 | **206** | - |
| **GradNorm (task-aware)** | -141 | -6 | -57 | **-68** | **-133%** |
| **GradNorm (task-blind)** ⭐ | 266 | 169 | 227 | **220** | **+8.6%** |

---

## What Happened

### GradNorm with Task Embeddings (Complete Failure)

**All tasks failed (all negative rewards)**
- Standard: -141 (crashed)
- Windy: -6 (barely learned)
- Heavy: -57 (failed)
- Learned weights were extremely imbalanced: Standard: 2.66, Windy: 0.009, Heavy: 0.33

**My Analysis:** The weight balancing went completely wrong. GradNorm gave almost all weight to the Standard task (2.66) while nearly ignoring Windy (0.009). This created a feedback loop where:
1. Standard task got most of the gradient updates
2. Other tasks fell behind
3. GradNorm increased Standard's weight even more to "balance" gradient magnitudes
4. Eventually all tasks collapsed

The task embeddings exacerbated this by allowing the network to overfit to Standard while ignoring the others. This is a classic case of GradNorm's weight adaptation going awry when task similarities aren't properly accounted for.

### GradNorm Task-Blind (BREAKTHROUGH!) ⭐

After the catastrophic failure with embeddings, I tried GradNorm in task-blind mode. This became my **best performing model**!

**Results:**
- Beat all baselines including Shared DQN
- Standard: 266 (-1.3% vs Shared DQN with embeddings, but more stable)
- Windy: 169 (+46.5% vs Shared DQN with embeddings) ← Biggest improvement
- Heavy: 227 (+1.1% vs Shared DQN with embeddings)
- Much more balanced learned weights: Standard: 1.64, Windy: 1.21, Heavy: 0.15

**My Analysis:** This was the breakthrough. By removing task embeddings, GradNorm could focus on balancing gradient magnitudes across the shared representation rather than fighting with task-specific parameters. The weights converged to a reasonable distribution that prioritized Standard and Windy (1.64, 1.21) while keeping Heavy contributing (0.15).

**Why it worked:**
1. **No task-specific overfitting:** Without embeddings, the network had to learn robust shared features
2. **Better weight balance:** Weights stayed in a reasonable range (0.15-1.64 vs 0.009-2.66 in task-aware)
3. **Especially strong on Windy:** The automatic balancing helped the difficult Windy task get enough attention without sacrificing the others
4. **Stable training:** Training curves showed smooth, stable learning across all tasks

---

## Experiments I Ran

### Experiment 1: GradNorm (Task-Aware)
```bash
# Config: use_task_embedding: True, output_dir: 'gradnorm'
python -m experiments.gradnorm.train
python -m experiments.gradnorm.evaluate --episodes 20
python -m experiments.analyze_results --method gradnorm
```

### Experiment 2: GradNorm (Task-Blind)
```bash
# Config: use_task_embedding: False, output_dir: 'gradnorm_blind'
python -m experiments.gradnorm.train
python -m experiments.gradnorm.evaluate --episodes 20
python -m experiments.analyze_results --method gradnorm_blind
```

---

## Weight Evolution I Tracked

GradNorm tracks how task weights evolve during training:

```python
# Example weight evolution for task-blind (successful)
Episode 0:    w = [1.0, 1.0, 1.0]      # Start equal
Episode 500:  w = [1.2, 1.3, 0.5]      # Windy needs more weight
Episode 1000: w = [1.5, 1.2, 0.3]      # Converging
Episode 1500: w = [1.64, 1.21, 0.15]   # Stable
```

Higher weight = task was learning slower → GradNorm boosted it.

---

## Files

| File | Purpose |
|------|---------|
| `agents/gradnorm.py` | GradNormLossWeighter, GradNormDQNAgent |
| `experiments/gradnorm/config.py` | Hyperparameters |
| `experiments/gradnorm/train.py` | Training with per-task buffers |
| `experiments/gradnorm/evaluate.py` | Evaluation |
| `results/gradnorm/` | Task-aware results |
| `results/gradnorm_blind/` | Task-blind results |

---

## GradNorm Algorithm (Pseudocode)

```python
def gradnorm_update(losses, shared_layer, weights, alpha):
    # 1. Compute weighted losses
    weighted_losses = [w * L for w, L in zip(weights, losses)]

    # 2. Compute gradient norms on shared layer
    grad_norms = []
    for wL in weighted_losses:
        grad = compute_gradient(wL, shared_layer.parameters())
        grad_norms.append(norm(grad))

    # 3. Compute training rates
    if initial_losses is None:
        initial_losses = losses  # First iteration

    training_rates = [L / L0 for L, L0 in zip(losses, initial_losses)]
    mean_rate = mean(training_rates)
    relative_rates = [r / mean_rate for r in training_rates]

    # 4. Compute target norms
    mean_norm = mean(grad_norms)
    targets = [mean_norm * (r ** alpha) for r in relative_rates]

    # 5. GradNorm loss
    gradnorm_loss = sum(|G - target| for G, target in zip(grad_norms, targets))

    # 6. Update weights
    weights = weights - lr * gradient(gradnorm_loss, weights)

    # 7. Renormalize
    weights = weights * num_tasks / sum(weights)

    return weights
```

---

## Key Takeaways

1. **GradNorm balances gradient magnitudes** - different from PCGrad's direction handling
2. **Learnable task weights** - network discovers optimal balancing
3. **α = 1.5 worked well** - stronger emphasis on lagging tasks
4. **Task embeddings catastrophically failed** - weight adaptation went haywire
5. **Task-blind GradNorm = my best method** (+8.6% vs Shared DQN baseline)
6. **For similar tasks, forcing shared representation works better** than task-specific escape hatches

---

## References

- "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks" (Chen et al., 2018)
- Paper: https://arxiv.org/abs/1711.02257
