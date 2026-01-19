# BRC (Bigger, Regularized, Categorical) - Analysis Report

## Overview

BRC was implemented as an advanced multi-task baseline combining:
- **Bigger**: Larger network with residual blocks (BroNet architecture)
- **Regularized**: Weight decay (L2 regularization)
- **Categorical**: Distributional RL with C51 (categorical DQN)

**Goal**: Outperform Shared DQN by using increased capacity and distributional learning to handle gradient conflicts in multi-task learning.

**Result**: BRC underperformed both baselines despite significant tuning efforts.

---

## Baseline Comparison

| Method | Standard | Windy | Heavy | Avg | Parameters |
|--------|----------|-------|-------|-----|------------|
| Independent DQN | 228 | 100 | 194 | **174** | 107K |
| Shared DQN | 263 | 130 | 224 | **206** | 38K |
| **BRC (best attempt)** | 124 | 49 | 18 | **64** | 297K |

---

## BRC Architecture Summary

### BroNet Structure
```
State (8) + Task Embedding (32) → Concatenate (40)
                ↓
        Linear(40 → 256) + ReLU
                ↓
        ResidualBlock 1 (LayerNorm → Linear → ReLU → Linear + skip)
                ↓
        ResidualBlock 2 (LayerNorm → Linear → ReLU → Linear + skip)
                ↓
        Final LayerNorm
                ↓
        Linear(256 → 84)  [4 actions × 21 atoms]
```

### Categorical DQN (Atoms)

Instead of predicting single Q-values, BRC predicts probability distributions over possible returns.

**What are atoms?**
Atoms are fixed return values that discretize the return range:
```
v_min = -100 (crash), v_max = +300 (perfect landing)
21 atoms: [-100, -80, -60, ..., +260, +280, +300]
delta_z = 20 (spacing between atoms)
```

**How it works:**
1. Network outputs 21 probabilities per action (sum to 1.0)
2. Q-value = expected value = Σ(atom × probability)
3. Loss = cross-entropy between predicted and target distributions

**Why distributions?**
- Captures uncertainty (same Q, different risk profiles)
- More stable gradients (cross-entropy vs MSE)
- Better for tasks with different return variances

---

## Issues Encountered

### 1. Slow Training (Critical)

**Problem**: Initial training took 7+ hours for 1500 episodes.

**Root Cause**: The `_project_distribution` function used nested Python for-loops:
```python
# SLOW: O(batch_size × num_atoms) Python iterations
for i in range(batch_size):      # 64 iterations
    for j in range(num_atoms):   # 21 iterations
        # ... projection logic
```

**Fix**: Vectorized using `scatter_add_`:
```python
# FAST: Fully vectorized with PyTorch
target_probs.view(-1).scatter_add_(
    0,
    (batch_idx * self.num_atoms + l).view(-1),
    l_weight.view(-1)
)
```

**Impact**: ~10-50x speedup on projection step.

---

### 2. Insufficient Training Episodes

**Problem**: Initial config had only 200 episodes per task (600 total) vs 500 per task for Shared DQN.

**Fix**: Increased to 500 episodes per task (1500 total) to match Shared DQN.

---

### 3. Training Instability / Catastrophic Forgetting

**Problem**: Performance oscillated wildly during training. Example from Heavy task:
```
Episode  300:   -19.4
Episode  350:  -188.4  ← Sudden drop
Episode  400:   -91.9
...
Episode 1350:    73.3  ← Best
Episode 1400:   -43.0  ← Immediate regression
```

All three tasks showed similar instability in later training, suggesting **catastrophic forgetting** from multi-task interference.

**Attempted Fixes**:
- Lower learning rate (3e-4 → 2.5e-4)
- Slower epsilon decay (0.995 → 0.997)
- More frequent target updates (10 → 5 episodes)
- Reduced atoms (51 → 21)
- Reduced residual blocks (3 → 2)

**Result**: Reduced but did not eliminate instability.

---

### 4. Model Complexity Mismatch

**Problem**: BRC's ~297K parameters was overkill for Lunar Lander's 8-dim state space.

**Attempted Fix**: Simplified architecture:
| Parameter | Original | Simplified |
|-----------|----------|------------|
| num_blocks | 3 | 2 |
| num_atoms | 51 | 21 |
| Parameters | 460K | 297K |

**Result**: Marginal improvement in training stability, but evaluation still poor.

---

### 5. Evaluation vs Training Mismatch

**Problem**: "Best" rewards during training (270, 178, 108) did not match evaluation results (124, 49, 18).

**Root Cause**:
- Training evaluation used only 5 episodes per task
- High variance meant lucky evaluations triggered "best" model saves
- True performance with 20 episodes revealed instability

**Recommendation**: Use 10+ episodes for training evaluation, or track moving average.

---

## Configuration Evolution

### Initial Config (Failed)
```python
'num_episodes_per_task': 200,
'num_blocks': 3,
'num_atoms': 51,
'learning_rate': 3e-4,
'epsilon_decay': 0.995,
'target_update_freq': 10,
```
**Result**: Standard 191, Windy 89, Heavy -19

### Optimized Config (Still Failed)
```python
'num_episodes_per_task': 500,
'num_blocks': 2,
'num_atoms': 21,
'learning_rate': 2.5e-4,
'epsilon_decay': 0.997,
'target_update_freq': 5,
```
**Result**: Standard 124, Windy 49, Heavy 18

---

## Why BRC Failed on Lunar Lander

### 1. Task Simplicity
Lunar Lander has an 8-dimensional state space and 4 discrete actions. Standard DQN with ~35K parameters already achieves good performance. BRC's additional complexity added noise without benefit.

### 2. Categorical DQN Overhead
Distributional RL shines in environments with:
- High stochasticity
- Multi-modal return distributions
- Need for risk-sensitive policies

Lunar Lander's returns are relatively predictable. The 21-atom distribution added training complexity without capturing useful information.

### 3. Multi-Task Interference
The residual blocks and shared representations caused task interference. When improving on one task, the model would regress on others. Shared DQN's simpler architecture (with smaller task embeddings) handled this better.

### 4. Hyperparameter Sensitivity
BRC required careful tuning of:
- Learning rate (too high = instability, too low = slow convergence)
- Number of atoms and support range [v_min, v_max]
- Network depth and width
- Target update frequency

The simpler baselines were more robust to hyperparameter choices.

---

## Atoms Deep Dive

### What Atoms Represent

Each atom is a **possible return value**. The network learns to predict which returns are likely.

```
Atom values (21 total):
[-100, -80, -60, -40, -20, 0, +20, +40, +60, +80, +100,
 +120, +140, +160, +180, +200, +220, +240, +260, +280, +300]
```

### Network Output

For each action, the network outputs 21 logits → softmax → 21 probabilities:

```
Action "fire main engine":
  Atom -100: 0.01 probability (1% chance of crash-level return)
  Atom -80:  0.02
  Atom -60:  0.03
  ...
  Atom +200: 0.15 (15% chance of good landing return)
  Atom +220: 0.12
  ...
  Atom +300: 0.02 (2% chance of perfect return)
```

### Computing Q-Value

```python
Q = sum(atom_value * probability for all atoms)
Q = (-100 * 0.01) + (-80 * 0.02) + ... + (+300 * 0.02)
Q = 156.4  # Expected return
```

### Distributional Bellman Update

The tricky part: after computing `r + γ * next_atoms`, the values don't align with our fixed atoms.

```
Original atoms: [-100, -80, -60, ...]
After Bellman:  [-90, -70, -50, ...]  ← Shifted, doesn't match!
```

**Solution**: Project back onto fixed atoms using linear interpolation:
```
Target value -70 → split between atoms -80 (50%) and -60 (50%)
```

This projection step is what made BRC slow initially (nested loops) and was fixed with vectorization.

---

## Recommendations

### For This Project
1. **Move to PCGrad/GradNorm**: These methods directly address gradient conflicts without architectural complexity
2. **Skip BRC**: The added complexity doesn't pay off for Lunar Lander

### For Future BRC Usage
BRC may work better on:
- More complex environments (Atari, MuJoCo)
- Environments with high return variance
- Tasks requiring risk-sensitive behavior

If attempting BRC again:
1. Start with fewer atoms (11-21) and increase if needed
2. Use longer evaluation windows (10+ episodes)
3. Implement early stopping based on moving average
4. Consider per-task learning rate scaling

---

## Files Modified

- `agents/brc.py`: Vectorized `_project_distribution` method
- `experiments/brc/config.py`: Multiple hyperparameter adjustments
- `experiments/brc/evaluate.py`: Fixed import path

---

## Conclusion

BRC was not suitable for the Lunar Lander multi-task setting. Despite multiple optimization attempts, it consistently underperformed the simpler Shared DQN baseline by a significant margin (64 avg vs 206 avg). The categorical DQN component added complexity without benefit, and the larger network capacity led to overfitting and task interference.

**Key Takeaway**: Bigger networks and distributional RL don't automatically solve multi-task learning. The gradient conflict problem requires targeted solutions like PCGrad.

**Final Recommendation**: Proceed with gradient-based multi-task methods (PCGrad, GradNorm) which address the core challenge (gradient conflicts) more directly.
