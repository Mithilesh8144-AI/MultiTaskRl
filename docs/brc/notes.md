# BRC (Bigger, Regularized, Categorical) - Experiment Notes

**Date:** 2026-01-07
**Status:** Complete (Failed to outperform baselines)
**Method:** BRC - Large network with distributional RL

---

## Overview

I tested the hypothesis: **"Can large networks absorb gradient conflicts without gradient surgery?"**

The name BRC stands for:
- **B**igger: ~297K parameters (8× Shared DQN)
- **R**egularized: Weight decay + LayerNorm + Gradient clipping
- **C**ategorical: Distributional RL with 21 atoms

**Result:** BRC underperformed both baselines. Large networks don't automatically solve multi-task learning.

---

## Architecture: BroNet

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

### ResidualBlock

```python
def forward(self, x):
    residual = x           # Save input
    x = LayerNorm(x)       # Normalize
    x = ReLU(Linear(x))    # Transform
    x = Linear(x)          # Transform again
    return x + residual    # Skip connection
```

**Why I Used Skip Connections:**
- Prevent vanishing gradients in deep networks
- Allow gradients to flow directly backward

**Why LayerNorm (not BatchNorm):**
- RL data is non-stationary
- BatchNorm assumes i.i.d. data
- LayerNorm normalizes per-sample

---

## Categorical DQN & Atoms

### The Core Difference

**Standard DQN** predicts a single Q-value:
```
Q(state, action) = 150.5  (just one number)
```

**Categorical DQN** predicts a probability distribution:
```
Q(state, action) = "30% chance of ~100, 50% chance of ~150, 20% chance of ~200"
```

### What Are Atoms?

Atoms are **fixed return values** that discretize the range of possible returns.

```
Lunar Lander returns: -100 (crash) to +300 (perfect landing)

21 atoms (fixed points):
  -100, -80, -60, -40, -20, 0, +20, +40, +60, +80, +100,
  +120, +140, +160, +180, +200, +220, +240, +260, +280, +300

Spacing: delta_z = (300 - (-100)) / (21 - 1) = 20
```

Think of atoms as **buckets** for possible returns.

### How Atoms Work

**Step 1: Network outputs probabilities**

For each action, output 21 probabilities (sum to 1.0):
```
Action "fire main engine":
  Atom -100: 0.01 (1% chance of crash-level return)
  Atom -80:  0.02
  ...
  Atom +200: 0.15 (15% chance of good landing)
  ...
  Atom +300: 0.02 (2% chance of perfect return)
```

**Step 2: Compute Q-value as expected value**

```python
Q = sum(atom_value × probability)
Q = (-100 × 0.01) + (-80 × 0.02) + ... + (+300 × 0.02)
Q = 156.4
```

### Visual Comparison

```
Standard DQN output (4 numbers):
┌─────────────────────────────────────────┐
│  Action 0: Q = 142.5                    │
│  Action 1: Q = 98.3                     │
│  Action 2: Q = 156.2  ← Best action     │
│  Action 3: Q = 112.7                    │
└─────────────────────────────────────────┘

Categorical DQN output (4 × 21 = 84 numbers):
┌─────────────────────────────────────────┐
│  Action 0: [0.01, 0.02, ..., 0.02]      │ ← 21 probabilities
│  Action 1: [0.03, 0.05, ..., 0.01]      │
│  Action 2: [0.00, 0.01, ..., 0.05]      │ ← Best (highest E[Q])
│  Action 3: [0.02, 0.04, ..., 0.03]      │
└─────────────────────────────────────────┘
```

### Why I Used Distributions

**1. Captures Uncertainty**

Two actions can have the same average Q but different risk:
```
Risky:  Bimodal distribution (might get -100 or +200)
Safe:   Narrow distribution around +50
```

**2. More Stable Learning**

- Standard DQN: MSE loss = `(Q - target)²`
- Categorical: Cross-entropy = `-Σ target_prob × log(pred_prob)`
- Cross-entropy has smoother gradients

**3. Better for Multi-Task (I thought)**

Different tasks have different return distributions:
- Standard: Narrow around +200
- Windy: Wide distribution (high variance)
- Heavy: Shifted distribution

### Why I Chose 21 Atoms (not 51)

Original C51 paper used 51. I reduced for Lunar Lander:

| Atoms | Output Size | Trade-off |
|-------|-------------|-----------|
| 51 | 4 × 51 = 204 | More precise, harder to train |
| 21 | 4 × 21 = 84 | Less precise, more stable |

### Atoms Quick Reference

| Concept | Meaning |
|---------|---------|
| **Atom** | Fixed return value (-100, -80, ..., +300) |
| **Support** | Full set of atoms |
| **Distribution** | Probabilities over atoms (sums to 1.0) |
| **Q-value** | Expected value = Σ(atom × probability) |
| **delta_z** | Spacing between atoms (20 for my config) |

---

## Parameters

```
Task Embedding:      96      (3 × 32)
Input Layer:         10,496
Residual Block 1:    132,096
Residual Block 2:    132,096
Final LayerNorm:     512
Output Layer:        21,588  (256 × 84 + 84)
─────────────────────────────
TOTAL:               ~297,000 parameters
```

**Comparison:**
- Independent DQN: 107K (3 networks)
- Shared DQN: 38K
- **My BRC: 297K (8× Shared)**

---

## Regularization

### 1. Weight Decay
```python
optimizer = AdamW(params, lr=2.5e-4, weight_decay=1e-4)
```
Penalizes large weights, prevents overfitting.

### 2. LayerNorm
Normalizes activations, stabilizes training.

### 3. Gradient Clipping
```python
clip_grad_norm_(parameters, max_norm=10.0)
```
Prevents gradient explosion.

### 4. Lower Learning Rate
2.5e-4 (vs 5e-4 for Shared DQN) - more conservative updates.

---

## Configuration

```python
{
    'num_episodes_per_task': 500,
    'batch_size': 64,
    'replay_buffer_size': 100000,
    'min_replay_size': 2000,

    # Architecture
    'hidden_dim': 256,
    'num_blocks': 2,
    'embedding_dim': 32,

    # Categorical DQN
    'num_atoms': 21,
    'v_min': -100.0,
    'v_max': 300.0,

    # Optimization
    'learning_rate': 2.5e-4,
    'weight_decay': 1e-4,
    'gamma': 0.99,

    # Exploration
    'epsilon_decay': 0.997,  # Slower decay
    'target_update_freq': 5,  # More frequent updates
}
```

---

## Results

| Method | Standard | Windy | Heavy | Avg | Params |
|--------|----------|-------|-------|-----|--------|
| Independent DQN | 228 | 100 | 194 | **174** | 107K |
| Shared DQN | 263 | 130 | 224 | **206** | 38K |
| **My BRC** | 124 | 49 | 18 | **64** | 297K |

**BRC failed dramatically.**

---

## Why BRC Failed

### 1. Task Too Simple
Lunar Lander has 8D state, 4 actions. ~35K params already achieves good performance. ~297K is massive overkill.

### 2. Categorical Overkill
Lunar Lander returns are relatively predictable. The 21-atom distribution added complexity without capturing useful information.

### 3. Catastrophic Forgetting
My large network oscillated between tasks. When improving on one task, it regressed on others.

### 4. Training Instability
Performance oscillated wildly:
```
Episode  300:   -19.4
Episode  350:  -188.4  ← Sudden drop
Episode  400:   -91.9
Episode 1350:    73.3  ← Best
Episode 1400:   -43.0  ← Immediate regression
```

---

## Key Takeaway

**Bigger networks don't automatically solve multi-task learning.**

My BRC hypothesis was:
> "Can large networks absorb gradient conflicts without gradient surgery?"

**Answer: No** (at least for Lunar Lander)

Simple Shared DQN (38K params) outperformed my BRC (297K params) by 3× on average reward.

---

## Files

| File | Purpose |
|------|---------|
| `agents/brc.py` | ResidualBlock, BroNet, BRCAgent |
| `experiments/brc/config.py` | Hyperparameters |
| `experiments/brc/train.py` | Training loop |
| `experiments/brc/evaluate.py` | Evaluation |
| `results/brc/` | Models and metrics |

---

## Commands

### Training
```bash
python -m experiments.brc.train
```

### Evaluation
```bash
python -m experiments.brc.evaluate --episodes 20
python -m experiments.brc.evaluate --task windy --render
```

### Analysis
```bash
python -m experiments.analyze_results --method brc
```

---

## When BRC Might Work Better

- More complex environments (Atari, MuJoCo)
- Higher-dimensional state spaces
- Tasks with genuinely different return distributions
- Environments requiring risk-sensitive behavior

---

## Conclusion

BRC was an interesting experiment but not suitable for Lunar Lander. The added complexity (distributional RL, large networks) didn't help and actually hurt performance.

**My Next Step:** Try gradient-based methods (PCGrad) that directly address gradient conflicts instead of trying to absorb them with capacity.
