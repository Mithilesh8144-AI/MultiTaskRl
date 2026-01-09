# BRC (Bigger, Regularized, Categorical) - Model Summary

**Date:** 2026-01-07
**Status:** ⏳ Implementation Complete, Ready to Train
**Method:** BRC - Bigger, Regularized, Categorical DQN

---

## 1. What is BRC?

**BRC** is a high-capacity multi-task RL approach that uses:
- **B**igger networks (459K params vs Shared DQN's 37K)
- **R**egularized training (weight decay, LayerNorm, gradient clipping)
- **C**ategorical DQN (distributional RL with 51-atom value distributions)

**Core Hypothesis:**
Large networks with proper regularization can absorb gradient conflicts in multi-task learning, approaching Independent DQN performance while maintaining parameter sharing.

**Expected Result:**
- Should outperform Shared DQN due to higher capacity
- May approach or match Independent DQN performance
- Test: Can bigger networks solve multi-task learning without gradient surgery?

---

## 2. Architecture Overview

### BroNet (Brother Network)

**High-Level Structure:**
```
State (8) + Task Embedding (32) → Input Layer (256)
                                    ↓
                            Residual Block 1 (256)
                                    ↓
                            Residual Block 2 (256)
                                    ↓
                            Residual Block 3 (256)
                                    ↓
                            Final LayerNorm
                                    ↓
                        Output Layer (4 actions × 51 atoms)
```

### Detailed Component Breakdown

**1. Task Embedding Layer:**
```python
Embedding(3 tasks, 32 dim) = 96 parameters
```
- Learned 32-dim vector per task (4× larger than Shared DQN's 8-dim)
- Initialized: `N(0, 1/32)` (Xavier-style initialization)
- Concatenated with state before input layer

**2. Input Layer:**
```python
Linear(8 + 32 → 256) + Bias(256) = 10,496 parameters
```
- Takes concatenated [state, task_embedding]
- Projects to 256-dim hidden space
- ReLU activation

**3. Residual Blocks (×3):**
```python
Each block:
  LayerNorm(256) = 512 parameters (gamma + beta)
  Linear(256 → 256) + Bias = 65,792 parameters
  Linear(256 → 256) + Bias = 65,792 parameters
  Total per block: 132,096 parameters

Total 3 blocks: 396,288 parameters
```

**ResidualBlock Structure:**
```python
def forward(self, x):
    residual = x  # Save input for skip connection
    x = self.ln(x)  # LayerNorm
    x = F.relu(self.fc1(x))  # First linear + ReLU
    x = self.fc2(x)  # Second linear
    return x + residual  # Add skip connection
```

**Why Residual Connections?**
- Prevent vanishing gradients in deep networks
- Allow gradients to flow directly through skip connections
- Enable stable training of 3-block network

**4. Final LayerNorm:**
```python
LayerNorm(256) = 512 parameters
```

**5. Output Layer (Categorical DQN):**
```python
Linear(256 → 4 actions × 51 atoms) + Bias = 52,428 parameters
```
- Outputs logits for 51-atom distribution per action
- Total output: 4 actions × 51 atoms = 204 values

### Total Parameters

```
Task Embedding:      96
Input Layer:         10,496
Residual Blocks:     396,288
Final LayerNorm:     512
Output Layer:        52,428
─────────────────────────────
TOTAL:               459,820 parameters
```

**Comparison:**
- Independent DQN: 107,148 params (3 networks)
- Shared DQN: 37,788 params
- **BRC: 459,820 params (12.2× Shared, 4.3× Independent)**

---

## 3. Categorical DQN (C51)

### What is Categorical DQN?

**Standard DQN:**
- Predicts single Q-value per action: `Q(s, a) = scalar`
- Uses MSE loss: `(Q(s,a) - target)²`

**Categorical DQN (C51):**
- Predicts full distribution over returns: `Z(s, a) ~ distribution`
- Uses 51 atoms from v_min to v_max
- Uses cross-entropy loss between distributions

### The 51-Atom Distribution

**Support (fixed):**
```python
v_min = -100.0
v_max = 300.0
num_atoms = 51

support = [-100, -92, -84, ..., 284, 292, 300]  # 51 evenly-spaced values
delta_z = (300 - (-100)) / (51 - 1) = 8.0
```

**Interpretation:**
- Each atom represents a possible return value
- Network predicts probability mass over atoms
- Agent learns full distribution, not just mean

**Example Output:**
```
Action 0: [0.01, 0.02, 0.05, ..., 0.12, 0.08]  # 51 probabilities (sum=1)
Action 1: [0.02, 0.03, 0.04, ..., 0.10, 0.15]
Action 2: [0.03, 0.05, 0.08, ..., 0.05, 0.02]
Action 3: [0.01, 0.01, 0.02, ..., 0.20, 0.18]
```

**Q-Value Extraction:**
```python
Q(s, a) = sum(p_i * z_i)  # Expected value of distribution
```

### Distributional Bellman Update

**Standard DQN Bellman:**
```
Q(s, a) = r + γ * max_a' Q(s', a')
```

**Categorical Bellman:**
```
Z(s, a) = r + γ * Z(s', a*)  (distribution shift + projection)
```

**Algorithm:**
1. Get next-state distribution: `Z(s', a*)`
2. Shift by reward: `r + γ * support`
3. Project onto fixed support atoms (distributional projection)
4. Minimize cross-entropy: `H(target_dist, predicted_dist)`

**Why Better than MSE?**
- More stable gradients (cross-entropy vs MSE)
- Captures uncertainty (full distribution vs single value)
- Better for multi-task (task-specific distributions)

---

## 4. Regularization (The "R" in BRC)

### 1. Weight Decay (L2 Regularization)

**AdamW Optimizer:**
```python
optimizer = torch.optim.AdamW(
    params=model.parameters(),
    lr=3e-4,
    weight_decay=1e-4  # L2 penalty on weights
)
```

**Effect:**
- Penalizes large weights: `loss += 1e-4 * sum(w²)`
- Prevents overfitting to task-specific noise
- Encourages learning shared features

### 2. LayerNorm (Not BatchNorm!)

**Why LayerNorm?**
- RL data is non-stationary (policy changes over time)
- BatchNorm assumes stable input distribution (bad for RL)
- LayerNorm normalizes per sample (independent of batch)

**Structure:**
```python
LayerNorm(256): gamma(256) + beta(256) = 512 params
```

**Effect:**
- Stabilizes activations
- Prevents gradient explosion/vanishing
- Helps with deeper networks (3 residual blocks)

### 3. Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=10.0
)
```

**Effect:**
- Prevents gradient explosion
- Stabilizes training with categorical loss
- Critical for large networks

### 4. Lower Learning Rate

```python
learning_rate = 3e-4  # vs Shared DQN's 5e-4
```

**Effect:**
- More conservative updates
- Prevents overshooting with large network
- Better stability for categorical loss

---

## 5. Setup & Configuration

### Hyperparameters

```python
{
    # Training
    'num_episodes_per_task': 500,      # 500 × 3 = 1500 total episodes
    'batch_size': 64,
    'replay_buffer_size': 100000,
    'min_replay_size': 2000,           # Match Shared DQN

    # BroNet Architecture
    'hidden_dim': 256,                 # Can increase to 512 for more capacity
    'num_blocks': 3,                   # Number of residual blocks
    'embedding_dim': 32,               # 4× larger than Shared DQN

    # Categorical DQN
    'num_atoms': 51,                   # Standard C51
    'v_min': -100.0,                   # Min return
    'v_max': 300.0,                    # Max return (successful landing)

    # Optimization (Regularized)
    'learning_rate': 3e-4,             # Lower than Shared DQN
    'weight_decay': 1e-4,              # L2 regularization
    'gamma': 0.99,

    # Exploration
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,

    # Target network
    'target_update_freq': 10,          # Every 10 episodes

    # Evaluation
    'eval_freq': 50,
    'eval_episodes': 5,
    'save_freq': 100,

    # Task-specific timeouts (same as Shared DQN)
    'max_episode_steps': {
        'standard': 1000,
        'windy': 400,
        'heavy': 800,
    }
}
```

---

## 6. Training Details

### Files Created

**Implementation:**
- `agents/brc.py` (459 lines):
  - `ResidualBlock` - Residual block with LayerNorm
  - `BroNet` - Large network with 3 residual blocks
  - `BRCAgent` - Agent with categorical DQN
  - `MultiTaskReplayBuffer` - Shared buffer with task IDs

**Experiment Setup:**
- `experiments/brc/config.py` - BRC hyperparameters
- `experiments/brc/train.py` (316 lines) - Multi-task training loop
- `experiments/brc/evaluate.py` (267 lines) - Per-task evaluation

**Testing:**
- `test_brc.py` - Comprehensive test suite

### Test Results ✅

**All 7 tests passed:**
1. ✅ Agent creation
2. ✅ Forward pass (logits shape: [4, 4, 51], Q-values: [4, 4])
3. ✅ Action selection
4. ✅ Replay buffer (100 transitions, batch sampling)
5. ✅ Update step (loss computation)
6. ✅ Environment interaction (10 steps)
7. ✅ Save/load (model persistence)

### Training Strategy

**Same as Shared DQN:**
1. Round-robin task cycling (episode i → task i % 3)
2. Single shared replay buffer with task IDs
3. Mixed-task batches (gradient conflicts!)
4. Task-specific episode timeouts

**Key Difference:**
- 12× larger network to absorb gradient conflicts
- Categorical loss instead of MSE
- Weight decay for regularization

### Expected Training Time
- **~3-4 hours (Mac M1)**
- 1500 episodes total (500 per task)
- Similar to Shared DQN but slightly slower (larger network)

---

## 7. Expected Results

### Hypothesis

**BRC should outperform Shared DQN because:**
1. **Higher Capacity:** 459K params vs 37K (can learn task-specific features)
2. **Better Loss:** Categorical cross-entropy vs MSE (more stable)
3. **Residual Connections:** Prevent vanishing gradients in deep network
4. **Regularization:** Weight decay prevents overfitting

**BRC may approach Independent DQN because:**
1. Large network can dedicate different neurons to different tasks
2. 32-dim task embeddings provide strong task identification
3. Residual blocks allow gradient flow to task-specific layers

### Expected Performance

**Conservative Estimate:**
```
Shared DQN:     205.61 avg (actual)
BRC:            220-240 avg (expected: +10-15%)
Independent:    173.98 avg (baseline to beat)
```

**Optimistic Estimate:**
```
BRC could match or beat Shared DQN's 205.61 if:
- Categorical loss provides better value estimates
- Large capacity allows better task separation
- Regularization prevents overfitting
```

### Open Questions

1. **Will 12× more parameters help or hurt?**
   - Pro: More capacity to separate tasks
   - Con: Could overfit with only 500 episodes per task

2. **Is categorical loss necessary?**
   - Shared DQN did well with MSE
   - Cross-entropy may provide better gradients

3. **Can BRC beat Shared DQN's surprising performance?**
   - Shared DQN: 205.61 (+18% vs Independent)
   - High bar to beat!

---

## 8. What We're Testing

### Primary Goal
**Can a bigger network solve multi-task RL without gradient surgery?**

### Comparison Matrix

| Method | Params | Expected Avg Reward | Parameter Efficiency |
|--------|--------|---------------------|----------------------|
| Independent | 107K | 173.98 (actual) | 1.62e-3 |
| Shared DQN | 37K | 205.61 (actual) | 5.44e-3 |
| **BRC** | 459K | 220-240 (expected) | **0.48-0.52e-3** |

**Key Insight:**
BRC trades parameter efficiency for raw performance.

### Questions to Answer

1. **Does size matter?**
   - Can 12× parameters improve over Shared DQN?
   - Or is Shared DQN's 37K already sufficient?

2. **Is categorical better?**
   - Does distributional RL help multi-task learning?
   - Or is MSE loss sufficient?

3. **Do residual connections help?**
   - Can deep networks (3 blocks) learn better than shallow?
   - Or does 2-layer MLP suffice?

4. **Is regularization critical?**
   - Does weight decay prevent overfitting?
   - Or is 500 episodes enough to train 459K params?

---

## 9. Files to be Generated (After Training)

### Models
```
results/brc/
├── models/
│   ├── best.pth (459,820 params)
│   ├── checkpoint_ep100.pth
│   ├── checkpoint_ep200.pth
│   └── ... (every 100 episodes)
└── logs/
    └── metrics.json (all task data)
```

### Metrics to Log
- Episode rewards per task
- Average reward across tasks
- Loss values (cross-entropy)
- Epsilon (exploration rate)
- Q-value statistics (distributional)
- Gradient norms (conflict analysis)
- Evaluation history

---

## 10. Commands

### Training
```bash
# Start training
python -m experiments.brc.train

# Expected output:
# - Progress bar with 1500 episodes
# - Evaluation every 50 episodes
# - Checkpoints every 100 episodes
# - ~3-4 hours training time
```

### Evaluation
```bash
# Evaluate all tasks (20 episodes each)
python -m experiments.brc.evaluate --episodes 20

# Evaluate single task
python -m experiments.brc.evaluate --task windy --episodes 20

# Evaluate with rendering
python -m experiments.brc.evaluate --task heavy --render --episodes 5

# Load specific checkpoint
python -m experiments.brc.evaluate --model checkpoint_ep500 --episodes 20
```

### Analysis
```bash
# Generate BRC analysis plots
python -m experiments.analyze_results --method brc

# Compare all methods
python -m experiments.analyze_results --method all
python generate_comparison_plots.py
```

### Testing
```bash
# Run test suite
python test_brc.py

# Expected output:
# ✅ ALL TESTS PASSED!
# BRC implementation is ready for training.
```

---

## 11. What Makes BRC Different?

### vs Independent DQN

| Aspect | Independent | BRC |
|--------|------------|-----|
| **Networks** | 3 separate | 1 shared |
| **Parameters** | 107K | 459K (4.3×) |
| **Gradient Conflicts** | None | Yes (but absorbed) |
| **Transfer Learning** | None | Yes |
| **Training Time** | 6.5 hours | ~3 hours |

### vs Shared DQN

| Aspect | Shared DQN | BRC |
|--------|-----------|-----|
| **Parameters** | 37K | 459K (12.2×) |
| **Architecture** | 2-layer MLP | 3 residual blocks |
| **Loss** | MSE | Cross-entropy |
| **Task Embedding** | 8-dim | 32-dim (4×) |
| **Regularization** | None | Weight decay, LayerNorm, clipping |
| **Depth** | 2 hidden layers | 5 layers (input + 3 blocks + output) |

### Key Innovations

**1. Residual Connections:**
- Skip connections: `output = f(x) + x`
- Prevent vanishing gradients
- Enable deep multi-task networks

**2. Categorical DQN:**
- Learn distributions, not scalars
- More stable gradients (cross-entropy)
- Better uncertainty quantification

**3. Aggressive Regularization:**
- Weight decay: 1e-4
- LayerNorm: per-sample normalization
- Gradient clipping: max_norm=10.0
- Lower LR: 3e-4 vs 5e-4

**4. High Capacity:**
- 459K params can dedicate neurons to tasks
- 32-dim embeddings provide strong task signal
- 3 residual blocks allow hierarchical features

---

## 12. Success Criteria

### Minimum Success (Beat Shared DQN)
```
BRC avg reward > 205.61 (Shared DQN baseline)
```

### Good Success (Approach Independent DQN)
```
BRC avg reward ≈ 173.98 ± 10% (Independent DQN)
```

### Exceptional Success (Beat Both!)
```
BRC avg reward > max(205.61, 173.98)
```

### Per-Task Goals

**Standard:**
- Target: > 263 (beat Shared DQN)
- Stretch: > 270

**Windy:**
- Target: > 130 (beat Shared DQN)
- Stretch: > 150 (avoid hovering)

**Heavy:**
- Target: > 224 (beat Shared DQN)
- Stretch: > 230

---

## 13. Potential Issues

### Overfitting Risk
- 459K params with only 500 episodes per task
- Solution: Weight decay, early stopping

### Training Instability
- Large network + categorical loss
- Solution: Gradient clipping, LayerNorm, lower LR

### Slower Training
- 12× more parameters to update
- Solution: Accept 3-4 hours (still faster than 3 independent trainings)

### May Not Beat Shared DQN!
- Shared DQN's 205.61 is strong baseline
- Bigger isn't always better
- Will test hypothesis: capacity vs efficiency

---

## 14. Next Steps (After Training)

1. **Analyze Results:**
   - Compare BRC vs Shared vs Independent
   - Plot training curves (all 3 methods)
   - Calculate parameter efficiency

2. **Understand What Worked:**
   - Did categorical help?
   - Did residual connections help?
   - Did regularization prevent overfitting?

3. **Decide on PCGrad:**
   - If BRC beats Shared: Maybe gradient conflicts aren't the problem
   - If BRC fails: Maybe need gradient surgery
   - If Shared still wins: Maybe capacity isn't the answer

4. **Plan VarShare:**
   - Use best baseline (Shared or BRC)
   - Add Bayesian task-specific adapters
   - Test sparse parameter efficiency

---

## 15. The Big Picture

**BRC Tests the Hypothesis:**
> "Multi-task learning fails due to insufficient capacity.
> Give the network enough parameters and it can learn all tasks well."

**Three Possible Outcomes:**

**Outcome 1: BRC Wins**
- Confirms capacity hypothesis
- Large networks can absorb conflicts
- Proceed to VarShare (sparse adapters on BRC)

**Outcome 2: Shared DQN Still Wins**
- Rejects capacity hypothesis
- Suggests transfer learning > capacity
- Proceed to analyze why small networks win

**Outcome 3: Independent Still Wins**
- Suggests fundamental multi-task difficulty
- Gradient conflicts can't be absorbed
- Proceed to PCGrad (gradient surgery)

---

**Status:** ⏳ Implementation Complete, Ready to Train

**Expected Time:** ~3-4 hours

**Expected Result:** Will find out soon! 🚀

---

## 16. Quick Commands

### Training Commands

```bash
# Train BRC on all 3 tasks simultaneously (round-robin)
/opt/anaconda3/envs/mtrl/bin/python -m experiments.brc.train
```

### Evaluation Commands

```bash
# Activate conda environment
# (Use the full path if conda activate doesn't work)
/opt/anaconda3/envs/mtrl/bin/python -m experiments.brc.evaluate [OPTIONS]

# Evaluate Windy task with rendering (5 episodes)
/opt/anaconda3/envs/mtrl/bin/python -m experiments.brc.evaluate --task windy --episodes 5 --render

# Evaluate Heavy task (20 episodes, no rendering)
/opt/anaconda3/envs/mtrl/bin/python -m experiments.brc.evaluate --task heavy --episodes 20

# Evaluate Standard task with rendering
/opt/anaconda3/envs/mtrl/bin/python -m experiments.brc.evaluate --task standard --render

# Evaluate ALL 3 tasks (recommended - shows multi-task performance)
/opt/anaconda3/envs/mtrl/bin/python -m experiments.brc.evaluate --episodes 20

# Quick evaluation (default 20 episodes per task)
/opt/anaconda3/envs/mtrl/bin/python -m experiments.brc.evaluate
```

### Testing Commands

```bash
# Run comprehensive test suite (7 tests)
/opt/anaconda3/envs/mtrl/bin/python test_brc.py

# Test with verbose output
/opt/anaconda3/envs/mtrl/bin/python test_brc.py -v
```

### Analysis Commands

```bash
# Generate all analysis plots for BRC
/opt/anaconda3/envs/mtrl/bin/python -m experiments.analyze_results --method brc

# Generate comparison plots (Independent vs Shared vs BRC)
/opt/anaconda3/envs/mtrl/bin/python -m experiments.analyze_results --method all
/opt/anaconda3/envs/mtrl/bin/python generate_comparison_plots.py
```
