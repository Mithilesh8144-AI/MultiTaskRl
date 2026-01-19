# Multi-Task Reinforcement Learning: VarShare Experiment

## Project Overview
Multi-Task RL experiments comparing different approaches on Lunar Lander variants. This project evaluates various multi-task learning methods including baselines (Independent DQN, Shared DQN) and state-of-the-art gradient-based optimization techniques (PCGrad, GradNorm) with and without task embeddings.

**Reference:** VarShare Proposal.pdf

---

## Domain & Tasks

**Base Environment:** Lunar Lander (Gymnasium's LunarLander-v2)

**Three Task Variants:**
1. **Standard** - Unchanged baseline
2. **Windy** - Random lateral wind force each step (wind_power=20.0)
3. **Heavy** - 1.25x gravity multiplier

**Task-Specific Timeouts:**
- Standard: 1000 steps
- Windy: 400 steps (prevents hovering)
- Heavy: 800 steps (allows full descent)

---

## Algorithms

### Baselines
| Method | Description | Parameters |
|--------|-------------|------------|
| **Independent DQN** | Separate network per task | 35,716 × 3 = 107K |
| **Shared DQN** | Single network + task embeddings | 37,788 |
| **Shared DQN Blind** | Single network without task embeddings | 35,716 |

### Advanced Methods
| Method | Description | Parameters |
|--------|-------------|------------|
| **BRC** | Bigger, Regularized, Categorical DQN (21 atoms) | 296,884 |
| **PCGrad** | Project conflicting gradients | 37,788 |
| **PCGrad Blind** | PCGrad without task embeddings | 35,716 |
| **GradNorm** | Balance gradient magnitudes with adaptive task weights | 37,791 |
| **GradNorm Blind** | GradNorm without task embeddings | 35,719 |

### VarShare (Future Work)
Variational weight adapters: `W_m = θ + φ_m` with KL penalty for automatic sharing vs specialization.

---

## Project Structure
```
/RL
├── environments/
│   └── lunar_lander_variants.py
├── agents/
│   ├── dqn.py              # Independent DQN
│   ├── shared_dqn.py       # Shared DQN with task embeddings
│   ├── brc.py              # Bigger, Regularized, Categorical DQN
│   ├── pcgrad.py           # PCGrad implementation
│   └── gradnorm.py         # GradNorm implementation
├── experiments/
│   ├── independent_dqn/    # Independent DQN experiments
│   ├── shared_dqn/         # Shared DQN experiments
│   ├── brc/                # BRC experiments
│   ├── pcgrad/             # PCGrad experiments
│   ├── gradnorm/           # GradNorm experiments
│   └── analyze_results.py  # Analysis and comparison tools
├── utils/
│   ├── replay_buffer.py    # Experience replay implementations
│   ├── metrics.py          # Evaluation metrics
│   ├── visualize.py        # Plotting and visualization
│   └── plotting.py         # Additional plotting utilities
├── results/
│   ├── standard/, windy/, heavy/  # Independent DQN results
│   ├── shared_dqn/, brc/, pcgrad/, gradnorm/  # Multi-task methods
│   ├── shared_dqn_blind/, pcgrad_blind/, gradnorm_blind/  # Task-blind versions
│   └── analysis/           # Generated plots and comparisons
├── docs/                   # Detailed documentation and analysis
├── notebooks/              # Jupyter notebooks for exploration
└── scripts/                # Utility scripts
```

---

## DQN Architecture
```python
state_dim(8) → Linear(256) → ReLU → Linear(128) → ReLU → Linear(4)
```

### Hyperparameters
```python
# Standard task
HYPERPARAMS = {
    'num_episodes': 1500, 'batch_size': 64, 'replay_buffer_size': 100000,
    'min_replay_size': 1000, 'learning_rate': 5e-4, 'gamma': 0.99,
    'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.995,
    'target_update_freq': 10, 'eval_freq': 50, 'save_freq': 100,
}

# Windy/Heavy tasks (harder - need more stability)
HYPERPARAMS_HARD = {
    'min_replay_size': 2000, 'learning_rate': 2.5e-4,
    'epsilon_decay': 0.992, 'target_update_freq': 20,
}
```

---

## Quick Commands

```bash
# Test environments
python -m environments.lunar_lander_variants

# Independent DQN
python -m experiments.independent_dqn.train
python -m experiments.independent_dqn.evaluate --task heavy --episodes 20

# Shared DQN
python -m experiments.shared_dqn.train
python -m experiments.shared_dqn.evaluate --episodes 20

# BRC
python -m experiments.brc.train
python -m experiments.brc.evaluate --episodes 20

# PCGrad
python -m experiments.pcgrad.train
python -m experiments.pcgrad.evaluate --episodes 20

# GradNorm
python -m experiments.gradnorm.train
python -m experiments.gradnorm.evaluate --episodes 20

# Analysis
python -m experiments.analyze_results
python -m docs.analysis.comparison_results
```

---

## Final Results (Updated: January 19, 2026)

### Complete Results Table

| Method | Standard | Windy | Heavy | **Average** | vs Best Baseline | Params |
|--------|----------|-------|-------|-------------|------------------|--------|
| **Baselines** |
| Independent DQN | 228.0 | 100.0 | 194.0 | **174.0** | -14.2% | 107,148 |
| Shared DQN | 269.3 | 115.2 | 224.2 | **202.9** | baseline | 37,788 |
| Shared DQN Blind | 204.8 | 138.4 | 220.1 | **187.7** | -7.5% | 35,716 |
| **Large Capacity Network** |
| BRC (8x capacity) | 134.3 | 39.4 | 26.2 | **66.6** | -67.2% | 296,884 |
| **SOTA with Task Embeddings** |
| PCGrad | 223.8 | 29.7 | -358.9 | **-35.1** | -117.3% | 37,788 |
| GradNorm | -140.8 | -5.5 | -57.1 | **-67.8** | -133.4% | 37,788 |
| **SOTA Task-Blind** |
| PCGrad Blind | 189.1 | 70.4 | 178.4 | **146.0** | -28.0% | 35,716 |
| **GradNorm Blind** | **265.7** | **168.8** | **226.6** | **220.3** | **+8.6%** | **35,716** |

### Key Findings

**🏆 Winner:** GradNorm Blind achieved the best overall performance with 220.3 average reward (+8.6% vs Shared DQN baseline)

**Major Insights:**
1. **Task-blind methods outperformed task-aware methods** for gradient-based optimization
2. **Simple Shared DQN is a strong baseline** (202.9 average reward)
3. **Task embeddings hurt SOTA methods** - both PCGrad and GradNorm failed catastrophically with embeddings
4. **Larger networks don't help** - BRC (297K params) underperformed Shared DQN (38K params) by 3×
5. **GradNorm's dynamic balancing works best with task-blind networks**

**Method Performance Ranking:**
1. GradNorm Blind: 220.3 avg reward (winner)
2. Shared DQN: 202.9 avg reward (strong baseline)
3. Shared DQN Blind: 187.7 avg reward
4. Independent DQN: 174.0 avg reward
5. PCGrad Blind: 146.0 avg reward
6. BRC: 66.6 avg reward
7. GradNorm: -67.8 avg reward (failed)
8. PCGrad: -35.1 avg reward (failed)

---

## Current Status

**✅ All Experiments Completed**

| Method | Status | Performance | Notes |
|--------|--------|-------------|-------|
| Independent DQN | ✅ Complete | 174.0 avg | Baseline performance |
| Shared DQN | ✅ Complete | 202.9 avg | Strong baseline |
| Shared DQN Blind | ✅ Complete | 187.7 avg | Mixed results |
| BRC | ✅ Complete | 66.6 avg | Failed - overcapacity |
| PCGrad | ✅ Complete | -35.1 avg | Failed with embeddings |
| PCGrad Blind | ✅ Complete | 146.0 avg | Underperformed baseline |
| GradNorm | ✅ Complete | -67.8 avg | Failed with embeddings |
| GradNorm Blind | ✅ Complete | 220.3 avg | **Winner** |

**Completed Features:**
- ✅ All 8 methods implemented and tested
- ✅ Comprehensive analysis and comparison
- ✅ Task-aware vs task-blind comparison
- ✅ Parameter efficiency analysis
- ✅ Sample efficiency analysis
- ✅ Training curves and visualizations
- ✅ Detailed documentation and troubleshooting

**Next Steps:**
1. Implement VarShare (variational weight adapters)
2. Test on more dissimilar tasks where task conditioning might help
3. Try CAGrad (Conflict-Averse Gradient Descent)
4. Investigate why task embeddings failed so badly for gradient methods

---

## Evaluation Metrics

1. **Conflict Robustness** - Per-task + average rewards
2. **Sample Efficiency** - Steps to reach thresholds (50, 100, 150, 200)
3. **Parameter Efficiency** - Performance vs parameter count
4. **Training Stability** - Convergence and variance analysis

---

## Critical Implementation Notes

### Environment Fixes
- **Heavy gravity persistence:** Must override `step()` to re-apply gravity (Box2D resets it)
- **Windy hovering:** Agent learns to hover; use 400-step timeout to force landing

### Multi-Task Training
- Round-robin task cycling for Shared DQN/BRC
- Task embeddings: 8-dim (Shared DQN), 32-dim (BRC)
- Per-task replay buffers for PCGrad and GradNorm

### Key Discoveries
- **Task embeddings help simple methods but hurt complex ones**
- **Per-task buffers are critical for gradient-based methods**
- **Network capacity is not a substitute for proper multi-task learning**
- **GradNorm's dynamic balancing works best with task-blind networks**

---

## References
- VarShare Proposal.pdf
- BRC: "Bigger, Regularized, Categorical" paper
- PCGrad: "Gradient Surgery for Multi-Task Learning"
- GradNorm: "Gradient Normalization for Adaptive Loss Balancing"
- CAGrad: "Conflict-Averse Gradient Descent"
