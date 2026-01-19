# Multi-Task Reinforcement Learning: VarShare Experiment

## Project Overview
Multi-Task RL experiments comparing different approaches on Lunar Lander variants.

**Reference:** VarShare Proposal.pdf

---

## Domain & Tasks

**Base Environment:** Lunar Lander (Gym's LunarLander-v2)

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
| **BRC** | BroNet + Categorical DQN (51 atoms) | 459,820 |

### SOTA Optimization (TODO)
- **PCGrad** - Project conflicting gradients (Priority 1)
- **GradNorm** - Balance gradient magnitudes (Priority 2)
- **CAGrad** - Conflict-averse gradient descent (Priority 3)

### VarShare (Optional)
Variational weight adapters: `W_m = θ + φ_m` with KL penalty for automatic sharing vs specialization.

---

## Project Structure
```
/RL
├── environments/lunar_lander_variants.py
├── agents/
│   ├── dqn.py, shared_dqn.py, brc.py
│   └── pcgrad.py, gradnorm.py (TODO)
├── experiments/
│   ├── independent_dqn/  (train.py, evaluate.py, config.py)
│   ├── shared_dqn/       (train.py, evaluate.py, config.py)
│   └── brc/              (train.py, evaluate.py, config.py)
├── utils/
│   └── replay_buffer.py, metrics.py, visualize.py
└── results/
    ├── standard/, windy/, heavy/  (Independent DQN)
    ├── shared_dqn/, brc/          (Multi-task methods)
    └── analysis/                   (Generated plots)
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

# Analysis
python -m experiments.analyze_results --method all
python generate_comparison_plots.py
```

---

## Current Status

**Phase 4: Advanced Baselines**

| Method | Standard | Windy | Heavy | Avg | Status |
|--------|----------|-------|-------|-----|--------|
| Independent DQN | 228 | 100 | 194 | 174 | ✅ Complete |
| Shared DQN | 263 | 130 | 224 | 206 | ✅ Complete |
| BRC | - | - | - | - | ⏳ Ready to train |

**Key Finding:** Shared DQN outperformed Independent by +18% (unexpected - multi-task transfer helps)

**Next Steps:**
1. Train BRC: `python -m experiments.brc.train`
2. Implement PCGrad (Priority 1)

---

## Evaluation Metrics

1. **Conflict Robustness** - Per-task + average rewards
2. **Sample Efficiency** - Steps to reach thresholds (50, 100, 150, 200)
3. **Parameter Efficiency** - Performance vs parameter count

---

## Critical Implementation Notes

### Environment Fixes
- **Heavy gravity persistence:** Must override `step()` to re-apply gravity (Box2D resets it)
- **Windy hovering:** Agent learns to hover; use 400-step timeout to force landing

### Multi-Task Training
- Round-robin task cycling for Shared DQN/BRC
- Task embeddings: 8-dim (Shared DQN), 32-dim (BRC)
- Single shared replay buffer

---

## References
- VarShare Proposal.pdf
- BRC: "Bigger, Regularized, Categorical" paper
- PCGrad: "Gradient Surgery for Multi-Task Learning"
