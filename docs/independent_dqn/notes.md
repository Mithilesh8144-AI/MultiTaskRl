# Independent DQN - Experiment Notes

**Date:** 2026-01-07
**Status:** Complete
**Method:** Independent DQN (Separate networks per task)

---

## Overview

I trained 3 completely separate DQN networks, one for each task variant. This serves as my **upper bound** for multi-task methods since there are no gradient conflicts between tasks.

---

## Architecture

**Network Structure (per task):**
```
State (8) → Linear(256) → ReLU → Linear(128) → ReLU → Linear(4 actions)
```

**Parameters:**
- Per network: 35,716 parameters
- Total: 3 × 35,716 = **107,148 parameters**

**Key Characteristics:**
- Upper bound on performance (no gradient conflicts)
- Lower bound on efficiency (no transfer learning)
- Each task learns independently

---

## Task Variants

| Task | Modification | Physics | Timeout |
|------|-------------|---------|---------|
| **Standard** | None (baseline) | Gravity: -10.0 | 1000 steps |
| **Windy** | Random lateral wind | Wind: ±20.0 | 400 steps |
| **Heavy** | 1.25× gravity | Gravity: -12.5 | 800 steps |

### Why I Used Different Timeouts

- **Standard (1000):** Episodes naturally finish in 100-300 steps
- **Windy (400):** Prevents hovering behavior - creates landing urgency
- **Heavy (800):** Allows full descent with stronger gravity

---

## Hyperparameters

**Standard Task:**
```python
{
    'num_episodes': 1500,
    'batch_size': 64,
    'replay_buffer_size': 100000,
    'min_replay_size': 1000,
    'learning_rate': 5e-4,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    'target_update_freq': 10,
    'max_episode_steps': 1000
}
```

**Windy/Heavy Tasks (tuned for harder tasks):**
```python
{
    'min_replay_size': 2000,      # More samples before training
    'learning_rate': 2.5e-4,       # Halved for stability
    'epsilon_decay': 0.992,        # Slower decay, more exploration
    'target_update_freq': 20,      # Less frequent target updates
}
```

---

## Results

### Final Evaluation (20 episodes)

| Task | Mean Reward | Success Rate | Avg Steps |
|------|-------------|--------------|-----------|
| **Standard** | 228.19 | 95% | 165 |
| **Windy** | 100.03 | 90% | 366 |
| **Heavy** | 193.71 | 100% | 165 |
| **Average** | **174** | 95% | - |

### Training Time
- Standard: ~2 hours
- Windy: ~2 hours
- Heavy: ~2.5 hours
- **Total: ~6.5 hours** (Mac M1)

---

## Key Findings

### The Windy Hovering Problem

My Windy agent learned to **hover** rather than land:
- Wind forces make landing risky
- Hovering accumulates positive reward safely
- Hits timeout frequently (365 avg steps)

**What I tried:**
1. 800-step timeout → Agent hovers for 800 steps
2. 400-step timeout → Agent hovers for 400 steps

**Conclusion:** Hovering is a robust local optimum that's hard to escape via hyperparameter tuning.

### Heavy Task - Gravity Bug

**Problem:** Box2D resets gravity mid-episode

**Symptom:** Inconsistent physics, agent can't learn stable policy

**My Fix:** Override `step()` to re-apply gravity every step:
```python
def step(self, action):
    # Re-apply gravity before each step
    self.world.gravity = (0, self.gravity)
    return super().step(action)
```

---

## Sample Efficiency

Episodes I needed to reach reward thresholds:

| Task | 50 | 100 | 150 | 200 |
|------|----|-----|-----|-----|
| Standard | 150 | 200 | 300 | 400 |
| Windy | 300 | 500 | 744 | Never |
| Heavy | 200 | 350 | 500 | 624 |

---

## Files

| File | Purpose |
|------|---------|
| `agents/dqn.py` | Base DQN implementation |
| `experiments/independent_dqn/config.py` | Task-specific hyperparameters |
| `experiments/independent_dqn/train.py` | Training script |
| `experiments/independent_dqn/evaluate.py` | Evaluation script |
| `results/{task}/` | Saved models and metrics |

---

## Commands

### Training
```bash
# Edit TASK_NAME in train.py first
python -m experiments.independent_dqn.train
```

### Evaluation
```bash
python -m experiments.independent_dqn.evaluate --task standard --episodes 20
python -m experiments.independent_dqn.evaluate --task windy --episodes 20
python -m experiments.independent_dqn.evaluate --task heavy --episodes 20
```

### Analysis
```bash
python -m experiments.analyze_results --method independent_dqn
```

---

## Lessons Learned

### What Worked
- Task-specific hyperparameter tuning
- Task-specific episode timeouts
- Separate replay buffers per task
- Checkpointing every 100 episodes

### What Didn't Work
- Single timeout for all tasks → hovering
- Standard hyperparameters for harder tasks → instability

### Key Insights
1. **Modified environments need task-specific configs**
2. **Local optima are real** - Windy hovering is hard to escape
3. **Physics bugs matter** - Gravity persistence bug caused catastrophic failure
4. **Timeout creates urgency** - Critical for preventing lazy strategies

---

## Role in My Multi-Task Comparison

Independent DQN serves as my **baseline** for multi-task methods:

| Aspect | Independent DQN |
|--------|----------------|
| Performance | Upper bound (no conflicts) |
| Transfer | None (isolated learning) |
| Parameters | 107K (3 separate networks) |
| Training time | 6.5 hours (3 separate runs) |

**My Expectation:** Multi-task methods will show some degradation vs Independent DQN due to gradient conflicts.

**Actual result:** Shared DQN outperformed Independent by +18%! (unexpected)
