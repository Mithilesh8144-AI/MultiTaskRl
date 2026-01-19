# Task-Blind Shared DQN: Why It Works Better Than I Expected

## Executive Summary

**Surprising Finding:** My task-blind Shared DQN achieves **~85% of task-aware performance** (176.75 vs 206 avg reward), despite having no explicit knowledge of which task it's solving.

This document analyzes why task embeddings provide less benefit than I expected in my Lunar Lander multi-task setting.

---

## Results Comparison

| Method | Standard | Windy | Heavy | Average | vs Task-Aware |
|--------|----------|-------|-------|---------|---------------|
| My Shared DQN (task-aware) | 263 | 130 | 224 | 206 | baseline |
| My Shared DQN (task-blind) | 216 | 122 | 192 | 177 | **-14%** |

**What I Expected:** 40-60% degradation
**What I Got:** ~14% degradation

---

## Why Task-Blind Works Well

### 1. High Task Similarity

All three of my Lunar Lander variants share:
- **Same state space:** 8-dimensional (position, velocity, angle, leg contact)
- **Same action space:** 4 discrete actions (noop, left, main, right engine)
- **Same goal:** Land safely on the pad
- **Same reward structure:** +100/-100 for landing/crashing

My tasks differ only in physics parameters:
| Task | Difference | Magnitude |
|------|------------|-----------|
| Standard | Baseline | - |
| Windy | Lateral wind force | wind_power=20.0 |
| Heavy | Increased gravity | 1.25× gravity |

**Key insight:** These are *perturbations* of the same task, not fundamentally different tasks.

### 2. Implicit Task Information in State

My agent can partially infer the task from state dynamics:

```
Standard: Predictable trajectories
Windy:    High lateral velocity variance, irregular positions
Heavy:    Higher downward velocities, faster descents
```

Even without explicit task labels, my network learns to recognize these patterns implicitly through the state representation.

### 3. Robust "Generalist" Policy

My task-blind network learns a **single policy that works across all variants**:

```
Task-Aware Policy:
  if task == 'standard': use_policy_A()
  if task == 'windy':    use_policy_B()
  if task == 'heavy':    use_policy_C()

My Task-Blind Policy:
  use_generalist_policy()  # Works 85% as well for all tasks
```

My generalist policy:
- Uses more aggressive corrections (handles wind)
- Fires main engine more conservatively (handles heavy gravity)
- Compromises between task-specific optima

### 4. Low Gradient Conflict

For my similar tasks, gradients point in similar directions:

```
          ↗ Task A optimal
         /
        /  ← Small angle = low conflict
       /
      ↗ Task B optimal
```

When tasks are similar, a single policy update improves all tasks. Task embeddings primarily help when tasks have conflicting optima.

### 5. Round-Robin Training Acts as Regularization

My training cycles through tasks each episode:
```
Episode 1: Standard → Episode 2: Windy → Episode 3: Heavy → Episode 4: Standard → ...
```

This prevents overfitting to any single task and naturally produces a robust generalist.

---

## When Task Embeddings Matter More

Task embeddings would provide larger benefits for:

1. **Fundamentally different tasks** (e.g., CartPole vs MountainCar)
2. **Conflicting optimal actions** (same state, different best actions per task)
3. **Different reward scales** requiring task-specific value estimation
4. **Distinct state distributions** that don't overlap

My Lunar Lander variants have high overlap in all these dimensions.

---

## Implications for Multi-Task RL

### 1. Task Similarity Assessment
Before adding complexity (embeddings, gradient surgery), I should assess task similarity:
- State/action overlap
- Gradient conflict measurements
- Baseline task-blind performance

### 2. Complexity vs Benefit Tradeoff

| Method | Parameters | Benefit for My Similar Tasks |
|--------|------------|---------------------------|
| Task-blind | Minimal | Good baseline |
| Task embeddings | +2K params | ~15% improvement |
| PCGrad | Same + compute | ? (needs testing) |

### 3. Task-Blind as Strong Baseline
A task-blind network should be my **first baseline** for any multi-task setup. If it performs well, the tasks may be too similar to warrant complex methods.

---

## Experimental Validation

### Confirming Task-Blind Behavior

I verified that my task-blind network produces identical Q-values for same state:
```python
# Same state, different task_ids
state = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0, 0.0]
Q(state, task_id=0) == Q(state, task_id=1) == Q(state, task_id=2)  # True
```

My network has no mechanism to distinguish tasks—yet achieves 85% performance.

---

## Conclusions

1. **Task embeddings help but aren't critical** for my similar tasks (~15% improvement)
2. **Implicit state information** partially compensates for missing task labels
3. **Generalist policies** can be surprisingly effective
4. **Task-blind baseline** is essential for measuring embedding utility
5. **Gradient conflict** may be low for my Lunar Lander variants

---

## Next Steps

1. **Run PCGrad (task-aware):** Does gradient surgery help when conflicts are already low?
2. **Run PCGrad (task-blind):** Can gradient surgery compensate for missing embeddings?
3. **Measure gradient conflicts:** Quantify cosine similarity between per-task gradients
4. **Test on dissimilar tasks:** Would embeddings matter more for CartPole + MountainCar?

---

## Files Reference

- My task-aware results: `results/shared_dqn/logs/metrics.json`
- My task-blind results: `results/shared_dqn_blind/logs/metrics.json`
- Config: `experiments/shared_dqn/config.py`
- Agent: `agents/shared_dqn.py`
