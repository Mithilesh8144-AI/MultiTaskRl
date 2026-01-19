# Shared DQN - Experiment Notes

**Date:** 2026-01-07
**Status:** Complete
**Method:** Shared DQN (Single network with task embeddings)

---

## Overview

I trained a single network conditioned on task ID for all 3 task variants. I used learned task embeddings to tell the network which task it's solving.

**Key Result:** Unexpectedly outperformed Independent DQN by +18%!

---

## Architecture

**Network Structure:**
```
State (8) + Task Embedding (8) → Linear(256) → ReLU → Linear(128) → ReLU → Linear(4 actions)
                 ↑
      Learned Embedding(3 tasks, 8 dim)
```

**Parameters:**
- Task embeddings: 3 × 8 = 24
- Input layer: (8 + 8) × 256 + 256 = 4,352
- Hidden layer: 256 × 128 + 128 = 32,896
- Output layer: 128 × 4 + 4 = 516
- **Total: 37,788 parameters** (65% fewer than Independent)

---

## Task Embeddings Explained

### The Core Question: How Does My Agent Know Which Task?

My agent receives both the state AND the task ID:
```python
action = agent.select_action(state, task_id=1, epsilon)
#                                    ↑
#                            "You're on Windy task"
```

This is NOT cheating - in real deployment, a lunar lander would know "windy planet" vs "heavy gravity planet".

### Why I Used Embeddings Instead of Raw Task ID

**Option 1: Raw ID (BAD)**
```python
input = [...state..., 1]  # Just the number
```
Problem: Creates false ordinal relationships (task 2 is not "twice" task 1)

**Option 2: One-Hot (OKAY)**
```python
task_id = 1 → [0, 1, 0]
```
Limited expressiveness (only 3 dimensions)

**Option 3: Learned Embedding (BEST - what I chose)**
```python
embedding_layer = nn.Embedding(3 tasks, 8 dimensions)
task_id = 1 → [0.12, -0.34, 0.56, -0.78, 0.91, -0.23, 0.45, -0.67]
```
- 8 dimensions (more expressive)
- Learned during training
- Network discovers optimal values

### What My Network Learned

**Start of training (random):**
```python
Standard: [0.02, -0.05, 0.08, -0.01, ...]
Windy:    [-0.03, 0.06, -0.02, 0.04, ...]
Heavy:    [0.04, -0.07, 0.01, -0.03, ...]
```

**After 1500 episodes (learned):**
```python
Standard: [0.52, -0.23, 0.15, -0.08, ...]  # Optimized via gradient descent
Windy:    [-0.73, 0.41, -0.56, 0.68, ...]
Heavy:    [0.61, -0.34, 0.22, -0.11, ...]
```

My network discovered what values work best for each task!

### Forward Pass

```python
def forward(self, state, task_id):
    # Step 1: Look up embedding
    task_emb = self.task_embedding(task_id)  # [batch, 8]

    # Step 2: Concatenate
    x = torch.cat([state, task_emb], dim=-1)  # [batch, 16]

    # Step 3: Process
    x = F.relu(self.fc1(x))  # [batch, 256]
    x = F.relu(self.fc2(x))  # [batch, 128]
    return self.fc3(x)       # [batch, 4] Q-values
```

---

## Task-Blind Mode

I also tested Shared DQN in **task-blind** mode (no embeddings):
- Set `use_task_embedding: False` in config
- Network receives only state, not task ID
- Same Q-values regardless of which task
- Tests if network can learn generalist policy

**Task-Blind Results:**
| Mode | Standard | Windy | Heavy | Avg |
|------|----------|-------|-------|-----|
| Task-Aware | 263 | 130 | 224 | 206 |
| Task-Blind | 216 | 122 | 192 | 177 |

Task-blind performs ~85% as well (only 14% worse), showing embeddings help but aren't critical for similar tasks.

---

## Multi-Task Training

### Round-Robin Task Cycling
```
Episode 0: Standard (task_id=0)
Episode 1: Windy (task_id=1)
Episode 2: Heavy (task_id=2)
Episode 3: Standard (task_id=0)
...
```

### Shared Replay Buffer
- Single buffer stores: `(state, action, reward, next_state, done, task_id)`
- Mixed transitions from all tasks
- Random batch sampling → gradient conflicts

### Gradient Conflicts
Each batch may contain transitions from different tasks. Gradients from different tasks sum together, potentially conflicting.

**My Expectation:** Conflicts hurt performance
**Actual Result:** Conflicts provided beneficial regularization!

---

## Configuration

```python
{
    'num_episodes_per_task': 500,     # 1500 total
    'batch_size': 64,
    'replay_buffer_size': 100000,
    'min_replay_size': 2000,
    'learning_rate': 5e-4,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    'target_update_freq': 10,

    # Task embeddings
    'embedding_dim': 8,
    'use_task_embedding': True,  # False for task-blind

    # Task-specific timeouts
    'max_episode_steps': {
        'standard': 1000,
        'windy': 400,
        'heavy': 800,
    },

    # Output directory
    'output_dir': 'shared_dqn',  # or 'shared_dqn_blind'
}
```

---

## Results

### Final Evaluation (20 episodes)

| Task | Shared DQN | vs Independent |
|------|------------|----------------|
| **Standard** | 263.09 | **+15.3%** |
| **Windy** | 129.54 | **+29.5%** |
| **Heavy** | 224.19 | **+15.7%** |
| **Average** | **205.61** | **+18.2%** |

### Training Time
- **~3 hours** (Mac M1)
- 2× faster than Independent DQN (6.5 hours)

### Parameter Efficiency
```
Independent: 107,148 params → 174 avg reward
Shared:       37,788 params → 206 avg reward

Performance per 1K params:
  Independent: 1.62
  Shared:      5.44 (3.4× better!)
```

---

## Why Shared DQN Won (Unexpected!)

**My Expectation:** 60% performance degradation due to gradient conflicts
**Actual Result:** 18% performance **improvement**

### Theory 1: Multi-Task Transfer Learning
- Tasks share underlying dynamics (same LunarLander physics)
- Shared network learns general control policy
- Transfer learning outweighs gradient conflicts

### Theory 2: Beneficial Regularization
- Gradient conflicts prevent overfitting
- Shared network forced to learn robust features

### Theory 3: Sample Efficiency
- Shared network sees diverse experiences from all tasks
- 1500 mixed episodes > 500 task-specific episodes

### Theory 4: Escaped Local Optima
- Windy: 129.54 (Shared) vs 100.03 (Independent)
- Shared DQN avoided hovering trap better!
- Gradients from Heavy/Standard pushed away from hovering

---

## Files

| File | Purpose |
|------|---------|
| `agents/shared_dqn.py` | SharedQNetwork, MultiTaskReplayBuffer, SharedDQNAgent |
| `experiments/shared_dqn/config.py` | Hyperparameters |
| `experiments/shared_dqn/train.py` | Round-robin training loop |
| `experiments/shared_dqn/evaluate.py` | Per-task evaluation |
| `results/shared_dqn/` | Models and metrics |
| `results/shared_dqn_blind/` | Task-blind results |

---

## Commands

### Training
```bash
# Task-aware (default)
python -m experiments.shared_dqn.train

# Task-blind (edit config first: use_task_embedding: False, output_dir: 'shared_dqn_blind')
python -m experiments.shared_dqn.train
```

### Evaluation
```bash
python -m experiments.shared_dqn.evaluate --episodes 20
python -m experiments.shared_dqn.evaluate --task windy --render
```

### Analysis
```bash
python -m experiments.analyze_results --method shared_dqn
python -m experiments.analyze_results --method shared_dqn_blind
```

---

## Key Takeaways

1. **Multi-task RL can outperform single-task** when tasks share structure
2. **Gradient conflicts aren't always bad** - can prevent overfitting
3. **8-dim task embeddings are sufficient** for 3 tasks
4. **Task-blind works surprisingly well** (~85% of task-aware)
5. **3.4× better parameter efficiency** than Independent DQN

---

## Comparison Table

| Metric | Independent | Shared | Winner |
|--------|-------------|--------|--------|
| Parameters | 107,148 | 37,788 | Shared (65% fewer) |
| Avg Reward | 173.98 | 205.61 | Shared (+18%) |
| Training Time | 6.5 hours | 3 hours | Shared (2× faster) |
| Perf/Param | 1.62e-3 | 5.44e-3 | Shared (3.4× better) |

**Verdict:** Shared DQN is the clear winner for my multi-task RL problem!
