# Experiment Documentation

All my experiment notes and analysis are organized here in first-person.

## Project Structure

```
/RL
├── docs/                            # All documentation (you are here)
│   ├── independent_dqn/
│   │   └── notes.md                # My Independent DQN experiment notes
│   ├── shared_dqn/
│   │   └── notes.md                # My Shared DQN experiment notes
│   ├── brc/
│   │   └── notes.md                # My BRC experiment notes
│   ├── pcgrad/
│   │   └── notes.md                # My PCGrad experiment notes
│   ├── gradnorm/
│   │   └── notes.md                # My GradNorm experiment notes
│   ├── analysis/
│   │   ├── comparison_results.md   # Full comparison of all methods
│   │   └── task_blind_analysis.md  # Why task-blind works well
│   └── IMPLEMENTATION_DETAILS.md   # Technical implementation notes
├── scripts/                         # Utility scripts
│   ├── tests/                      # Test and verification scripts
│   └── analysis/                   # Analysis and plotting scripts
└── notebooks/                       # Jupyter notebooks for exploration
```

## Quick Links

### Baselines
- [Independent DQN](independent_dqn/notes.md) - Separate networks per task
- [Shared DQN](shared_dqn/notes.md) - Single network with task embeddings
- [Shared DQN (Task-Blind)](shared_dqn/notes.md#task-blind-mode) - Single network without task embeddings

### Advanced Methods
- [BRC](brc/notes.md) - Large capacity network with distributional RL (failed)
- [PCGrad](pcgrad/notes.md) - Gradient surgery for multi-task learning
- [GradNorm](gradnorm/notes.md) - Adaptive loss balancing ⭐ Best performer

### Analysis
- [Comprehensive Comparison](analysis/comparison_results.md) - Full results table and method-by-method analysis
- [Task-Blind Analysis](analysis/task_blind_analysis.md) - Why task-blind works surprisingly well

## Key Findings

1. **Simple Shared DQN is strong:** Beat Independent DQN by +18%
2. **Task-blind works well:** Only 14% worse than task-aware for similar tasks
3. **GradNorm (task-blind) won:** +8.6% improvement over Shared DQN baseline
4. **Task embeddings hurt SOTA methods:** Both PCGrad and GradNorm failed catastrophically with task embeddings
5. **Bigger networks don't help:** BRC (297K params) underperformed Shared DQN (38K params) by 3×

## Best Method

**GradNorm (Task-Blind)** achieved the best overall performance:
- Average reward: 220.3 (+8.6% vs Shared DQN)
- Especially strong on Windy task: 168.8 (+46.5%)
- Only 35,716 parameters
- Stable, balanced training across all tasks

---

## Additional Resources

### Scripts
See [scripts/README.md](../scripts/README.md) for:
- Test scripts (test_brc.py, test_environments.py, etc.)
- Analysis scripts (generate_comparison_plots.py)
- Verification scripts (verify_shared_dqn.py)

### Notebooks
See [notebooks/README.md](../notebooks/README.md) for:
- Jupyter notebooks for exploration
- CartPole experiments
