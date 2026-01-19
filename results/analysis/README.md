# Analysis Results Directory

This directory contains comprehensive analysis plots for all multi-task RL experiments.

## Directory Structure

```
analysis/
├── README.md                    (this file)
├── comparisons/                 (cross-method comparison plots)
├── independent_dqn/            (baseline: separate network per task)
├── shared_dqn/                 (baseline: shared network WITH task embeddings)
├── shared_dqn_blind/           (shared network WITHOUT task embeddings)
├── brc/                        (Bigger, Regularized, Categorical DQN)
├── pcgrad/                     (PCGrad WITH task embeddings)
├── pcgrad_blind/               (PCGrad WITHOUT task embeddings)
├── gradnorm/                   (GradNorm WITH task embeddings)
└── gradnorm_blind/             (GradNorm WITHOUT task embeddings) ⭐ BEST
```

## Plot Types (per method)

Each method directory contains 7 plots:

1. **`{method}_conflict_robustness.png`**
   - Bar chart showing average reward + per-task breakdown
   - Compares performance across Standard, Windy, Heavy tasks

2. **`{method}_sample_efficiency_table.png`**
   - Table showing episodes/steps needed to reach reward thresholds
   - Thresholds: 50, 100, 150, 200 reward

3. **`{method}_sample_efficiency_curves.png`**
   - Line plots of cumulative reward vs episodes
   - Shows learning speed across all tasks

4. **`{method}_parameter_efficiency.png`**
   - Scatter plot: parameter count vs performance
   - Helps identify parameter-efficient methods

5. **`{method}_standard_training.png`**
   - Training curves for Standard task (episode reward + moving average)

6. **`{method}_windy_training.png`**
   - Training curves for Windy task

7. **`{method}_heavy_training.png`**
   - Training curves for Heavy task

## Quick Reference: Best Results

| Metric | Winner | Score |
|--------|--------|-------|
| **Overall Performance** | GradNorm Blind | 220.3 avg reward |
| **Standard Task** | GradNorm Blind | 265.7 reward |
| **Windy Task** | GradNorm Blind | 168.8 reward |
| **Heavy Task** | GradNorm Blind | 226.6 reward |
| **Parameter Efficiency** | Shared DQN Blind | 37,788 params, 206.8 avg |
| **Sample Efficiency (Standard)** | Shared DQN | Fastest to 200 reward |

## Key Findings

### ⭐ Winners
- **GradNorm Blind**: Only SOTA method to beat baselines (+7.1% vs Shared DQN)
- **Shared DQN**: Strong baseline, hard to beat
- **Task-Blind > Task-Aware**: Across all methods

### ❌ Failures
- **PCGrad with embeddings**: Catastrophic failure on Heavy task (-358.9)
- **GradNorm with embeddings**: Failed on all tasks (avg -67.8)
- **Task embeddings hurt**: Both SOTA methods performed worse with embeddings

## How to Navigate

### Want to see overall comparison?
→ Look in `comparisons/` folder

### Want to compare a specific method?
→ Open `{method}/` folder and check:
1. `conflict_robustness.png` for final performance
2. Training curves to see learning dynamics
3. `sample_efficiency_table.png` for learning speed

### Want to compare task-aware vs task-blind?
Compare:
- `pcgrad/` vs `pcgrad_blind/`
- `gradnorm/` vs `gradnorm_blind/`
- `shared_dqn/` vs `shared_dqn_blind/`

## Full Results Document

For detailed analysis and methodology explanations, see:
**`/results/COMPARISON_RESULTS.md`**

---

*Generated: 2026-01-19*
*Experiment: Multi-Task Lunar Lander (Standard, Windy, Heavy)*
