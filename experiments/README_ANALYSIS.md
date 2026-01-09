# Analysis & Evaluation Tools

Comprehensive tools for analyzing and comparing RL experiments.

---

## Quick Start

### 1. Generate All Plots & Analysis

```bash
cd /Users/mithileshr/RL
conda activate mtrl
python experiments/analyze_results.py
```

**Output**: Creates plots and summary in `results/plots/`:
- `training_curve_{task}.png` - Individual training curves
- `sample_efficiency_comparison.png` - Reward vs steps across tasks
- `parameter_efficiency_comparison.png` - Performance vs model size
- `multi_task_comparison.png` - Bar chart comparing tasks
- `analysis_summary.txt` - Text summary of all results

---

### 2. Evaluate Trained Model

```bash
# Analyze training results + evaluate model
python experiments/independent_dqn/evaluate.py heavy

# Evaluate with visualization (watch agent play)
python experiments/independent_dqn/evaluate.py heavy --render

# Compare all tasks
python experiments/independent_dqn/evaluate.py --all

# Just analyze training metrics (no model testing)
python experiments/independent_dqn/evaluate.py heavy --analyze-only
```

---

### 3. Create Individual Plots

```python
from utils.visualize import plot_training_curve, plot_sample_efficiency
from pathlib import Path

# Plot single task training curve
plot_training_curve('heavy', save_path=Path('results/plots/heavy_training.png'))

# Compare sample efficiency across experiments
experiments = [
    ('standard', 'independent_dqn'),
    ('windy', 'independent_dqn'),
    ('heavy', 'independent_dqn')
]
plot_sample_efficiency(experiments, save_path=Path('results/plots/sample_eff.png'))
```

---

## Metrics Tracked

### Sample Efficiency ⚡
Measures how quickly the agent learns (fewer samples = better):
- **Total environment steps**: Total interactions with environment
- **Total gradient updates**: Total learning steps
- **Threshold achievements**: Episodes/steps when reaching 50, 100, 150, 200 reward

### Parameter Efficiency 🔢
Measures model size vs performance:
- **Parameter count**: Total trainable parameters
- **Performance**: Final average reward (last 100 episodes)
- **Ratio**: Performance per 1K parameters

### Performance Metrics 📊
- **Final avg reward**: Mean of last 100 episodes
- **Best eval reward**: Highest evaluation score
- **Success rate**: % of episodes with positive reward
- **Reward variance**: Standard deviation of final performance

---

## File Structure

```
experiments/
├── analyze_results.py          # Generate all plots and summaries
├── independent_dqn/
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation script
│   └── config.py               # Hyperparameters

utils/
├── metrics.py                  # Metrics computation
└── visualize.py                # Plotting functions

results/
├── logs/                       # JSON metrics files
│   ├── independent_dqn_standard_metrics.json
│   ├── independent_dqn_windy_metrics.json
│   └── independent_dqn_heavy_metrics.json
├── models/                     # Saved checkpoints
│   ├── independent_dqn_standard.pth
│   ├── independent_dqn_windy.pth
│   └── independent_dqn_heavy.pth
└── plots/                      # Generated plots
    ├── training_curve_*.png
    ├── sample_efficiency_comparison.png
    ├── parameter_efficiency_comparison.png
    └── analysis_summary.txt
```

---

## Metrics JSON Format

Each `*_metrics.json` file contains:

```json
{
  "episode_rewards": [list of rewards per episode],
  "episode_losses": [list of losses per episode],
  "eval_rewards": [evaluation rewards],
  "eval_episodes": [episodes when evaluated],
  "total_env_steps": total environment interactions,
  "total_gradient_updates": total learning updates,
  "performance_thresholds": {
    "50": {"episode": 123, "total_steps": 10000, "gradient_updates": 5000},
    "100": {...},
    "150": {...},
    "200": {...}
  },
  "timestamp": "2026-01-06 23:09:33"
}
```

---

## Usage Examples

### Example 1: Analyze Heavy Task Training

```bash
python experiments/independent_dqn/evaluate.py heavy --analyze-only
```

**Output**:
```
================================================================================
EXPERIMENT SUMMARY: INDEPENDENT_DQN - HEAVY
================================================================================

📊 PERFORMANCE METRICS:
  Final Avg Reward (last 100): 240.56 ± 25.33
  Best Eval Reward: 263.66
  Success Rate: 95.0%

⚡ SAMPLE EFFICIENCY:
  Total Environment Steps: 652,847
  Total Gradient Updates: 580,123
  Steps per Episode (avg): 435.2

🎯 PERFORMANCE THRESHOLDS:
  Reward ≥  50: Episode  780 | Steps: 340,000 | Updates: 305,000
  Reward ≥ 100: Episode  850 | Steps: 370,000 | Updates: 332,000
  Reward ≥ 150: Episode  920 | Steps: 400,000 | Updates: 359,000
  Reward ≥ 200: Episode 1000 | Steps: 435,000 | Updates: 391,000
```

### Example 2: Watch Trained Agent Play

```bash
python experiments/independent_dqn/evaluate.py heavy --render --episodes 5
```

Opens Gym window showing agent landing the lander!

### Example 3: Compare All Tasks

```bash
python experiments/independent_dqn/evaluate.py --all
```

**Output**:
```
================================================================================
MULTI-TASK MODEL COMPARISON
================================================================================

Task             Mean Reward     Success Rate
--------------------------------------------------------------------------------
Standard         235.42          92.0%
Windy            156.78          68.0%
Heavy            240.56          95.0%
================================================================================
```

---

## Adding New Experiments

When you implement Shared DQN, PCGrad, etc.:

1. **Save metrics** in same JSON format:
   ```python
   save_progress_checkpoint(
       episode_rewards, episode_losses, eval_rewards, eval_episodes,
       task_name='windy',
       total_env_steps=total_env_steps,
       total_gradient_updates=total_gradient_updates,
       performance_thresholds=performance_thresholds
   )
   ```

2. **Update analysis script** to include new experiment:
   ```python
   experiments = [
       ('heavy', 'independent_dqn'),
       ('heavy', 'shared_dqn'),
       ('heavy', 'pcgrad'),
   ]
   ```

3. **Re-run analysis**:
   ```bash
   python experiments/analyze_results.py
   ```

All plots and comparisons will update automatically!

---

## Troubleshooting

### "No metrics found for {task}"
- Check if training completed and saved metrics
- Verify file exists: `results/logs/independent_dqn_{task}_metrics.json`
- Re-run training or check for errors

### "Module not found" errors
- Activate conda environment: `conda activate mtrl`
- Ensure you're in project root: `cd /Users/mithileshr/RL`
- Check all dependencies installed: `pip install matplotlib seaborn`

### Plots look wrong
- Check if JSON metrics contain `total_env_steps` and `performance_thresholds`
- May need to re-run training with updated `train.py` to save new metrics

---

## Next Steps

After completing Independent DQN baselines:
1. Implement **Shared DQN** (single network for all tasks)
2. Implement **PCGrad** (gradient surgery for multi-task)
3. Re-run analysis to compare all methods
4. Use these plots for your final report!

Refer to `CLAUDE.md` for full project roadmap.
