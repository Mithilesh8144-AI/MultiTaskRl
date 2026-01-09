# Independent DQN Experiments

Train separate DQN agents on individual Lunar Lander task variants (Standard, Windy, Heavy).

This serves as the **upper bound baseline** - each task gets its own dedicated network with no multi-task interference.

## Quick Start

### 1. Choose Your Task

Edit `train.py` line 21:
```python
TASK_NAME = 'heavy'  # Options: 'standard', 'windy', 'heavy'
```

### 2. Run Training

```bash
cd /Users/mithileshr/RL
python experiments/independent_dqn/train.py
```

That's it! Training will start with task-specific hyperparameters from `config.py`.

## File Structure

```
experiments/independent_dqn/
├── train.py       # Main training script (change TASK_NAME here)
├── config.py      # Hyperparameter configurations for each task
└── README.md      # This file
```

## Hyperparameters

All hyperparameters are defined in `config.py`:

### Standard Task
- Episodes: 1000
- Max steps: 1000
- Learning rate: 5e-4
- Epsilon decay: 0.995
- Target update: every 10 episodes

### Windy Task
- Episodes: 1000
- Max steps: 400 ⚠️ (reduced to prevent hovering)
- Learning rate: 5e-4
- Epsilon decay: 0.995
- Target update: every 10 episodes

### Heavy Task
- Episodes: 1500 ⬆️ (harder task needs more training)
- Max steps: 800 ⬆️ (1.25x gravity needs more time)
- Learning rate: 2.5e-4 ⬇️ (halved for stability)
- Epsilon decay: 0.992 ⬇️ (slower decay = more exploration)
- Target update: every 20 episodes ⬆️ (less frequent for stability)
- Min replay size: 2000 ⬆️ (better initial data)

## Output

### Saved Models
- **Best model**: `results/models/independent_dqn_{task_name}.pth`
- **Checkpoints**: `results/models/independent_dqn_{task_name}_ep{episode}.pth` (every 100 episodes)

### Training Metrics
- **JSON logs**: `results/logs/independent_dqn_{task_name}_metrics.json` (saved every 50 episodes)

### Console Output
- Progress bar with live stats (reward, avg_100, epsilon, loss)
- Evaluation results every 50 episodes
- Performance threshold alerts (50, 100, 150, 200 reward)
- Final summary with sample efficiency metrics

## Modifying Hyperparameters

Just edit `config.py`:

```python
HEAVY_CONFIG = {
    **STANDARD_CONFIG,
    'task_name': 'heavy',
    'num_episodes': 1500,        # ← Change this
    'learning_rate': 2.5e-4,     # ← Or this
    # ...
}
```

No need to hunt through notebook cells!

## Expected Results

### Standard Task
- **Target**: 200+ reward (solved)
- **Episodes to solve**: ~500-800
- **Success rate**: 80%+

### Windy Task
- **Target**: 150+ reward (challenging)
- **Episodes to solve**: ~700-1000
- **Known issue**: May get stuck hovering at 400-step timeout (monitor until ep 500-600)

### Heavy Task
- **Target**: 50-100 reward (realistic for 1.25x gravity)
- **Episodes to solve**: ~1000-1200
- **Previous bugs**: All fixed (gravity persistence, timeout, hyperparams)

## Debugging

If training hangs or crashes:
1. Check `results/logs/independent_dqn_{task_name}_metrics.json` for last saved progress
2. Verify environment: `python -m environments.lunar_lander_variants`
3. Test agent: `python -m agents.dqn`
4. Check disk space for model checkpoints

## Next Steps

After training all 3 tasks:
1. Compare results in `results/logs/`
2. Implement Shared DQN (expect worse performance due to gradient conflict)
3. Implement PCGrad (expect to recover most of Independent DQN performance)
