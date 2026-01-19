# Lessons Learned Checklist: Multi-Task RL Experiments

**Purpose:** Quick reference guide for future experiments based on findings from Independent DQN and Shared DQN experiments.

**Usage:** Review this checklist before starting new multi-task RL experiments to avoid common pitfalls and leverage successful strategies.

---

## ✅ Pre-Experiment Checklist

### Environment Setup

- [ ] **Verify environment modifications work correctly**
  - [ ] Test each variant individually with random policy
  - [ ] Confirm physics modifications persist throughout episodes
  - [ ] Check for physics engine reset bugs (e.g., Box2D gravity)
  - [ ] Validate environment class names and attributes

- [ ] **Set task-specific episode timeouts strategically**
  - [ ] Use timeout as curriculum design tool, not just safety
  - [ ] Shorter timeouts for preventing lazy strategies (hovering)
  - [ ] Longer timeouts for tasks needing exploration
  - [ ] Document rationale for each timeout value

- [ ] **Create verification script**
  - [ ] Manual testing of each environment variant
  - [ ] Print environment class name and parameters
  - [ ] Test with 5-10 random episodes per variant
  - [ ] Save verification script for future reference

### Hyperparameter Configuration

- [ ] **Start with task-specific configs, not one-size-fits-all**
  - [ ] Baseline config for easiest task
  - [ ] Tuned configs for harder tasks (lower LR, slower epsilon decay)
  - [ ] Document which tasks need special treatment

- [ ] **Set appropriate replay buffer sizes**
  - [ ] Larger min_replay_size for harder/stochastic tasks (2000 vs 1000)
  - [ ] Ensure sufficient diverse experiences before training
  - [ ] Consider task-specific buffer sizes if using separate buffers

- [ ] **Tune target network update frequency**
  - [ ] More frequent (every 10 eps) for easier/stable tasks
  - [ ] Less frequent (every 20 eps) for harder/unstable tasks
  - [ ] Monitor Q-value divergence to adjust

- [ ] **Consider task physics when setting hyperparameters**
  - [ ] Stronger gravity → slower learning rate, more exploration
  - [ ] Stochastic forces → larger replay buffer, slower epsilon decay
  - [ ] Deterministic physics → can train faster, less exploration needed

### Baseline Experiments

- [ ] **Always train single-task baselines first**
  - [ ] Provides upper bound on per-task performance
  - [ ] Reveals task-specific challenges (local optima, instability)
  - [ ] Establishes parameter count baseline
  - [ ] Validates environment and hyperparameters work

- [ ] **Train separate network per task (Independent DQN)**
  - [ ] Measure: final performance, sample efficiency, success rate
  - [ ] Save: checkpoints, training curves, evaluation videos
  - [ ] Document: unexpected behaviors, tuning needed

- [ ] **Run for sufficient episodes**
  - [ ] At least 1000-1500 episodes per task minimum
  - [ ] Wait for convergence before changing hyperparameters
  - [ ] Avoid premature stopping based on early success

---

## 📊 During Training Checklist

### Monitoring & Debugging

- [ ] **Watch for local optima and unexpected strategies**
  - [ ] Review evaluation videos, not just reward curves
  - [ ] Check for hovering, circling, or other lazy behaviors
  - [ ] Monitor episode length (hitting timeout = warning sign)
  - [ ] Look for "safe but suboptimal" strategies

- [ ] **Track multiple metrics beyond reward**
  - [ ] Episode length (steps per episode)
  - [ ] Loss values and gradient norms
  - [ ] Q-value statistics (mean, max, variance)
  - [ ] Success rate (task-specific threshold, e.g., reward ≥ 200)
  - [ ] Epsilon (exploration rate)

- [ ] **Evaluate periodically during training**
  - [ ] Every 50-100 episodes with greedy policy (epsilon=0)
  - [ ] Use task-specific timeouts for consistency
  - [ ] Save best checkpoint based on evaluation, not training
  - [ ] Multiple evaluation episodes (5-10) for stability

- [ ] **Log comprehensive training data**
  - [ ] Per-episode: reward, loss, steps, epsilon, task_id
  - [ ] Periodic eval: mean reward, std, success rate, avg steps
  - [ ] Hyperparameters and config
  - [ ] Total environment steps and gradient updates (sample efficiency)

### Multi-Task Specific

- [ ] **For shared networks, track task-wise performance separately**
  - [ ] Don't just average across tasks
  - [ ] Watch for one task degrading while others improve (negative transfer)
  - [ ] Monitor gradient magnitudes per task
  - [ ] Check for catastrophic forgetting in round-robin cycling

- [ ] **Verify task conditioning works correctly**
  - [ ] Print task IDs during training
  - [ ] Confirm task embeddings/IDs match environments
  - [ ] Test: does agent behave differently given different task IDs?
  - [ ] Validate replay buffer stores task information

- [ ] **Consider gradient conflicts as potential feature, not bug**
  - [ ] Don't immediately apply PCGrad/GradNorm
  - [ ] Vanilla shared network may outperform "optimized" methods
  - [ ] Conflicts can provide regularization
  - [ ] Test baseline shared network first

---

## 🔬 Post-Training Checklist

### Evaluation & Validation

- [ ] **Run comprehensive final evaluation**
  - [ ] 20-50 episodes per task (more for stochastic environments)
  - [ ] Multiple independent runs (3-5) with different seeds
  - [ ] Report mean ± std across runs
  - [ ] Use task-specific timeouts matching training

- [ ] **Compare training vs evaluation performance**
  - [ ] Last 100 training episodes vs greedy evaluation
  - [ ] Large discrepancy = overfitting or timeout mismatch
  - [ ] Evaluation can show higher variance due to random seeds

- [ ] **Verify environment correctness (again)**
  - [ ] Create verification script that loads trained model
  - [ ] Test on each environment variant separately
  - [ ] Print environment class names to confirm correct variant
  - [ ] Check for physics bugs that only appear with trained policy

- [ ] **Test for evaluation variance in stochastic environments**
  - [ ] Run evaluation multiple times (2-3 runs)
  - [ ] If high variance (std > 50), need more episodes
  - [ ] Report both training metrics and evaluation metrics
  - [ ] Consider averaging across multiple evaluation runs

### Analysis & Reporting

- [ ] **Calculate parameter efficiency**
  - [ ] Total parameters per method
  - [ ] Reward per 1K parameters
  - [ ] Compare multi-task vs sum of single-task
  - [ ] Plot parameter count vs performance (Pareto frontier)

- [ ] **Measure sample efficiency**
  - [ ] Episodes to reach thresholds (50, 100, 150, 200 reward)
  - [ ] Total environment steps
  - [ ] Total gradient updates
  - [ ] Compare multi-task vs single-task sample efficiency

- [ ] **Analyze per-task performance**
  - [ ] Don't just report average across tasks
  - [ ] Check if one task sacrificed for others
  - [ ] Look for positive/negative transfer patterns
  - [ ] Identify which tasks benefit most from sharing

- [ ] **Document unexpected findings**
  - [ ] What contradicted your hypotheses?
  - [ ] What performance differences were surprising?
  - [ ] Any bugs or issues discovered?
  - [ ] What worked better/worse than expected?

### Visualization

- [ ] **Generate comprehensive plots**
  - [ ] Training curves (reward over episodes, smoothed)
  - [ ] Per-task performance comparison
  - [ ] Parameter efficiency scatter plot
  - [ ] Sample efficiency curves
  - [ ] Conflict robustness (average + per-task)

- [ ] **Create comparison plots for multiple methods**
  - [ ] Side-by-side bar charts
  - [ ] Overlaid training curves
  - [ ] Parameter vs performance scatter
  - [ ] Summary dashboard (2×2 grid)

- [ ] **Save evaluation videos**
  - [ ] Record greedy policy on each task
  - [ ] Useful for debugging unexpected behaviors
  - [ ] Can reveal strategies not obvious from metrics

---

## 🚨 Common Pitfalls to Avoid

### Environment Issues

- [ ] ❌ **Using same timeout for all tasks**
  - ✅ Use task-specific timeouts based on physics
  - ✅ Document timeout rationale in config

- [ ] ❌ **Forgetting to maintain physics modifications**
  - ✅ Override `step()` to reapply modifications each step
  - ✅ Especially important for Box2D environments

- [ ] ❌ **Not verifying environments before training**
  - ✅ Create and run verification script
  - ✅ Test with random policy first

### Hyperparameter Issues

- [ ] ❌ **Using same hyperparameters for all tasks**
  - ✅ Tune per-task based on difficulty
  - ✅ Harder tasks need slower LR, more exploration

- [ ] ❌ **Changing hyperparameters too early**
  - ✅ Wait 500-600 episodes before tuning
  - ✅ Early improvements can be misleading

- [ ] ❌ **Not adjusting min_replay_size**
  - ✅ Larger buffer for harder tasks (2000 vs 1000)
  - ✅ Ensures diverse experiences before training

### Training Issues

- [ ] ❌ **Stopping training at first sign of success**
  - ✅ Train for full episode count
  - ✅ Early success may be local optimum

- [ ] ❌ **Only monitoring average reward**
  - ✅ Track episode length, success rate, Q-values
  - ✅ Watch evaluation videos for behaviors

- [ ] ❌ **Ignoring task-specific local optima**
  - ✅ Watch for hovering, circling, etc.
  - ✅ Use timeout strategically to discourage

### Multi-Task Issues

- [ ] ❌ **Assuming gradient conflicts always hurt**
  - ✅ Test vanilla shared network first
  - ✅ Conflicts can provide beneficial regularization

- [ ] ❌ **Not tracking per-task performance separately**
  - ✅ Log task ID with each episode
  - ✅ Analyze per-task curves, not just average

- [ ] ❌ **Using wrong task IDs or embeddings**
  - ✅ Verify task ID → environment mapping
  - ✅ Print and validate during training

### Evaluation Issues

- [ ] ❌ **Only running 20 episodes for stochastic tasks**
  - ✅ Use 50+ episodes or multiple runs
  - ✅ Report std dev and variance

- [ ] ❌ **Using different timeouts for training vs eval**
  - ✅ Match evaluation timeouts to training
  - ✅ Inconsistency inflates/deflates results

- [ ] ❌ **Comparing training metrics to eval metrics**
  - ✅ Compare like-to-like (training-to-training or eval-to-eval)
  - ✅ Note that eval can show higher variance

---

## 💡 Best Practices Discovered

### Transfer Learning

✅ **Try shared networks first**
- May outperform task-specific networks
- 3.4× more parameter efficient in our experiments
- Don't assume gradient conflicts are always harmful

✅ **Mixed replay buffer can be beneficial**
- Provides natural curriculum learning
- Helps escape task-specific local optima
- Forces learning of generalizable features

✅ **Round-robin task cycling works well**
- Simple and effective
- No need for complex task sampling initially
- Can optimize later if needed

### Environment Design

✅ **Deterministic challenges easier than stochastic**
- Heavy (1.25× gravity) easier than Windy (random wind)
- Predictable physics → faster learning
- Stochastic forces → require more exploration

✅ **Timeout is a curriculum design tool**
- Short timeout: Forces decisive action
- Long timeout: Allows exploration
- Tune per-task based on desired behavior

✅ **Physics engine bugs are common**
- Always override `step()` for modifications
- Don't trust physics to persist automatically
- Verify with random policy tests

### Training Strategy

✅ **Task-specific hyperparameters essential**
- Harder tasks need: slower LR, more exploration, less frequent target updates
- Can't use one-size-fits-all approach
- Document tuning rationale

✅ **Wait for convergence before tuning**
- 500-600 episodes minimum before changing hyperparameters
- Early success can be local optimum
- Patience pays off

✅ **Monitor episode length as key signal**
- Hitting timeout frequently = warning sign
- Can indicate hovering, inefficiency, or wrong timeout
- Use alongside reward for full picture

### Evaluation

✅ **Multiple evaluation runs for stochastic environments**
- Single 20-episode run has high variance
- Use 2-3 independent runs or 50+ episodes
- Report mean ± std

✅ **Training metrics more stable than eval**
- Last 100 training episodes averaged
- Less variance than small eval samples
- Use both for complete picture

✅ **Record videos during evaluation**
- Reveals strategies not obvious from metrics
- Catches unexpected behaviors (hovering, etc.)
- Essential for debugging

### Analysis

✅ **Report parameter efficiency**
- Reward per 1K parameters
- Often overlooked but critical for multi-task
- Shared networks can be 3-4× more efficient

✅ **Don't just average across tasks**
- Per-task breakdown essential
- Reveals positive/negative transfer
- Shows which tasks benefit from sharing

✅ **Document unexpected findings**
- Often most scientifically interesting
- Challenge assumptions and conventional wisdom
- Lead to new research questions

---

## 🎯 Quick Reference: Task-Specific Settings

### Standard Task (Easiest)
```python
{
    'max_episode_steps': 1000,      # Episodes finish naturally in 100-300 steps
    'learning_rate': 5e-4,          # Standard rate
    'epsilon_decay': 0.995,         # Standard decay
    'target_update_freq': 10,       # Frequent updates OK
    'min_replay_size': 1000,        # Standard buffer
}
```

### Windy Task (Hardest - Stochastic)
```python
{
    'max_episode_steps': 400,       # Force landing urgency (prevent hovering)
    'learning_rate': 2.5e-4,        # Halved for stability
    'epsilon_decay': 0.992,         # Slower decay (more exploration)
    'target_update_freq': 20,       # Less frequent for stability
    'min_replay_size': 2000,        # Larger buffer for diversity
}
```

### Heavy Task (Medium - Deterministic)
```python
{
    'max_episode_steps': 800,       # Allows full descent with 1.25× gravity
    'learning_rate': 2.5e-4,        # Halved for stability
    'epsilon_decay': 0.992,         # Slower decay
    'target_update_freq': 20,       # Less frequent
    'min_replay_size': 2000,        # Larger buffer
}
```

### Shared DQN (Multi-Task)
```python
{
    'max_episode_steps': {          # Task-specific timeouts
        'standard': 1000,
        'windy': 400,
        'heavy': 800,
    },
    'learning_rate': 5e-4,          # Use Standard (easiest) task params
    'epsilon_decay': 0.995,
    'target_update_freq': 10,
    'min_replay_size': 2000,        # Match harder tasks for stability
    'embedding_dim': 8,             # Learned task embeddings
}
```

---

## 📚 Additional Resources

### Scripts to Run Before Each Experiment

```bash
# 1. Verify environments
python verify_environments.py

# 2. Test single task baseline
python -m experiments.independent_dqn.train  # Edit TASK_NAME

# 3. Train multi-task method
python -m experiments.shared_dqn.train

# 4. Comprehensive evaluation
python verify_shared_dqn.py  # Or equivalent for your method

# 5. Generate analysis plots
python -m experiments.analyze_results --method all
python generate_comparison_plots.py
```

### Files to Check

- [ ] `claude.md` - Project documentation
- [ ] `EXPERIMENTAL_RESULTS.md` - Full analysis from previous experiments
- [ ] `TROUBLESHOOTING.md` - Known issues and solutions
- [ ] `EXECUTIVE_SUMMARY.md` - Quick reference for findings
- [ ] This checklist!

### Metrics to Always Track

- [ ] Episode reward (per episode)
- [ ] Episode length (steps)
- [ ] Loss values
- [ ] Epsilon (exploration rate)
- [ ] Q-value statistics (mean, max)
- [ ] Success rate (reward ≥ threshold)
- [ ] Evaluation performance (periodic)
- [ ] Total environment steps
- [ ] Total gradient updates

---

## ✨ Success Criteria

**Before starting next experiment, confirm:**

- [x] Verified all environments work correctly
- [x] Set task-specific hyperparameters with documented rationale
- [x] Trained single-task baselines (Independent DQN)
- [x] Achieved reasonable performance on all tasks
- [x] Documented all unexpected findings
- [x] Generated comprehensive analysis plots
- [x] Reviewed this checklist

**After completing experiment, confirm:**

- [ ] Ran evaluation with sufficient episodes (20-50)
- [ ] Calculated parameter efficiency metrics
- [ ] Analyzed per-task performance (not just average)
- [ ] Generated comparison plots
- [ ] Documented lessons learned
- [ ] Updated this checklist with new findings

---

## 🔄 Version History

**v1.0** (2026-01-07)
- Initial checklist based on Independent DQN and Shared DQN experiments
- Covers environment setup, hyperparameter tuning, multi-task training
- Includes common pitfalls and best practices discovered
- Task-specific configuration quick reference

**Future Updates:**
- Add findings from BRC experiments
- Add findings from PCGrad experiments
- Refine based on additional methods tested

---

**Remember:** This checklist is a living document. Update it after each experiment with new lessons learned!
