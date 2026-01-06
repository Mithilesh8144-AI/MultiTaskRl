# Multi-Task Reinforcement Learning: VarShare Experiment

## Project Overview
Implementation of Multi-Task RL experiments comparing different approaches on Lunar Lander variants, culminating in the VarShare method - a principled approach to task specialization via variational weight adapters.

**Reference Document:** VarShare Proposal.pdf (November 2025)

---

## 1. Domain & Tasks

**Base Environment:** Lunar Lander (Modified from Gym's LunarLander-v2)

**Three Task Variants:**
1. **Standard** - Unchanged baseline LunarLander-v2
2. **Windy** - Add random lateral wind force each step
3. **Heavy Weight** - Increase lander mass/gravity to make it heavier

**Implementation Notes:**
- Each environment should inherit from standard LunarLander-v2
- Only specific physics modifications needed (minimal changes)
- Must be testable individually

---

## 2. Baselines & Algorithms

### Standard Baselines

#### a) Independent DQN
- **Description:** Separate network for each task
- **Characteristics:**
  - Upper bound on performance
  - Lower bound on efficiency
  - No transfer learning between tasks
- **Implementation:** Train 3 completely separate DQN agents

#### b) Completely Shared DQN
- **Description:** Single network for all tasks
- **Characteristics:**
  - Expected to perform badly due to gradient conflict
  - Tasks may interfere with each other
- **Implementation:** One network conditioned on task ID

#### c) BRC (Bigger, Regularized, Categorical)
- **Description:** Bigger, Regularized, Categorical network
- **Characteristics:**
  - High-capacity value functions
  - Use categorical cross-entropy loss
  - Massive residual networks (BroNet architecture)
- **Implementation:** Large network with residual connections

### SOTA Optimization Methods

#### a) PCGrad (Priority 1 - Focus Here First)
- **Description:** Project Conflicting Gradients
- **Purpose:** Priority 1 for conflict resolution
- **Mechanism:** When gradients from different tasks conflict, project them to remove interference

#### b) GradNorm (Priority 2)
- **Description:** Gradient Normalization
- **Purpose:** Balance gradient magnitudes across tasks
- **Mechanism:** Dynamically adjust loss weights to balance learning rates

#### c) CAGrad (Priority 3 - Bonus)
- **Description:** Conflict-Averse Gradient Descent
- **Purpose:** Bonus/Priority 3 - focus on PCGrad first
- **Mechanism:** Find gradient direction that minimizes worst-case loss

### Our Method: VarShare (Optional/Advanced)

**Core Idea:** Variational Weight-Space Adapters for principled task specialization

**Key Concepts:**
1. **The Stability-Plasticity Dilemma:** Agents must be plastic enough to learn task-specific nuances but stable enough to maintain shared dynamics
2. **The Limitation of Unconstrained Embeddings:** Current methods (like BRC) use learned task embeddings without constraints, leading to:
   - Redundant parameterization (inefficient transfer)
   - Initialization failure (random policy for new tasks)
   - Gradient interference

**VarShare Solution:**
- Treat task-specific parameters as random variables constrained by a shared prior
- Use Evidence Lower Bound (ELBO) to balance sharing vs. specialization
- Automatic arbitration: share by default, specialize only when reward signal outweighs information cost

**Mathematical Formulation:**
- Effective weight matrix: `W_m = θ + φ_m`
  - θ: deterministic shared parameters
  - φ_m: stochastic task-specific residual parameters

- Prior (Hypothesis of Similarity): `p(φ_m) = N(0, σ²_prior * I)`
  - Fixed prior creates "gravity" pulling task parameters to shared base

- Variational Posterior: `q_m(φ_m) = N(μ_m, diag(σ²_m))`
  - μ_m: learned shift/specialization
  - σ_m: uncertainty/"looseness" of specialization

- ELBO Objective:
  ```
  L(θ, μ_m, σ_m) = E[log p(D_m|W_m)] - β * KL(q_m(φ_m) || p(φ_m))
                    \_____________/      \______________________/
                    RL Performance        Complexity Cost
  ```

**The "Adaptive Rubber Band" Intuition:**
- φ_m connected to origin by "rubber band" (KL penalty)
- Anchor (Prior): assumes φ_m = 0 by default (100% shared)
- Force (Likelihood): RL loss pulls φ_m to minimize error
- Elasticity (Variance): learned σ_m allows "loosening" to absorb conflict

**Why Sharing is Preferred:**
1. **Cost Asymmetry:** Updating θ is "free" (no KL penalty), updating μ_m incurs penalty
2. **Signal Amplification:** θ receives gradients from ALL M tasks, μ_m only from task m
3. Result: Optimizer moves θ to solve shared structure; specialization only when necessary

**Implementation Details:**
- Use Flipout Estimator for gradient stability
- Post-training pruning: remove μ_m where |μ_m| < ε
- Results in compressed model with sparse task-specific "diffs"

---

## 3. Evaluation Metrics

### Conflict Robustness
- **Average Reward** across all tasks
- **Per-Task Reward** (crucial to see if one task is sacrificed for another)

### Sample Efficiency
- Measure number of gradient steps and/or samples required to reach specific performance thresholds (e.g., "Solved")

### Parameter Efficiency
- Log total parameter count for all algorithms/models
- Compare **Parameters vs. Performance** (cost-benefit ratio)

---

## 4. Implementation Architecture

### Project Structure
```
/RL
├── claude.md                    # This file - project documentation
├── VarShare Proposal.pdf        # Reference paper
├── preview.webp                 # Project specification image
├── environments/
│   └── lunar_lander_variants.py # 3 environment variants
├── agents/
│   ├── dqn.py                   # Base DQN implementation
│   ├── independent_dqn.py       # Separate network per task
│   ├── shared_dqn.py            # Single shared network
│   ├── brc.py                   # BRC implementation
│   ├── pcgrad.py                # PCGrad optimizer
│   ├── gradnorm.py              # GradNorm (priority 2)
│   ├── cagrad.py                # CAGrad (bonus/priority 3)
│   └── varshare.py              # VarShare implementation (optional)
├── utils/
│   ├── replay_buffer.py         # Experience replay
│   ├── training.py              # Training utilities
│   ├── metrics.py               # Logging and evaluation
│   └── plotting.py              # Visualization
├── experiments/
│   ├── train_baselines.py       # Train Independent/Shared DQN
│   ├── train_brc.py             # Train BRC baseline
│   ├── train_pcgrad.py          # Train with PCGrad
│   ├── train_gradnorm.py        # Train with GradNorm
│   └── train_varshare.py        # Train VarShare (optional)
└── results/
    ├── plots/                   # Training curves
    ├── logs/                    # Training logs
    └── models/                  # Saved checkpoints
```

### DQN Network Architecture
- **Input:** State dimension (8 for Lunar Lander)
- **Hidden Layers:**
  - Layer 1: 256 units + ReLU
  - Layer 2: 128 units + ReLU
- **Output:** Number of actions (4 for Lunar Lander)

**Pseudocode:**
```python
state_dim → Linear(256) → ReLU → Linear(128) → ReLU → Linear(num_actions)
```

### DQN Hyperparameters (Implemented)

**Used in Baseline DQN and Independent DQN (all tasks):**
```python
HYPERPARAMS = {
    'num_episodes': 1000,
    'batch_size': 64,
    'replay_buffer_size': 100000,
    'min_replay_size': 1000,
    'learning_rate': 5e-4,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    'target_update_freq': 10,        # Update target network every 10 episodes
    'eval_freq': 50,                 # Evaluate every 50 episodes
    'eval_episodes': 5,
    'save_freq': 100,                # Save checkpoint every 100 episodes
}
```

**Episode Timeout (MAX_EPISODE_STEPS):**
- **Standard:** 1000 steps (default Gymnasium timeout)
- **Windy:** 400 steps (reduced to break hovering behavior)
- **Heavy:** 400 steps (reduced to break hovering behavior)

**Rationale for 400 steps:**
- Standard episodes typically complete in 100-300 steps
- 400 is generous for successful landing but prevents endless hovering
- 2.5x faster training than 1000 steps

---

## 5. Development Roadmap

### Phase 1: Environment Setup ✅
- [x] Create `claude.md` documentation
- [x] Implement 3 Lunar Lander variants
- [x] Test each environment individually
- [x] Verify physics modifications work correctly

### Phase 2: DQN Baseline ✅
- [x] Implement replay buffer
- [x] Implement base DQN agent
- [x] Train on standard Lunar Lander (Mac - running locally)
- [x] Verify training curves look reasonable

### Phase 3: Multi-Task Baselines 🔄 (In Progress)
- [x] Implement Independent DQN Colab notebooks (Windy + Heavy only)
- [x] Auto-detection for Colab/Mac environment
- [x] Implement training utilities and metrics logging
- [x] Fix evaluation timeout issues (Issue #1 - see TROUBLESHOOTING.md)
- [x] Fix training episode timeout issues (Issue #2 - MAX_EPISODE_STEPS = 400)
- [x] Fix Heavy notebook bugs (Cell 5 gravity implementation)
- [ ] Complete Windy training (Episode 300/1000 - stuck in local optimum, avg reward 35.57)
- [ ] Complete Heavy training (notebook ready to test)
- [ ] Collect and analyze results from all 3 tasks

### Phase 4: Advanced Baselines
- [ ] Implement BRC (larger network with categorical loss)
- [ ] Run BRC experiments

### Phase 5: SOTA Methods
- [ ] Implement PCGrad (Priority 1)
- [ ] Run PCGrad experiments
- [ ] Implement GradNorm (Priority 2)
- [ ] Run GradNorm experiments
- [ ] (Optional) Implement CAGrad

### Phase 6: VarShare (Optional/Advanced)
- [ ] Implement variational weight adapters
- [ ] Implement ELBO objective with KL penalty
- [ ] Implement Flipout estimator
- [ ] Run VarShare experiments
- [ ] Compare against all baselines

### Phase 7: Analysis & Visualization
- [ ] Plot training curves (rewards over time)
- [ ] Compare sample efficiency across methods
- [ ] Compare parameter efficiency
- [ ] Analyze per-task performance
- [ ] Generate comparison tables

---

## 6. Key Implementation Notes

### Environment Modifications

**Windy Variant:**
```python
class WindyLunarLander(LunarLander):
    def __init__(self, wind_power=20.0, **kwargs):
        super().__init__(**kwargs)
        self.wind_power = wind_power
        self.task_name = "Windy"

    def step(self, action):
        # Apply random lateral wind force
        wind_force = np.random.uniform(-self.wind_power, self.wind_power)
        if self.lander is not None:
            self.lander.ApplyForceToCenter((wind_force, 0.0), True)
        return super().step(action)
```
- **Parameter:** wind_power = 20.0 (tested range: 5.0-30.0, all manageable)
- **Effect:** Random horizontal force each step, requires constant counteraction

**Heavy Weight Variant:**
```python
class HeavyWeightLunarLander(LunarLander):
    def __init__(self, gravity_multiplier=1.5, **kwargs):
        self.gravity_multiplier = gravity_multiplier
        super().__init__(**kwargs)
        self.task_name = "Heavy Weight"

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        if self.world is not None:
            # Standard gravity is (0, -10), multiply the y-component
            self.world.gravity = (0, -10.0 * self.gravity_multiplier)
        return result
```
- **Parameter:** gravity_multiplier = 1.5 (gravity: -10.0 → -15.0)
- **Effect:** 50% stronger gravity = faster descent, more thrust needed
- **Note:** Gravity modification in reset() is cleaner than mass modification

### Multi-Task Training Strategy
1. **Batch Composition:** Sample equally from all task replay buffers
2. **Task ID:** Pass task index as additional input or use separate heads
3. **Gradient Aggregation:** Sum gradients across tasks (for shared network)

### Logging Requirements
For each training run, log:
- Episode reward (per task)
- Average reward (across tasks)
- Loss values
- Epsilon (exploration rate)
- Gradient norms (for conflict analysis)
- Q-value statistics

---

## 7. Success Criteria

### Minimum Viable Product (MVP)
1. All 3 environments working and testable
2. DQN successfully trains on standard Lunar Lander
3. Independent DQN trains on all 3 tasks separately
4. Shared DQN attempts to train on all 3 tasks
5. Basic plotting of training curves

### Full Implementation
1. All baselines implemented (Independent, Shared, BRC)
2. PCGrad implemented and tested
3. Comprehensive metrics collection
4. Comparative analysis plots
5. Clear demonstration of gradient conflict in Shared DQN
6. PCGrad shows improvement over Shared DQN

### Stretch Goals
1. GradNorm implementation
2. CAGrad implementation
3. VarShare implementation
4. Full comparative analysis across all methods
5. Parameter efficiency analysis (sparse VarShare adapters)

---

## 8. Expected Results

### Hypotheses
1. **Independent DQN:** Best per-task performance, but M times the parameters
2. **Shared DQN:** Poor performance due to gradient conflicts
3. **BRC:** Better than Shared DQN due to higher capacity
4. **PCGrad:** Significantly better than Shared DQN, approaching Independent DQN
5. **VarShare:** Best parameter efficiency, good performance, automatic sharing

### Key Comparisons
- **Performance vs Parameters:** Plot average reward vs total parameter count
- **Sample Efficiency:** Steps to reach 200 reward threshold (per method)
- **Conflict Analysis:** Per-task reward breakdown (ensure no task is sacrificed)

---

## 9. References

- **VarShare Proposal.pdf** - Full mathematical formulation and theoretical analysis
- Gym LunarLander-v2 documentation
- BRC Paper: "Bigger, Regularized, Categorical: High-Capacity Value Functions are Efficient Multi-Task Learners"
- PCGrad Paper: "Gradient Surgery for Multi-Task Learning"
- Variational Continual Learning papers

---

## 10. Quick Start Commands

```bash
# Test individual environments
python -m environments.lunar_lander_variants

# Train baseline DQN on standard task
python -m experiments.train_baselines --mode single --task standard

# Train Independent DQN on all tasks
python -m experiments.train_baselines --mode independent

# Train Shared DQN on all tasks
python -m experiments.train_baselines --mode shared

# Train with PCGrad
python -m experiments.train_pcgrad

# Visualize results
python -m utils.plotting --experiment all
```

---

## Notes & Observations

### Current Training Status (2026-01-06)

**Baseline DQN (Standard Task):** ✅ Complete (Running on Mac)
- Uses same hyperparameters as Independent DQN for fair comparison
- Results will be used for Standard task in final analysis

**Independent DQN - Windy:** 🔄 In Progress (Episode 300/1000)
- **Current Issue:** Agent stuck in local optimum (avg reward: 35.57)
- **Symptom:** Almost all episodes hitting 400-step timeout
- **Root Cause:** Agent learned to hover/drift safely but not land
- **Action:** Monitoring until episode 500-600 before adjusting hyperparameters
- **Notebook:** `notebooks/2a_independent_dqn_windy_colab.ipynb`

**Independent DQN - Heavy:** ⏳ Ready to Test
- Fixed critical bugs in Cell 5 (gravity implementation)
- Environment verified, ready for Mac testing
- **Notebook:** `notebooks/2b_independent_dqn_heavy_colab.ipynb`

---

### Key Lessons Learned

#### 1. Episode Timeout Tuning is Critical
- **Problem:** Standard timeout (1000 steps) allows agents to hover indefinitely
- **Solution:** Reduced MAX_EPISODE_STEPS to 400 for Windy/Heavy variants
- **Insight:** Timeout creates urgency to land, breaks "safe hovering" local optimum
- **Trade-off:** 400 steps still generous for successful landing but prevents endless drift

#### 2. Local Optima in RL Training
- Agents can learn "safe but non-optimal" strategies (e.g., hovering to avoid crashes)
- Early training reward improvements can be misleading
- Need to wait for sufficient exploration before changing hyperparameters (500-600 episodes)

#### 3. Environment-Specific Hyperparameters
- Modified environments (Windy/Heavy) may need different timeouts than Standard
- Wind strength (wind_power=20.0) is manageable - not the bottleneck
- Gravity multiplier (1.5x) significantly increases difficulty

#### 4. Colab/Mac Compatibility
- Auto-detection via `IS_COLAB = 'COLAB_GPU' in os.environ or 'google.colab' in sys.modules`
- Allows same notebook to run locally or on Colab seamlessly
- Critical for iterative testing and parallel experiments

---

### If Windy Training Still Stuck at Episode 500-600

**Option 1: Increase Exploration (Recommended First)**
```python
'epsilon_decay': 0.998  # Slower decay = more exploration
```

**Option 2: Add Time Penalty**
```python
# In training loop after env.step()
reward -= 0.05  # Penalty for each step spent in air
```

**Option 3: Reduce Wind Temporarily**
```python
# In Cell 5
wind_power = 10.0  # Reduce from 20.0
```

---

### Files Created/Modified

**Notebooks:**
- `notebooks/2a_independent_dqn_windy_colab.ipynb` - Colab/Mac compatible, MAX_EPISODE_STEPS=400
- `notebooks/2b_independent_dqn_heavy_colab.ipynb` - Fixed gravity implementation, ready to test

**Documentation:**
- `TROUBLESHOOTING.md` - Comprehensive issue tracking (Issues #1 and #2)
- `test_wind_strength.py` - Investigation script for wind power testing

**Key Fixes Applied:**
1. Evaluation timeout fix (Cell 10) - prevents hanging at episode 250
2. Training timeout reduction (400 steps) - breaks hovering local optimum
3. Heavy environment bug fixes (Cell 5) - correct gravity implementation
4. Colab/Mac auto-detection (Cell 1, Cell 6) - seamless environment switching

---

**Last Updated:** 2026-01-06
**Status:** Phase 3 - Multi-Task Baselines (In Progress)
**Next Step:** Monitor Windy training to episode 500-600, then evaluate if hyperparameter tuning needed
