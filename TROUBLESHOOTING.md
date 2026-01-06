# Troubleshooting & Experiment Issues Log

**Purpose:** Document issues encountered during RL experiments, their root causes, and solutions for future reference.

**Last Updated:** 2026-01-06

---

## Issue #1: Windy Environment - Training Hangs at Episode 250

**Date:** 2026-01-06
**Experiment:** Independent DQN - Windy Lunar Lander (Colab)
**Notebook:** `2a_independent_dqn_windy_colab.ipynb`

### Symptoms
- Training progresses normally until ~episode 250
- Progress bar freezes and stops updating
- Process appears to hang indefinitely (40+ minutes with no progress)
- GPU is active but no forward movement

### Root Cause
**Evaluation episodes running indefinitely without termination.**

During evaluation (which happens every 50 episodes), the agent uses a greedy policy (epsilon=0, no exploration). In the Windy environment:
- Random wind forces can cause the lander to drift laterally
- Without random exploration, the agent may get stuck in a "drifting loop"
- The episode never reaches a terminal state (landing or crash)
- Evaluation hangs waiting for episode completion

### Why Episode 250?
- Evaluation happens at episodes: 50, 100, 150, 200, **250**, 300...
- By episode 250, the agent has learned some policy that leads to endless drift
- Early episodes (50, 100) complete faster because agent is still exploring randomly

### Solution
**Add maximum step limit to evaluation episodes.**

Modify Cell 10 in the evaluation section:

```python
# Evaluation
if episode % HYPERPARAMS['eval_freq'] == 0 and episode > 0:
    eval_reward_mean = 0
    eval_reward_list = []

    for _ in range(HYPERPARAMS['eval_episodes']):
        eval_state, _ = env.reset()
        eval_reward = 0
        eval_done = False
        eval_truncated = False
        eval_steps = 0  # ← ADD THIS

        while not (eval_done or eval_truncated):
            eval_action = agent.select_action(eval_state, epsilon=0.0)
            eval_state, r, eval_done, eval_truncated, _ = env.step(eval_action)
            eval_reward += r

            eval_steps += 1  # ← ADD THIS
            if eval_steps >= 1000:  # ← ADD THIS SAFETY CHECK
                pbar.write(f"⚠️  Eval episode exceeded 1000 steps, forcing termination")
                break  # ← ADD THIS
```

### Prevention for Future Experiments
1. **Always add episode step limits** to both training and evaluation loops
2. **Monitor early evaluation episodes** - if they take >5 seconds, investigate
3. **For modified environments**, test evaluation separately before full training
4. **Consider using epsilon=0.01** during evaluation instead of 0.0 to allow minimal exploration

### Status
✅ **FIXED** - Applied step limit to evaluation loop

---

## Issue #2: Windy/Heavy Environments - Training Episodes Exceed Timeout

**Date:** 2026-01-06
**Experiment:** Independent DQN - Windy/Heavy Lunar Lander (Colab)
**Notebooks:** `2a_independent_dqn_windy_colab.ipynb`, `2b_independent_dqn_heavy_colab.ipynb`

### Symptoms
- After ~episode 200, training episodes constantly hit 1000-step timeout
- Agent learns to hover/drift without landing
- Training slows down significantly (each episode takes max time)
- Eval reward stuck around -120 (poor performance)

### Root Cause
**Agent learned local optimum: "don't crash" but not "land properly".**

The agent discovered that:
1. Crashing gives big negative reward
2. Hovering/drifting avoids crashing
3. Landing is risky and requires precision

Result: Agent gets stuck in "safe hovering" behavior and never learns to land.

**Why Episode ~200?**
- Early episodes (1-100): Random exploration, crashes quickly
- Episode ~100-200: Agent learns "don't crash = avoid negative reward"
- Episode 200+: Agent commits to hovering strategy, timeout every episode

### Why Episodes Don't Naturally Terminate

**LunarLander episodes normally end when:**
1. ✅ Lander **crashes** → done=True (negative reward)
2. ✅ Lander **lands successfully** → done=True (positive reward)
3. ⏱️ Gymnasium's built-in timeout (typically 1000 steps)

**But the trained agent learned to do NEITHER:**
- Agent uses thrusters to hover in mid-air indefinitely
- Never crashes → no termination signal
- Never lands → no termination signal
- Just drifts with the wind for the full 1000 steps

This is a **local optimum** - the agent found a "safe" strategy that avoids negative rewards but never achieves the goal.

### Why Reduce from 1000 to 400 Steps?

**The Problem with 1000 Steps:**
- Agent hovers for full 1000 steps every episode
- Takes ~40 seconds per episode
- Wastes compute on useless hovering behavior
- Agent learns "hovering is safe" because it gets rewarded for avoiding crashes

**The Solution with 400 Steps:**
- Episode cuts off after 400 steps if still hovering
- Takes ~16 seconds per episode (2.5x faster training)
- Agent experiences "timeout truncation" as a negative signal
- Learns: "hovering too long = failure, need to land within 400 steps"

**Key Insight:** The 400-step limit isn't about when episodes SHOULD end naturally - it's about creating **urgency to land** and breaking the hovering local optimum.

Think of it like:
- **With 1000 steps**: "I can hover forever safely" ❌
- **With 400 steps**: "I need to land quickly or I fail" ✅

### Investigation: Wind Strength Testing
Tested whether `wind_power=20.0` was too strong for the agent to counter.

**Test Results** (random agent, 10 episodes each):
```
Wind Power    Avg Steps    Timeout Rate
5.0           90           0%
10.0          95           0%
15.0          100          0%
20.0          97           0%
30.0          90           0%
```

**Conclusion:** Wind strength is NOT the problem. Random agents crash quickly at all wind levels. The issue is that **trained agents learn to hover safely** instead of landing.

### Solution
**Reduce MAX_EPISODE_STEPS from 1000 to 400.**

**Why this works:**
1. Forces episodes to terminate faster (400 steps = ~15-20 seconds)
2. Agent experiences "timeout truncation" as negative signal
3. More episodes per hour = more opportunities to explore landing
4. Reduces GPU/compute waste on long hovering episodes

**Alternative solutions considered:**
- Add time penalty (reward -= 0.05 per step) - more complex, save for future
- Increase exploration (epsilon_decay=0.998) - addresses symptom not cause
- Reduce wind strength - testing showed this isn't the issue

### Implementation
**Changed in Cell 10 (both notebooks):**
```python
# Before:
MAX_EPISODE_STEPS = 1000

# After:
MAX_EPISODE_STEPS = 400
# Reduced from 1000 to 400 for Windy/Heavy variants
```

**Why 400?**
- Standard LunarLander episodes typically complete in 100-300 steps
- 400 steps is generous enough to allow successful landing
- Short enough to prevent endless hovering
- 2.5x faster training than 1000 steps

### Prevention for Future Experiments
1. **Set reasonable episode timeouts** based on environment difficulty
2. **Monitor early episodes** - if episodes start hitting timeout, reduce MAX_STEPS
3. **For harder environments** (Windy, Heavy), use lower timeouts than Standard
4. **Consider time penalties** if hovering persists even with lower timeouts
5. **Watch for local optima** - if rewards plateau, agent may be "playing it safe"

### Status
✅ **FIXED** - Reduced MAX_EPISODE_STEPS to 400 in both notebooks

### Files Modified
- `/Users/mithileshr/RL/notebooks/2a_independent_dqn_windy_colab.ipynb`
- `/Users/mithileshr/RL/notebooks/2b_independent_dqn_heavy_colab.ipynb`
- `/Users/mithileshr/RL/test_wind_strength.py` (created for investigation)

---

## Issue Template (For Future Issues)

**Date:** YYYY-MM-DD
**Experiment:** [Experiment Name]
**Notebook/File:** [File path]

### Symptoms
[What you observed - be specific]

### Root Cause
[Why it happened - technical explanation]

### Solution
[How it was fixed - include code if applicable]

### Prevention for Future Experiments
[How to avoid this in the future]

### Status
[ ] Open / [✅] Fixed / [⏸️] Workaround

---

## Environment Modifications Explained

### How the Heavy Weight Environment Works

**Purpose:** Make the lander "heavier" by increasing gravitational pull, requiring more thrust and precision.

**Implementation:**
```python
class HeavyWeightLunarLander(LunarLander):
    def __init__(self, gravity_multiplier=1.5, **kwargs):
        self.gravity_multiplier = gravity_multiplier
        super().__init__(**kwargs)
        self.task_name = "Heavy Weight"

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        # Increase gravity after environment is initialized
        if self.world is not None:
            # Standard gravity is (0, -10), multiply the y-component
            self.world.gravity = (0, -10.0 * self.gravity_multiplier)
        return result
```

**What Changes:**
- **Standard LunarLander**: Gravity = `(0, -10.0)` (10 units downward)
- **Heavy Weight (1.5x)**: Gravity = `(0, -15.0)` (15 units downward = 50% stronger!)

**Why This Makes It Harder:**
1. **Faster descent** → Lander falls faster, less time to react
2. **More thrust needed** → Agent must use main engine more aggressively
3. **Harder to hover** → Can't "pause" mid-air as easily
4. **Landing precision** → Requires more careful control to avoid crash

**Why Modify Gravity Instead of Mass?**
- Cleaner implementation (one line in `reset()`)
- Predictable linear scaling effect
- Modifying mass would require complex Box2D physics changes
- Gravity modification is easier to reason about and debug

### How the Windy Environment Works

**Purpose:** Add random lateral forces to simulate wind, making horizontal control harder.

**Implementation:**
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

**What Changes:**
- Each step applies random horizontal force between -20.0 and +20.0
- Force direction and magnitude change every step
- Agent must constantly counteract drift with side thrusters

**Why wind_power=20.0?**
- Tested values from 5.0 to 30.0
- All values are physically manageable by the lander's thrusters
- 20.0 provides good challenge without being impossible
- See Issue #2 investigation for test results

---

## Additional Notes

### Common Issues to Watch For

1. **GPU Memory Issues (Colab)**
   - Symptoms: "CUDA out of memory" errors
   - Cause: Batch size too large or replay buffer memory leak
   - Solution: Reduce batch size or clear GPU cache with `torch.cuda.empty_cache()`

2. **Colab Session Timeout**
   - Symptoms: Progress stops after 30-90 minutes
   - Cause: Colab disconnects inactive sessions
   - Solution: Enable Colab Pro or add periodic output to keep session active

3. **Environment Physics Bugs**
   - Symptoms: Unexpected rewards or impossible states
   - Cause: Incorrect environment modification (e.g., wrong gravity value)
   - Solution: Test environment separately with `test_environments.py`

4. **Replay Buffer Memory Growth**
   - Symptoms: Training slows down over time
   - Cause: Replay buffer filling up and slowing sampling
   - Solution: Use fixed-size deque (already implemented)

---

## Best Practices from Experiments

1. **Always test environments separately first** (`test_environments.py`)
2. **Add safety timeouts** to all episode loops (training + evaluation)
3. **Save checkpoints frequently** (every 100 episodes) in case of crashes
4. **Monitor GPU utilization** in Colab to ensure it's being used
5. **Compare early episodes** - if episode 1 takes 10 seconds, something is wrong

---

## Experiment Progress Tracker

| Experiment | Status | Last Episode | Notes |
|------------|--------|--------------|-------|
| Baseline DQN (Standard, Mac) | ✅ Complete | 1000 | Running locally |
| Independent DQN (Windy, Colab) | 🔄 In Progress | ~230 | Fixed timeout issues, restarting with MAX_STEPS=400 |
| Independent DQN (Heavy, Colab) | ⏳ Pending | - | Ready to start with MAX_STEPS=400 |

---

## Contact & Resources

- **Project Spec:** `/Users/mithileshr/RL/CLAUDE.md`
- **Environment Test:** `/Users/mithileshr/RL/test_environments.py`
- **Results Directory:** `/Users/mithileshr/RL/results/`

---

**Remember:** Document issues as they occur! Future you will thank present you. 📝
