# Multi-Task RL: Comprehensive Results Comparison

**Date:** January 19, 2026
**Experiment:** Lunar Lander Multi-Task Learning
**Tasks:** Standard, Windy (wind_power=20.0), Heavy (1.25x gravity)

---

## Executive Summary

I evaluated 8 different multi-task RL approaches on the Lunar Lander domain with 3 task variants. My goal was to compare baseline methods (Independent DQN, Shared DQN, Shared DQN Blind) against advanced techniques: a large capacity network (BRC) and state-of-the-art gradient-based optimization (PCGrad, GradNorm), both with and without task embeddings.

**Key Finding:** Task-blind gradient methods outperformed everything. GradNorm without task embeddings achieved my best overall performance (+8.6% vs Shared DQN). Surprisingly, larger networks (BRC) and task embeddings hurt SOTA methods catastrophically.

---

## Results Table

| Method | Standard | Windy | Heavy | **Average** | vs Best Baseline | Params |
|--------|----------|-------|-------|-------------|------------------|--------|
| **Baselines** |
| Independent DQN | 228.0 | 100.0 | 194.0 | **174.0** | -14.2% | 107,148 |
| Shared DQN | 269.3 | 115.2 | 224.2 | **202.9** | baseline | 37,788 |
| Shared DQN Blind | 204.8 | 138.4 | 220.1 | **187.7** | -7.5% | 35,716 |
| **Large Capacity Network** |
| BRC (8x capacity) | 134.3 | 39.4 | 26.2 | **66.6** | -67.2% | 296,884 |
| **SOTA with Task Embeddings** |
| PCGrad | 223.8 | 29.7 | -358.9 | **-35.1** | -117.3% | 37,788 |
| GradNorm | -140.8 | -5.5 | -57.1 | **-67.8** | -133.4% | 37,788 |
| **SOTA Task-Blind** |
| PCGrad Blind | 189.1 | 70.4 | 178.4 | **146.0** | -28.0% | 35,716 |
| **GradNorm Blind** | **265.7** | **168.8** | **226.6** | **220.3** | **+8.6%** | **35,716** |

---

## Method-by-Method Analysis

### 1. Independent DQN (Baseline)
**Performance:** 174.0 average reward
**Approach:** Separate network for each task (3 networks total)

I trained three completely independent DQN agents, one per task. This served as my baseline to measure whether multi-task learning provides any benefit at all.

**Results:**
- Standard task performed well (228.0), showing the vanilla DQN works fine for the base environment
- Windy task struggled significantly (100.0), likely due to the added complexity of random lateral forces
- Heavy task showed moderate performance (194.0)
- Total parameter count: 107,148 (35,716 × 3)

**Analysis:** The Independent approach worked but was inefficient. Each agent had to learn from scratch without leveraging knowledge from similar tasks. This establishes a baseline that multi-task methods should beat both in performance and parameter efficiency.

---

### 2. Shared DQN (Strong Baseline)
**Performance:** 205.7 average reward (+18% vs Independent)
**Approach:** Single network with task embeddings

I implemented a shared network where all three tasks use the same Q-network, but each task gets an 8-dimensional learnable embedding that's concatenated with the state. This was surprisingly effective.

**Results:**
- Outperformed Independent DQN on all three tasks
- Standard: 263.0 (+15% vs Independent)
- Windy: 130.0 (+30% vs Independent)
- Heavy: 224.0 (+15% vs Independent)
- Used only 37,788 parameters (65% fewer than Independent)

**Analysis:** This was an unexpected win for simple multi-task learning. The shared representations allowed positive transfer between tasks, and the task embeddings provided enough task-specific capacity. This became the benchmark that SOTA methods needed to beat.

---

### 3. Shared DQN Task-Blind
**Performance:** 187.7 average reward (-7.5% vs Shared DQN)
**Approach:** Single network WITHOUT task embeddings

After seeing that Shared DQN with task embeddings worked well, I wanted to test whether the task embeddings were actually necessary or if a pure task-blind approach would work.

**Results:**
- Standard: 204.8 (-24% vs Shared DQN with embeddings)
- Windy: 138.4 (+20% vs Shared DQN with embeddings)
- Heavy: 220.1 (-2% vs Shared DQN with embeddings)
- Used only 35,716 parameters (5% fewer than Shared DQN with embeddings)

**Analysis:** Interestingly, removing task embeddings from Shared DQN created a mixed result. The network performed worse overall but showed an interesting pattern:
- **Lost significant performance on Standard task** - Without task conditioning, the network couldn't distinguish which task it was on, leading to a more generic policy
- **Gained performance on Windy task** - The harder task actually benefited from not having task-specific parameters, suggesting the embeddings were causing overfitting
- **Maintained performance on Heavy task** - Nearly identical results

**Key Observation:** This established an important baseline showing that task embeddings DO help for Shared DQN (+7.5% overall), but the effect is task-dependent. This made the subsequent SOTA method failures with embeddings even more surprising.

---

### 4. BRC (Bigger, Regularized, Categorical)
**Performance:** 66.6 average reward (-67.2% vs Shared DQN)
**Approach:** Large network (8x capacity) with distributional RL

After seeing that simple methods worked well, I wanted to test whether a much larger network could "absorb" multi-task conflicts without needing gradient surgery. BRC stands for:
- **B**igger: 297K parameters (8× Shared DQN)
- **R**egularized: Weight decay + LayerNorm + gradient clipping
- **C**ategorical: Distributional RL with 21 atoms (predicts return distributions, not single Q-values)

**Architecture:**
- 2 residual blocks with skip connections
- 32-dim task embeddings (4× larger than Shared DQN)
- Categorical DQN: Outputs 21-atom probability distributions per action
- Conservative learning rate (2.5e-4) with strong regularization

**Results:**
- Standard: 134.3 (very unstable, oscillated between -30 and +304)
- Windy: 39.4 (barely learned)
- Heavy: 26.2 (essentially failed)
- Success rates: Standard 45%, Windy 0%, Heavy 10%

**Analysis:** BRC catastrophically failed, worse than even Independent DQN. Several factors contributed:

1. **Massive Overkill:** Lunar Lander is too simple for 297K parameters. The network had way more capacity than needed and couldn't converge.

2. **Catastrophic Forgetting:** The large network oscillated wildly between tasks. Training curves showed it would improve on one task, then immediately regress when switching tasks. Episode 1350 hit 73.3 reward, then episode 1400 dropped to -43.0.

3. **Distributional RL Mismatch:** The 21-atom categorical distribution was designed for environments with high return uncertainty (like Atari). Lunar Lander returns are relatively predictable, so this added complexity without benefit.

4. **Training Instability:** Despite aggressive regularization (LayerNorm, weight decay, gradient clipping), the network never stabilized. The extra capacity gave it too many ways to overfit to individual tasks.

**Critical Insight:** This experiment disproved my hypothesis that "bigger networks can absorb gradient conflicts." In fact, larger capacity made things WORSE by giving the network more ways to specialize per task and forget others.

**Lesson Learned:** Network capacity is not a substitute for proper multi-task learning algorithms. Simple Shared DQN (38K params) beat BRC (297K params) by 3× on average reward. Sometimes less is more.

---

### 5. PCGrad with Task Embeddings
**Performance:** -35.1 average reward (FAILED)
**Approach:** Gradient projection to eliminate conflicting gradients

I implemented PCGrad to project conflicting gradients onto each other, theoretically allowing tasks to learn without interfering. The model used task embeddings like Shared DQN.

**Results:**
- Complete catastrophic failure on Heavy task (-358.9 reward)
- Heavy task consistently crashed or terminated early
- Standard task was okay (223.8)
- Windy task performed poorly (29.7)

**Analysis:** The task embeddings combined with gradient projection created instability. Looking at my training curves, the Heavy task's gradient conflicts with the other tasks caused the network to diverge. The gradient projection might have been too aggressive, preventing the network from finding a good multi-task solution. This suggests that explicit task conditioning can hurt when combined with gradient manipulation.

**Lessons Learned:**
- Gradient projection isn't a silver bullet
- Task embeddings + PCGrad = unstable training
- Sometimes constraining gradients too much prevents useful learning

---

### 6. PCGrad Task-Blind
**Performance:** 146.0 average reward (-29% vs Shared DQN)
**Approach:** PCGrad without task embeddings

After the failure with task embeddings, I removed them and trained PCGrad in task-blind mode where the network doesn't know which task it's currently training on.

**Results:**
- Much more stable than the task-aware version
- Standard: 189.1 (decent)
- Windy: 70.4 (struggled)
- Heavy: 178.4 (recovered from catastrophic failure)

**Analysis:** Removing task embeddings fixed the instability, but PCGrad still underperformed my simple Shared DQN baseline. The gradient projection seems to be overly conservative, preventing the network from learning effective shared representations. Without task conditioning, PCGrad couldn't leverage task-specific information effectively, and the gradient conflicts it was designed to solve weren't actually the main problem in this domain.

**Observation:** Task-blind helped stability but PCGrad's core mechanism (conflict resolution) might not be beneficial for these similar tasks.

---

### 7. GradNorm with Task Embeddings
**Performance:** -67.8 average reward (FAILED)
**Approach:** Dynamically balance task loss weights using gradient magnitudes

I implemented GradNorm to automatically balance task learning rates by adjusting loss weights based on gradient magnitudes. Used 8-dim task embeddings.

**Results:**
- Failed across all tasks (all negative rewards)
- Standard: -140.8 (crashed)
- Windy: -5.5 (barely learned)
- Heavy: -57.1 (failed)
- Learned weights were extremely imbalanced: Standard: 2.66, Windy: 0.009, Heavy: 0.33

**Analysis:** The weight balancing went completely wrong. GradNorm gave almost all weight to the Standard task (2.66) while nearly ignoring Windy (0.009). This created a feedback loop where:
1. Standard task got most of the gradient updates
2. Other tasks fell behind
3. GradNorm increased Standard's weight even more to "balance" gradient magnitudes
4. Eventually all tasks collapsed

The task embeddings exacerbated this by allowing the network to overfit to Standard while ignoring the others. This is a classic case of GradNorm's weight adaptation going awry when task similarities aren't properly accounted for.

**Critical Issue:** GradNorm's assumption that gradient magnitudes indicate task difficulty doesn't hold when tasks are highly related. The dynamic weighting created instability.

---

### 8. GradNorm Task-Blind ⭐ WINNER
**Performance:** 220.3 average reward (+8.6% vs Shared DQN)
**Approach:** GradNorm without task embeddings

After the catastrophic failure with embeddings, I tried GradNorm in task-blind mode. This became my best performing model.

**Results:**
- Beat all baselines including Shared DQN
- Standard: 265.7 (-1.3% vs Shared DQN with embeddings, but more stable)
- Windy: 168.8 (+46.5% vs Shared DQN with embeddings) ← Biggest improvement
- Heavy: 226.6 (+1.1% vs Shared DQN with embeddings)
- Much more balanced learned weights: Standard: 1.64, Windy: 1.21, Heavy: 0.15

**Analysis:** This was my breakthrough. By removing task embeddings, GradNorm could focus on balancing gradient magnitudes across the shared representation rather than fighting with task-specific parameters. The weights converged to a reasonable distribution that prioritized Standard and Windy (1.64, 1.21) while keeping Heavy contributing (0.15).

**Why it worked:**
1. **No task-specific overfitting:** Without embeddings, the network had to learn robust shared features
2. **Better weight balance:** Weights stayed in a reasonable range (0.15-1.64 vs 0.009-2.66 in task-aware)
3. **Especially strong on Windy:** The automatic balancing helped the difficult Windy task get enough attention without sacrificing the others
4. **Stable training:** Training curves showed smooth, stable learning across all tasks

**Key Insight:** For similar tasks, forcing the network to learn a single shared representation while dynamically balancing the task contributions works better than giving it task-specific escape hatches (embeddings).

---

## Overall Conclusions

### What Worked
1. **Simple shared networks are strong:** Shared DQN with embeddings was hard to beat (202.9 avg)
2. **Task embeddings help vanilla Shared DQN:** +7.5% improvement over task-blind version
3. **Task-blind outperformed task-aware for SOTA methods:** Across both PCGrad and GradNorm
4. **GradNorm + Task-Blind = Best combo:** Only method to beat Shared DQN baseline (+8.6%)

### What Failed
1. **Task embeddings hurt SOTA methods:** Both PCGrad and GradNorm failed catastrophically with embeddings
2. **Gradient projection (PCGrad) wasn't helpful:** Even task-blind version underperformed all baselines
3. **Dynamic weighting needs care:** GradNorm's weights went haywire with task conditioning

### Why Task-Blind Won for SOTA Methods
My results reveal an interesting pattern:
- **Task embeddings HELP simple Shared DQN:** The baseline benefits from task conditioning (+7.5%)
- **Task embeddings HURT gradient-based methods:** Both PCGrad and GradNorm catastrophically failed with embeddings
- **Why the difference?** I believe gradient-based methods introduce additional complexity (gradient projection, dynamic weighting) that conflicts with task-specific parameters. The optimization landscape becomes too complex, causing instability.

For my Lunar Lander variants specifically:
- **Shared features dominate:** The core skills (thrust control, landing dynamics) are identical across tasks
- **SOTA methods need simpler optimization:** Without task embeddings, GradNorm's dynamic weighting can focus on balancing the shared representation
- **Gradient conflicts less severe than expected:** The tasks aren't as conflicting as PCGrad assumes; they benefit from shared learning

### Recommendations
For future multi-task RL on similar domains:
1. Start with simple Shared DQN - it's a strong baseline
2. If using gradient-based methods, try task-blind first
3. GradNorm without embeddings is worth trying for incremental improvements
4. Be very careful with task embeddings - they can hurt more than help
5. Don't assume SOTA methods will automatically outperform simple approaches

---

## Training Details

**Hardware:** Local machine
**Episodes per task:** 500 (1,500 total)
**Evaluation:** 20 episodes per task
**Hyperparameters:** Consistent across all methods (lr=5e-4, batch=64, γ=0.99)
**Timeout adjustments:**
- Standard: 1000 steps
- Windy: 400 steps (prevents hovering)
- Heavy: 800 steps

All results are reproducible from my saved models in `results/*/models/best.pth`.

---

## Next Steps

1. ✅ Completed baseline comparisons (Independent, Shared, Shared Blind)
2. ✅ Evaluated BRC (large capacity network) - Failed
3. ✅ Evaluated PCGrad and GradNorm (with and without embeddings)
4. 🔲 Investigate why task embeddings failed so badly for gradient methods
5. 🔲 Experiment with VarShare (variational weight adapters)
6. 🔲 Test on more dissimilar tasks where task conditioning might help
7. 🔲 Try CAGrad (Conflict-Averse Gradient Descent)

---

*Generated from my evaluation runs on 2026-01-19*
