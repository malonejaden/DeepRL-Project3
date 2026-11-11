# Project 3 Predictions and Reflection

**Authors:** Isaac Tapia & Jaden Malone  
**Environment:** `CarRacing-v3` (Continuous Action Space)  
**Algorithm:** Proximal Policy Optimization (PPO-Clip)  
**Evaluation Seeds:** `11, 22, 33, 44, 55`  

---

## Pre-run Predictions

*(Complete this section before viewing the final experiment results. Do not modify original predictions after running.)*

### 1. Advantage Estimation (Rewards-to-Go vs. GAE $\lambda=0.95$)
- **Condition Pair:**
  - `adv_rtg`: `--no_gae` (Monte Carlo Rewards-to-Go, $A_t = R_t - V(s_t)$)
  - `adv_gae`: `--use_gae --gae_lambda 0.95` (Generalized Advantage Estimation)
- **Which condition do you expect to learn faster?**
  - *Prediction:* GAE ($\lambda=0.95$) is expected to learn faster.
- **Which do you expect to be more stable across seeds?**
  - *Prediction:* GAE ($\lambda=0.95$).
- **Why?**
  - In `CarRacing-v3`, episodes span hundreds of frames. Pure Rewards-to-Go sums all future rewards without discounting TD errors, yielding high-variance advantage estimates that can destabilize policy updates. GAE exponentially discounts multi-step TD residuals, reducing advantage variance while properly bootstrapping state value $V(s_T)$ on time-limit truncations.

---

### 2. PPO Clipping Threshold ($\epsilon \in \{0.1, 0.2, 0.3\}$)
- **Condition Set:**
  - `clip_0.1`: `--clip 0.1` (Conservative trust region)
  - `clip_0.2`: `--clip 0.2` (Standard baseline)
  - `clip_0.3`: `--clip 0.3` (Aggressive policy updates)
- **Which clipping ratio do you expect to yield the best final performance?**
  - *Prediction:* $\epsilon = 0.2$.
- **Which do you expect to be most stable vs. prone to divergence?**
  - *Prediction:* $\epsilon = 0.1$ will be the most stable but slowest to make progress; $\epsilon = 0.3$ is most prone to policy collapse.
- **Why?**
  - Clipping limits how far the updated policy $\pi_\theta$ can diverge from the rollout behavior policy $\pi_{\theta_{old}}$. A small clip ($\epsilon=0.1$) limits gradient steps excessively when learning complex steering dynamics, while a wide clip ($\epsilon=0.3$) permits large policy swings on noisy image minibatches, risking destructive updates from which an on-policy method struggles to recover.

---

### 3. Entropy Regularization ($\text{coef} = 0.0$ vs. $\text{coef} = 0.01$)
- **Condition Pair:**
  - `ent_0.0`: `--ent_coef 0.0` (No entropy regularization)
  - `ent_0.01`: `--ent_coef 0.01` (Active exploration bonus)
- **Which do you expect to explore more effectively in early training?**
  - *Prediction:* $\text{coef} = 0.01$.
- **Do you expect the difference to persist late in training?**
  - *Prediction:* The difference will be most pronounced in early-to-mid training; unregularized runs are more likely to get stuck in local minima (e.g. timid driving or repetitive off-track skids).
- **Why?**
  - In visual continuous control, policy variance can prematurely shrink before the agent discovers effective throttle and steering combinations across varied track curves. An entropy bonus discourages premature determinism, maintaining exploration until the value baseline and policy gradients guide the car onto high-reward track segments.

---

## Post-run Reflection

*(Fill out after all runs on UNM CARC / teammate GPU are complete and analyzed via `analyze_results.py`.)*

### 1. Advantage Estimation Reflection
- *Observed outcome:*
- *Surprises:*
- *Impact on variance & AUC:*

### 2. PPO Clipping Threshold Reflection
- *Observed outcome:*
- *Comparison of $\epsilon=0.1, 0.2, 0.3$:*

### 3. Entropy Regularization Reflection
- *Observed outcome:*
- *Early vs. late training dynamics:*

