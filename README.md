# CS 491 / CS 591 Deep Reinforcement Learning: Final Project (Project 3)

**Team:** Christian Maestas, Jaden Malone, Isaac Tapia  
**Algorithm:** Proximal Policy Optimization (PPO-Clip) with GAE  
**Environment:** `CarRacing-v3` (Continuous & Discrete Action Spaces)  

---

## 1. Installation

### Local / Teammate Environment
Using `uv` (recommended):
```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Or using standard `pip`:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2. Quick Verification (Smoke Test)

Run a short 5,000-timestep smoke test to verify CNN shapes, GPU detection, and logging:

```bash
python main.py \
  --mode train \
  --env CarRacing-v3 \
  --continuous \
  --total_timesteps 5000 \
  --timesteps_per_batch 1024 \
  --condition smoke_test \
  --seed 11
```

---

## 3. Running Experiments

Target evaluation seeds: `11, 22, 33, 44, 55`.

### Comparison 1: Advantage Estimation (Rewards-To-Go vs. GAE)
- **Rewards-To-Go (No GAE):**
  ```bash
  python main.py --condition adv_rtg --no_gae --seed 11
  ```
- **GAE ($\lambda = 0.95$, Baseline):**
  ```bash
  python main.py --condition adv_gae --use_gae --gae_lambda 0.95 --seed 11
  ```

### Comparison 2: PPO Clipping Threshold ($\epsilon \in \{0.1, 0.2, 0.3\}$)
- **$\epsilon = 0.1$:**
  ```bash
  python main.py --condition clip_0.1 --clip 0.1 --seed 11
  ```
- **$\epsilon = 0.2$ (Baseline):**
  ```bash
  python main.py --condition clip_0.2 --clip 0.2 --seed 11
  ```
- **$\epsilon = 0.3$:**
  ```bash
  python main.py --condition clip_0.3 --clip 0.3 --seed 11
  ```

### Comparison 3: Entropy Regularization ($\text{coef} \in \{0.0, 0.01\}$)
- **$\text{coef} = 0.0$:**
  ```bash
  python main.py --condition ent_0.0 --ent_coef 0.0 --seed 11
  ```
- **$\text{coef} = 0.01$ (Baseline):**
  ```bash
  python main.py --condition ent_0.01 --ent_coef 0.01 --seed 11
  ```

---

## 4. Submitting on UNM CARC (Slurm)

To run all 5 seeds in parallel across CARC GPU nodes:
```bash
sbatch --export=CONDITION=adv_gae,EXTRA_ARGS="--use_gae" scripts/run_carc.slurm
sbatch --export=CONDITION=adv_rtg,EXTRA_ARGS="--no_gae" scripts/run_carc.slurm
```

---

## 5. Aggregating Results & Generating Plots

After completing runs across seeds, run:
```bash
python analyze_results.py
```
This generates comparison figures with shaded $\pm 1$ standard deviation bands under `results/`.

---

## 6. Testing / Visual Evaluation of Checkpoints

To evaluate and visually render a trained checkpoint on a local display:
```bash
python main.py \
  --mode test \
  --env CarRacing-v3 \
  --continuous \
  --actor_model results/adv_gae/seed_11/checkpoints/actor_best.pth \
  --eval_episodes 5 \
  --render
```

