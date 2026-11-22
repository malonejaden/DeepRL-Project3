# CS 491 / CS 591 Deep Reinforcement Learning: Final Project (Project 3)

**Team:** Christian Maestas, Jaden Malone, Isaac Tapia  
**Algorithm:** Proximal Policy Optimization (PPO-Clip) with GAE  
**Environment:** `CarRacing-v3` (Continuous Action Space)  
**Primary Study:** Baseline vs. Off-Road Early Termination Wrapper  
**Evaluation Seeds:** `11, 22, 33, 44, 55`  

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

Target evaluation seeds: `11, 22, 33, 44, 55` (500,000 timesteps per run).

### Primary Comparison: Baseline vs. Off-Road Termination Wrapper

In standard `CarRacing-v3`, episodes run until the full 1,000-frame limit or until all track tiles are visited. The custom `CarRacingOffRoadWrapper` (`off_road_wrapper.py`) hooks into Box2D's wheel contact listener to detect when the car drives completely off the asphalt track and terminates the episode immediately, preventing unproductive wandering in grass and accelerating training wall-clock time.

#### 1. Baseline (Standard Termination)
- **Direct CLI:**
  ```bash
  python main.py --mode train --env CarRacing-v3 --continuous --total_timesteps 500000 --condition baseline --seed 11
  ```
- **Automated Multi-Seed Scripts (Seeds 22, 33, 44, 55):**
  - PowerShell:
    ```powershell
    powershell -File scripts/trainingWallSeeds.ps1
    ```
  - Bash:
    ```bash
    bash scripts/trainingWallSeeds.sh
    ```

#### 2. Off-Road Termination Wrapper (`off_road_term`)
- **Direct CLI:**
  ```bash
  python main.py --mode train --env CarRacing-v3 --continuous --total_timesteps 500000 --condition off_road_term --terminate_on_off_road --seed 11
  ```
- **Automated Multi-Seed Script (Seeds 22, 33, 44, 55):**
  - PowerShell:
    ```powershell
    powershell -File scripts/trainingWallSeeds-off_road_term.ps1
    ```

---

### Secondary Hyperparameter Ablations

#### Advantage Estimation (Rewards-To-Go vs. GAE)
- **Rewards-To-Go (No GAE):**
  ```bash
  python main.py --condition adv_rtg --no_gae --seed 11
  ```
- **GAE ($\lambda = 0.95$, Baseline):**
  ```bash
  python main.py --condition adv_gae --use_gae --gae_lambda 0.95 --seed 11
  ```

#### PPO Clipping Threshold ($\epsilon \in \{0.1, 0.2, 0.3\}$)
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

#### Entropy Regularization ($\text{coef} \in \{0.0, 0.01\}$)
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

To run seeds in parallel across CARC GPU compute nodes:
```bash
sbatch --export=CONDITION=baseline scripts/run_carc.slurm
sbatch --export=CONDITION=off_road_term,EXTRA_ARGS="--terminate_on_off_road" scripts/run_carc.slurm
```

---

## 5. Aggregating Results & Generating Plots

To compute multi-seed metrics, generate comparison figures, and export summary tables:
```bash
python analyze_results.py
```

This generates:
- **`results/fig_baseline_vs_off_road.svg`**: Standalone vector comparison plot (individual runs + mean $\pm 1$ std band).
- **`results/fig_baseline_vs_off_road.png`**: High-resolution 300 DPI publication figure (when `matplotlib` is installed).
- **`results/summary_metrics.csv`**: Machine-readable aggregate metrics.
- **`results/summary_metrics.md`**: Formatted Markdown table.

### Multi-Seed Aggregate Results (Seeds 11, 22, 33, 44, 55)

| Condition | Seeds | Final Return (Mean ± Std) | Peak Return (Mean ± Std) | Normalized AUC (Mean ± Std) | Steps to 500 (Avg) | Solved (≥500) | Wall-Clock Time (hrs) | Final Ep Length |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`baseline`** (Standard Termination) | 5 | **730.1 ± 51.0** | **869.6 ± 27.5** | **626.8 ± 42.0** | **89,902** | 5/5 (100%) | 4.80h ± 0.79h | 246.4 |
| **`off_road_term`** (Off-Road Wrapper) | 5 | 568.9 ± 113.2 | 806.3 ± 41.9 | 506.1 ± 34.7 | 101,979 | 5/5 (100%) | **4.06h ± 0.05h** | 155.4 |

#### Per-Seed Detailed Breakdown

| Seed | Baseline Final Return | Baseline Peak Return | Baseline Time | Off-Road Final Return | Off-Road Peak Return | Off-Road Time |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **11** | 822.6 | 893.8 | 4.69h | 764.4 | 792.6 | 4.03h |
| **22** | 744.9 | 847.7 | 4.76h | 487.3 | 792.5 | 4.16h |
| **33** | 686.6 | 878.1 | 6.29h | 592.1 | 757.8 | 4.02h |
| **44** | 685.5 | 828.3 | 4.18h | 432.5 | 884.0 | 4.02h |
| **55** | 710.9 | 900.0 | 4.10h | 568.1 | 804.8 | 4.06h |

### Key Findings & Discussion:
1. **Training Efficiency & Wall-Clock Speedup:**  
   The `off_road_term` wrapper achieved a consistent training time of **4.06 hours** per 500k timesteps compared to **4.80 hours** for the baseline (**~15.4% faster**). By cutting off off-track episodes early, simulation steps are not wasted rendering and calculating physics in the grass.
2. **Episode Length Reduction:**  
   The final evaluation episode length in `off_road_term` averaged **155.4 steps** (compared to **246.4 steps** in baseline), confirming the wrapper reliably terminates episodes when off-track excursions occur.
3. **Driving Policy Performance & Stability:**  
   - Both conditions successfully solved the environment across all 5 seeds (reaching the $\ge 500$ return benchmark).
   - The baseline attained higher final returns ($730.1 \pm 51.0$ vs. $568.9 \pm 113.2$) and higher peak returns ($869.6 \pm 27.5$ vs. $806.3 \pm 41.9$).
   - *Why:* Terminating immediately upon leaving the road prevents the agent from experiencing and learning recovery trajectories (e.g. counter-steering back onto the track after a skid). In contrast, the baseline experiences negative tile penalties while on the grass but can occasionally recover, building a more robust policy around track boundaries.

---

## 6. Testing / Visual Evaluation of Checkpoints

To visually render and evaluate a trained checkpoint:

### Evaluating Baseline Agent:
```bash
python main.py \
  --mode test \
  --env CarRacing-v3 \
  --continuous \
  --actor_model results/baseline/seed_11/checkpoints/actor_best.pth \
  --eval_episodes 5 \
  --render
```

### Evaluating Off-Road Termination Agent:
```bash
python main.py \
  --mode test \
  --env CarRacing-v3 \
  --continuous \
  --actor_model results/off_road_term/seed_11/checkpoints/actor_best.pth \
  --terminate_on_off_road \
  --eval_episodes 5 \
  --render
```
