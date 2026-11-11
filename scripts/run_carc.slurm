#!/bin/bash
#SBATCH --job-name=deeprl_p3
#SBATCH --output=logs/carc_%A_%a.out
#SBATCH --error=logs/carc_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-4

# Module load or environment activation on UNM CARC
# module load python/3.11-anaconda or activate virtualenv
# source /path/to/venv/bin/activate

# 5 target seeds matching Project 2
SEEDS=(11 22 33 44 55)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# Default condition (override when launching: sbatch --export=CONDITION=gae_baseline scripts/run_carc.slurm)
CONDITION=${CONDITION:-"baseline"}
EXTRA_ARGS=${EXTRA_ARGS:-""}

echo "=== UNM CARC PPO Job ==="
echo "Job ID: $SLURM_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"
echo "Host: $(hostname)"
echo "Condition: $CONDITION, Seed: $SEED"
echo "GPU info:"
nvidia-smi

mkdir -p logs

python main.py \
  --mode train \
  --env CarRacing-v3 \
  --continuous \
  --condition "$CONDITION" \
  --seed "$SEED" \
  --total_timesteps 500000 \
  $EXTRA_ARGS

echo "Finished run for Condition: $CONDITION, Seed: $SEED"

