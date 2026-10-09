#!/bin/bash
# sbatch --nodes=32 scripts/slurm_train.sbatch configs/train/pretrain-7b.yaml
#SBATCH --job-name=yohogpt
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --exclusive
#SBATCH --time=14-00:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export NNODES=$SLURM_NNODES RUN_ID=$SLURM_JOB_ID
# Checkpoints auto-resume from out_dir/latest, so a --requeue after node failure just continues.
srun scripts/launch_multinode.sh "$@"
