#!/usr/bin/env bash
# SLURM array-job template for fan-out generation/validation (placeholder).
#SBATCH --job-name=aapg-array
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-15
#SBATCH --output=experiments/outputs/%x-%A_%a.log

set -euo pipefail

# $SLURM_ARRAY_TASK_ID selects the shard/seed for this task.
# uv run python -m aapg.pipelines.generate seed=${SLURM_ARRAY_TASK_ID} "$@"
echo "[placeholder] array task ${SLURM_ARRAY_TASK_ID:-?}"
