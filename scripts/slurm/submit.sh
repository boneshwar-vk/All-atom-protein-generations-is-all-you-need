#!/usr/bin/env bash
# SLURM submission template (placeholder). Adjust partition/resources per cluster.
#SBATCH --job-name=aapg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=experiments/outputs/%x-%j.log

set -euo pipefail

# Example: run the end-to-end pipeline on the cluster.
# uv run python -m aapg.pipelines.end2end +experiment=end2end launcher=slurm "$@"
echo "[placeholder] configure the command above, then submit with: sbatch scripts/slurm/submit.sh"
