#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodelist=TC2N03
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --mem=30G
#SBATCH --job-name=rg_v31_dev_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-/home/msai/xjiang026/projects/repostguard-lite}
ENV_PREFIX=${ENV_PREFIX:-/home/msai/xjiang026/.conda/envs/repostguard}
CONFIG=${STUDENT_CONFIG:?Set STUDENT_CONFIG to one v3.1 ablation config}

cd "$PROJECT_ROOT"
module load anaconda
set +u
eval "$(conda shell.bash hook)"
conda activate "$ENV_PREFIX"
set -u
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
OUTPUT_DIRECTORY=$(python -c 'from repostguard.config import load_config; import sys; print(load_config(sys.argv[1])["output"]["directory"])' "$CONFIG")
test -s "$OUTPUT_DIRECTORY/DONE"
test -s "$OUTPUT_DIRECTORY/best.pt"
python -m repostguard.evaluate \
    --config "$CONFIG" \
    --checkpoint "$OUTPUT_DIRECTORY/best.pt" \
    --output-directory "$OUTPUT_DIRECTORY/dev_family_unseen_robustness"
