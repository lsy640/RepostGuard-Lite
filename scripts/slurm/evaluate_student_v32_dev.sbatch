#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodelist=TC2N03
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --mem=30G
#SBATCH --job-name=rg_v32_dev_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-/home/msai/xjiang026/projects/repostguard-lite}
ENV_PREFIX=${ENV_PREFIX:-/home/msai/xjiang026/.conda/envs/repostguard}
CONFIG=configs/community_forensics_v3/student_v32_corrected.yaml
OUTPUT_DIRECTORY=outputs/community_forensics_v32/student_corrected
BASELINE_CONFIG=configs/community_forensics_v3/student_v31_ablation_baseline_t3.yaml
BASELINE_DIRECTORY=outputs/community_forensics_v31/student_ablation_baseline_t3

cd "$PROJECT_ROOT"
module load anaconda
set +u
eval "$(conda shell.bash hook)"
conda activate "$ENV_PREFIX"
set -u
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export REPOSTGUARD_NUM_WORKERS="$SLURM_CPUS_PER_TASK"

test -s "$OUTPUT_DIRECTORY/best.pt"
python -m repostguard.evaluate \
    --config "$CONFIG" \
    --checkpoint "$OUTPUT_DIRECTORY/best.pt" \
    --output-directory "$OUTPUT_DIRECTORY/dev_family_unseen_robustness"
python scripts/select_student_v31_ablation.py \
    --candidate "v31_baseline_t3:$BASELINE_CONFIG:$BASELINE_DIRECTORY" \
    --candidate "v32_corrected:$CONFIG:$OUTPUT_DIRECTORY" \
    --output "$OUTPUT_DIRECTORY/dev_selection_vs_v31_baseline.json"
