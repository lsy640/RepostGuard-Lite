#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodelist=TC2N03
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=05:50:00
#SBATCH --mem=30G
#SBATCH --job-name=rg_student_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/msai/xjiang026/projects/repostguard-lite}
ENV_PREFIX=${ENV_PREFIX:-/home/msai/xjiang026/.conda/envs/repostguard}
CONFIG=${STUDENT_CONFIG:-configs/community_forensics/student_mnv3_dual_teacher_frozen.yaml}
OUTPUT_DIRECTORY=outputs/community_forensics/student_mnv3_dual_teacher
CHECKPOINT="$OUTPUT_DIRECTORY/best.pt"

cd "$PROJECT_ROOT"
module load anaconda
set +u
eval "$(conda shell.bash hook)"
conda activate "$ENV_PREFIX"
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p logs
test -s "$CHECKPOINT"
nvidia-smi

python -m repostguard.evaluate --config "$CONFIG" --checkpoint "$CHECKPOINT"
THRESHOLD=$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["threshold_from_clean_validation"])' "$OUTPUT_DIRECTORY/summary.json")

for SLICE in external_exact_seen_generator hard_hourglass hard_dfgan hard_galip; do
    python -m repostguard.evaluate \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --matrix configs/clean_only.yaml \
        --val-manifest "data/manifests/community_forensics_val_${SLICE}.csv" \
        --threshold "$THRESHOLD" \
        --output-directory "$OUTPUT_DIRECTORY/val_${SLICE}"
done

python -m repostguard.evaluate \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --matrix configs/clean_only.yaml \
    --val-manifest data/manifests/community_forensics_test_external_seen_family.csv \
    --threshold "$THRESHOLD" \
    --output-directory "$OUTPUT_DIRECTORY/test_external_seen_family"

python -m repostguard.evaluate \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --matrix configs/community_forensics_robustness_v2.yaml \
    --val-manifest data/manifests/community_forensics_test_external_unseen_generator.csv \
    --threshold "$THRESHOLD" \
    --output-directory "$OUTPUT_DIRECTORY/test_external_unseen_generator_robustness_v2"

python scripts/export_student_mobile.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output-directory "$OUTPUT_DIRECTORY/mobile" \
    --format torchscript
test -s "$OUTPUT_DIRECTORY/mobile/export_metadata.json"
