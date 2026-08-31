#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodelist=TC2N03
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --mem=6G
#SBATCH --job-name=rg_v32_prepare
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/msai/xjiang026/projects/repostguard-lite}
ENV_PREFIX=${ENV_PREFIX:-/home/msai/xjiang026/.conda/envs/repostguard}
CONFIG=configs/community_forensics_v3/student_v32_corrected.yaml
CALIBRATION_CACHE=data/cache/teacher_logits/community_forensics_m2_m3_v3_calibration
CALIBRATION_OUTPUT=outputs/community_forensics_v32/teacher_affine_calibration.json
SOURCE_CACHE=data/cache/teacher_logits/community_forensics_m3_v31_features
OUTPUT_CACHE=data/cache/teacher_logits/community_forensics_m3_v32_features_gate

cd "$PROJECT_ROOT"
module load anaconda
set +u
eval "$(conda shell.bash hook)"
conda activate "$ENV_PREFIX"
set -u
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p logs "$(dirname "$CALIBRATION_OUTPUT")" "$OUTPUT_CACHE"

python scripts/validate_family_unseen_dev_split.py \
    --train data/manifests/community_forensics_train_v31_family_holdout.csv \
    --dev data/manifests/community_forensics_val_family_unseen_dev_v1.csv \
    --protected data/manifests/community_forensics_test_external_unseen_generator_v3_expanded.csv

if [[ ! -s "$CALIBRATION_OUTPUT" ]]; then
    python scripts/calibrate_teacher_logits.py \
        --cache-directory "$CALIBRATION_CACHE" \
        --output "$CALIBRATION_OUTPUT" \
        --method per_view_affine_platt \
        --require-preprocessing-digest
fi
python scripts/validate_teacher_affine_calibration.py --config "$CONFIG"

if [[ ! -s "$OUTPUT_CACHE/teacher_cache_000-of-001.pt" ]]; then
    test -s "$SOURCE_CACHE/teacher_cache_000-of-001.pt"
    python scripts/augment_teacher_cache_quality_gate.py \
        --config "$CONFIG" \
        --source-directory "$SOURCE_CACHE" \
        --output-directory "$OUTPUT_CACHE" \
        --batch-size 128
fi
python scripts/validate_teacher_cache_shard.py \
    --config "$CONFIG" --shard-index 0 --num-shards 1
