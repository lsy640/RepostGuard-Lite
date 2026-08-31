#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodelist=TC2N03
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=05:50:00
#SBATCH --mem=30G
#SBATCH --job-name=rg_teacher_cal
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/msai/xjiang026/projects/repostguard-lite}
ENV_PREFIX=${ENV_PREFIX:-/home/msai/xjiang026/.conda/envs/repostguard}
TEMPLATE_CONFIG=${STUDENT_TEMPLATE_CONFIG:-configs/community_forensics/student_mnv3_dual_teacher.yaml}
FROZEN_CONFIG=${STUDENT_FROZEN_CONFIG:-configs/community_forensics/student_mnv3_dual_teacher_frozen.yaml}
CALIBRATION_MANIFEST=${STUDENT_CALIBRATION_MANIFEST:-data/manifests/community_forensics_val_unseen_generator.csv}
CALIBRATION_CACHE=${STUDENT_CALIBRATION_CACHE:-data/cache/teacher_logits/community_forensics_m2_m3_calibration_v1}
CALIBRATION_JSON=${STUDENT_CALIBRATION_JSON:-outputs/community_forensics/student_mnv3_dual_teacher/teacher_calibration.json}

cd "$PROJECT_ROOT"
module load anaconda
set +u
eval "$(conda shell.bash hook)"
conda activate "$ENV_PREFIX"
set -u

export PYTHONUNBUFFERED=1
export TORCH_HOME="$PROJECT_ROOT/data/cache/torch"
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p logs "$CALIBRATION_CACHE" "$(dirname "$CALIBRATION_JSON")"
test -s data/raw/community_forensics/COMPLETE
test -s outputs/community_forensics/distillation_teacher_m2/best.pt
test -s outputs/community_forensics/distillation_teacher_m3/best.pt
nvidia-smi

if [[ ! -s "$CALIBRATION_CACHE/teacher_cache_000-of-001.pt" ]]; then
    python scripts/cache_teacher_logits.py \
        --config "$TEMPLATE_CONFIG" \
        --manifest "$CALIBRATION_MANIFEST" \
        --output-directory "$CALIBRATION_CACHE" \
        --shard-index 0 \
        --num-shards 1 \
        --batch-size "${TEACHER_CACHE_BATCH_SIZE:-8}"
fi

python scripts/calibrate_teacher_logits.py \
    --cache-directory "$CALIBRATION_CACHE" \
    --output "$CALIBRATION_JSON"
python scripts/freeze_student_distillation_config.py \
    --config "$TEMPLATE_CONFIG" \
    --calibration "$CALIBRATION_JSON" \
    --output "$FROZEN_CONFIG" \
    --overwrite

test -s "$CALIBRATION_JSON"
test -s "$FROZEN_CONFIG"
sha256sum "$CALIBRATION_JSON" "$FROZEN_CONFIG"
