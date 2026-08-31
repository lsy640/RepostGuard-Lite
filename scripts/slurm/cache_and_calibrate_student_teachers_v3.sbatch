#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodelist=TC2N03
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=05:50:00
#SBATCH --mem=30G
#SBATCH --job-name=rg_v3_teacher_cal
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/msai/xjiang026/projects/repostguard-lite}
ENV_PREFIX=${ENV_PREFIX:-/home/msai/xjiang026/.conda/envs/repostguard}
TEMPLATE_CONFIG=${STUDENT_TEMPLATE_CONFIG:-configs/community_forensics_v3/student_mnv3_dual_teacher.yaml}
FROZEN_CONFIG=${STUDENT_FROZEN_CONFIG:-configs/community_forensics_v3/student_mnv3_dual_teacher_frozen.yaml}
CALIBRATION_MANIFEST=${STUDENT_CALIBRATION_MANIFEST:-data/manifests/community_forensics_val_external_exact_seen_generator.csv}
CALIBRATION_CACHE=${STUDENT_CALIBRATION_CACHE:-data/cache/teacher_logits/community_forensics_m2_m3_v3_calibration}
CALIBRATION_JSON=${STUDENT_CALIBRATION_JSON:-outputs/community_forensics_v3/student_mnv3_dual_teacher/teacher_calibration.json}
M2_SHA=468d3a58603fdf8dfe1b234a24fd8e52a99c6e4881e921bef6bb0cea64bbac34
M3_SHA=c83f70641a9c8d7f6808e794cfc8c28c0e478feeca7506e489c772a512115b2f

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
test -s data/raw/community_forensics/TRAIN_V3_COMPLETE
test -s data/raw/community_forensics/EXTERNAL_UNSEEN_V3_COMPLETE
test -s outputs/community_forensics_v3/m2/best.pt
test -s outputs/community_forensics_v3/m3/best.pt
test -s "$TEMPLATE_CONFIG"
if [[ -e outputs/community_forensics_v3/student_mnv3_dual_teacher/latest.pt ]] || \
   [[ -e outputs/community_forensics_v3/student_mnv3_dual_teacher/DONE ]]; then
    echo "Student training has already started; refusing to overwrite the frozen config."
    exit 3
fi
nvidia-smi

python scripts/validate_distillation_calibration_manifest.py \
    --calibration "$CALIBRATION_MANIFEST" \
    --against data/manifests/community_forensics_train_v3.csv \
    --against data/manifests/community_forensics_val_unseen_generator.csv \
    --against data/manifests/community_forensics_test_external_unseen_generator.csv \
    --against data/manifests/community_forensics_test_external_unseen_generator_v3_expanded.csv \
    --expected-samples 2000 \
    --expected-per-label 1000

if [[ ! -s "$CALIBRATION_CACHE/teacher_cache_000-of-001.pt" ]]; then
    python scripts/cache_teacher_logits.py \
        --config "$TEMPLATE_CONFIG" \
        --manifest "$CALIBRATION_MANIFEST" \
        --output-directory "$CALIBRATION_CACHE" \
        --shard-index 0 \
        --num-shards 1 \
        --batch-size "${TEACHER_CACHE_BATCH_SIZE:-8}"
fi
python scripts/validate_teacher_cache_shard.py \
    --config "$TEMPLATE_CONFIG" \
    --manifest "$CALIBRATION_MANIFEST" \
    --output-directory "$CALIBRATION_CACHE" \
    --shard-index 0 \
    --num-shards 1 \
    | tee "$(dirname "$CALIBRATION_JSON")/teacher_calibration_cache_acceptance.json"

python scripts/calibrate_teacher_logits.py \
    --cache-directory "$CALIBRATION_CACHE" \
    --output "$CALIBRATION_JSON" \
    --m2-weight-grid 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1 \
    --minimum-dual-auroc-gain 0.002 \
    --maximum-clean-auroc-regression 0.001 \
    --require-preprocessing-digest
python scripts/check_teacher_calibration.py \
    --calibration "$CALIBRATION_JSON" \
    --expected-samples 2000 \
    --expected-views 4 \
    --m2-checkpoint-sha256 "$M2_SHA" \
    --m3-checkpoint-sha256 "$M3_SHA" \
    | tee "$(dirname "$CALIBRATION_JSON")/teacher_calibration_acceptance.json"
python scripts/freeze_student_distillation_config.py \
    --config "$TEMPLATE_CONFIG" \
    --calibration "$CALIBRATION_JSON" \
    --output "$FROZEN_CONFIG" \
    --overwrite

test -s "$CALIBRATION_JSON"
test -s "$FROZEN_CONFIG"
sha256sum "$CALIBRATION_JSON" "$FROZEN_CONFIG"
