#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodelist=TC2N03
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=05:50:00
#SBATCH --mem=30G
#SBATCH --job-name=rg_v32_full_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --signal=B:USR1@300
#SBATCH --requeue

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/msai/xjiang026/projects/repostguard-lite}
ENV_PREFIX=${ENV_PREFIX:-/home/msai/xjiang026/.conda/envs/repostguard}
CONFIG=configs/community_forensics_v3/student_v32_full_refit_e20.yaml
OUTPUT_DIRECTORY=outputs/community_forensics_v32_full_refit_e20/student_corrected
STOP_AFTER_EPOCH=${STOP_AFTER_EPOCH:-}
TRAIN_PID=
TIMEOUT_REQUESTED=0

forward_usr1() {
    TIMEOUT_REQUESTED=1
    if [[ -n "$TRAIN_PID" ]] && kill -0 "$TRAIN_PID" 2>/dev/null; then
        kill -USR1 "$TRAIN_PID"
    fi
}
trap forward_usr1 USR1

cd "$PROJECT_ROOT"
module load anaconda
set +u
eval "$(conda shell.bash hook)"
conda activate "$ENV_PREFIX"
set -u
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export REPOSTGUARD_NUM_WORKERS="$SLURM_CPUS_PER_TASK"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p logs

python scripts/validate_student_v32_full_refit_data.py \
    --train data/manifests/community_forensics_train_v3.csv \
    --val data/manifests/community_forensics_val_v32_full_refit_e20.csv \
    --protected data/manifests/community_forensics_test_external_unseen_generator_v3_expanded.csv \
    --expected-train-rows 24000 \
    --expected-train-sha256 fc0a7ab732faeb604ed1e77281fada715d7cffb353974a4985820548d871d9d6
python scripts/validate_teacher_affine_calibration.py --config "$CONFIG"
test -s data/cache/teacher_logits/community_forensics_m3_v32_full_refit_e20/teacher_cache_000-of-001.pt

if [[ -s "$OUTPUT_DIRECTORY/DONE" ]]; then
    python scripts/validate_training_checkpoint.py \
        --config "$CONFIG" --checkpoint "$OUTPUT_DIRECTORY/best.pt"
    python scripts/validate_training_checkpoint.py \
        --config "$CONFIG" --checkpoint "$OUTPUT_DIRECTORY/latest.pt"
    echo "Validated completed full refit; refusing duplicate training."
    exit 0
fi

TRAIN_ARGS=(python -m repostguard.train --config "$CONFIG")
if [[ -n "$STOP_AFTER_EPOCH" ]]; then
    TRAIN_ARGS+=(--stop-after-epoch "$STOP_AFTER_EPOCH")
fi
"${TRAIN_ARGS[@]}" &
TRAIN_PID=$!
set +e
wait "$TRAIN_PID"
TRAIN_STATUS=$?
set -e
if (( TIMEOUT_REQUESTED )) && [[ "$TRAIN_STATUS" -eq 138 ]]; then
    wait "$TRAIN_PID"
    TRAIN_STATUS=$?
fi
if [[ "$TRAIN_STATUS" -eq 75 ]]; then
    python scripts/validate_training_checkpoint.py \
        --config "$CONFIG" --checkpoint "$OUTPUT_DIRECTORY/latest.pt"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi
exit "$TRAIN_STATUS"
