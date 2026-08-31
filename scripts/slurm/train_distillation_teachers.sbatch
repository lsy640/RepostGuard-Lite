#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodelist=TC2N03
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=05:50:00
#SBATCH --mem=30G
#SBATCH --array=0-1%2
#SBATCH --job-name=rg_teacher_train
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --signal=B:USR1@300
#SBATCH --requeue

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/msai/xjiang026/projects/repostguard-lite}
ENV_PREFIX=${ENV_PREFIX:-/home/msai/xjiang026/.conda/envs/repostguard}
EXPERIMENTS=(m2 m3)
EXPERIMENT=${TEACHER_EXPERIMENT:-${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}}
CONFIG="configs/community_forensics/distillation_teacher_${EXPERIMENT}.yaml"
OUTPUT_DIRECTORY="outputs/community_forensics/distillation_teacher_${EXPERIMENT}"
TRAIN_PID=
TIMEOUT_REQUESTED=0
AUTO_REQUEUE=${AUTO_REQUEUE:-1}

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
export TORCH_HOME="$PROJECT_ROOT/data/cache/torch"
export HF_HOME="$PROJECT_ROOT/data/cache/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p logs "$OUTPUT_DIRECTORY"
test -s data/raw/community_forensics/COMPLETE
test -s data/manifests/community_forensics_train.csv
test -s data/manifests/community_forensics_val_unseen_generator.csv
nvidia-smi

python -m repostguard.train --config "$CONFIG" &
TRAIN_PID=$!
set +e
wait "$TRAIN_PID"
TRAIN_STATUS=$?
if (( TIMEOUT_REQUESTED )) && [[ "$TRAIN_STATUS" -eq 138 ]]; then
    wait "$TRAIN_PID"
    TRAIN_STATUS=$?
fi
set -e

if [[ "$TRAIN_STATUS" -eq 75 ]]; then
    python scripts/validate_training_checkpoint.py \
        --config "$CONFIG" \
        --checkpoint "$OUTPUT_DIRECTORY/latest.pt"
    if [[ "$AUTO_REQUEUE" == "1" ]] && scontrol requeue "$SLURM_JOB_ID"; then
        echo "Safe-stop complete for $EXPERIMENT; requeued $SLURM_JOB_ID."
        exit 0
    fi
    echo "Safe-stop complete for $EXPERIMENT; manual resubmission is required."
    exit 75
fi
if [[ "$TRAIN_STATUS" -ne 0 ]]; then
    exit "$TRAIN_STATUS"
fi

test -s "$OUTPUT_DIRECTORY/best.pt"
sha256sum "$OUTPUT_DIRECTORY/best.pt"
