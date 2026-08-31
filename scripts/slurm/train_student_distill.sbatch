#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodelist=TC2N03
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=05:50:00
#SBATCH --mem=30G
#SBATCH --job-name=rg_student_kd
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --signal=B:USR1@300
#SBATCH --requeue

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/msai/xjiang026/projects/repostguard-lite}
ENV_PREFIX=${ENV_PREFIX:-/home/msai/xjiang026/.conda/envs/repostguard}
CONFIG=${STUDENT_CONFIG:-configs/community_forensics/student_mnv3_dual_teacher_frozen.yaml}
OUTPUT_DIRECTORY=outputs/community_forensics/student_mnv3_dual_teacher
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
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p logs "$OUTPUT_DIRECTORY"
test -s data/raw/community_forensics/COMPLETE
test -s "$CONFIG"
test -s data/cache/teacher_logits/community_forensics_m2_m3_v1/teacher_cache_000-of-002.pt
test -s data/cache/teacher_logits/community_forensics_m2_m3_v1/teacher_cache_001-of-002.pt
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
        echo "Safe-stop complete; requeued $SLURM_JOB_ID."
        exit 0
    fi
    echo "Safe-stop complete; manual resubmission is required."
    exit 75
fi
exit "$TRAIN_STATUS"
