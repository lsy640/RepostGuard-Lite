#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --nodelist=TC2N08
#SBATCH --time=05:50:00
#SBATCH --mem=30G
#SBATCH --job-name=rg_sid_train
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --signal=B:USR1@300

set -euo pipefail

PROJECT_ROOT=/home/msai/lius0131/AGI/repostguard-lite
ENV_PREFIX=/home/msai/lius0131/.conda/envs/repostguard
EXPERIMENTS=(b0 b1 b2 m2 m3)
ARRAY_INDEX=${SLURM_ARRAY_TASK_ID:?This script must be submitted as an experiment array}
if (( ARRAY_INDEX < 0 || ARRAY_INDEX >= ${#EXPERIMENTS[@]} )); then
    echo "Invalid array index: $ARRAY_INDEX" >&2
    exit 2
fi
EXPERIMENT=${EXPERIMENTS[$ARRAY_INDEX]}
CONFIG="configs/sidset/$EXPERIMENT.yaml"
OUTPUT_DIRECTORY="outputs/sidset/$EXPERIMENT"
TRAIN_PID=

forward_usr1() {
    if [[ -n "$TRAIN_PID" ]] && kill -0 "$TRAIN_PID" 2>/dev/null; then
        echo "Forwarding SIGUSR1 to training PID $TRAIN_PID"
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
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p logs "$OUTPUT_DIRECTORY"

nvidia-smi
TRAIN_ARGS=(--config "$CONFIG")
if [[ -n "${REPOSTGUARD_RESUME_CHECKPOINT:-}" ]]; then
    TRAIN_ARGS+=(--resume "$REPOSTGUARD_RESUME_CHECKPOINT")
    echo "Using explicit resume checkpoint: $REPOSTGUARD_RESUME_CHECKPOINT"
fi
python -m repostguard.train "${TRAIN_ARGS[@]}" &
TRAIN_PID=$!
set +e
wait "$TRAIN_PID"
TRAIN_STATUS=$?
set -e
TRAIN_PID=
if [[ "$TRAIN_STATUS" -eq 75 ]]; then
    echo "Training checkpointed after SIGUSR1; rerun array index $ARRAY_INDEX to resume"
    exit 75
fi
if [[ "$TRAIN_STATUS" -ne 0 ]]; then
    echo "Training failed with status $TRAIN_STATUS" >&2
    exit "$TRAIN_STATUS"
fi

python -m repostguard.evaluate \
    --config "$CONFIG" \
    --checkpoint "$OUTPUT_DIRECTORY/best.pt"
