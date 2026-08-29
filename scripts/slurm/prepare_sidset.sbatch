#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=05:30:00
#SBATCH --mem=12G
#SBATCH --job-name=rg_sid_data
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --signal=B:USR1@300

set -euo pipefail

PROJECT_ROOT=/home/msai/lius0131/AGI/repostguard-lite
ENV_PREFIX=/home/msai/lius0131/.conda/envs/repostguard
WORK_PID=

forward_usr1() {
    if [[ -n "$WORK_PID" ]] && kill -0 "$WORK_PID" 2>/dev/null; then
        echo "Forwarding SIGUSR1 to SID-Set preparation PID $WORK_PID"
        kill -USR1 "$WORK_PID"
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
export HF_HOME="$PROJECT_ROOT/data/cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p logs "$HF_DATASETS_CACHE" data/manifests data/raw/sid_set

df -h "$PROJECT_ROOT"
python -m pip install --disable-pip-version-check "datasets==5.0.1"
python -m pip install --no-deps --editable .
python scripts/download_sidset_subset.py \
    --data-root data/raw/sid_set \
    --manifest-dir data/manifests \
    --cache-dir "$HF_DATASETS_CACHE" \
    --train-per-class 10000 \
    --validation-per-class 2000 \
    --seed 20260827 &
WORK_PID=$!
set +e
wait "$WORK_PID"
WORK_STATUS=$?
set -e
WORK_PID=

if [[ "$WORK_STATUS" -eq 75 ]]; then
    echo "SID-Set progress was checkpointed; resubmit this script to resume"
    exit 75
fi
if [[ "$WORK_STATUS" -ne 0 ]]; then
    echo "SID-Set preparation failed with status $WORK_STATUS" >&2
    exit "$WORK_STATUS"
fi

python -m pytest
du -sh data/raw/sid_set data/cache/huggingface 2>/dev/null || true
df -h "$PROJECT_ROOT"

