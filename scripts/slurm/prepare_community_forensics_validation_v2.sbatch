#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=05:50:00
#SBATCH --mem=30G
#SBATCH --job-name=rg_cf_val2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --signal=B:USR1@300

set -euo pipefail

PROJECT_ROOT=/home/msai/lius0131/AGI/repostguard-lite
ENV_PREFIX=/home/msai/lius0131/.conda/envs/repostguard
RESTART_COUNT=${CF_VAL2_RESTART_COUNT:-0}
MAX_RESTARTS=8
WORK_PID=
TIMEOUT_REQUESTED=0

forward_usr1() {
    TIMEOUT_REQUESTED=1
    if [[ -n "$WORK_PID" ]] && kill -0 "$WORK_PID" 2>/dev/null; then
        echo "Forwarding SIGUSR1 to validation-v2 builder PID $WORK_PID"
        kill -USR1 "$WORK_PID"
    fi
}
trap forward_usr1 USR1

submit_on_head() {
    local command=$1
    ssh -o BatchMode=yes -o ConnectTimeout=30 CCDS-TC2 \
        "cd '$PROJECT_ROOT' && $command"
}

cd "$PROJECT_ROOT"
module load anaconda
set +u
eval "$(conda shell.bash hook)"
conda activate "$ENV_PREFIX"
set -u

export PYTHONUNBUFFERED=1
export HF_HOME="$PROJECT_ROOT/data/cache/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p logs data/manifests data/state data/raw/community_forensics

df -h "$PROJECT_ROOT"
python -m pytest -q tests/test_community_forensics.py tests/test_config.py

python scripts/build_community_forensics_validation_v2.py \
    --data-root data/raw/community_forensics \
    --manifest-dir data/manifests \
    --state-database data/state/community_forensics_validation_v2.sqlite3 \
    --base-state-database data/state/community_forensics.sqlite3 \
    --phash-threshold 4 \
    --max-new-gib 8 &
WORK_PID=$!
set +e
wait "$WORK_PID"
WORK_STATUS=$?
if (( TIMEOUT_REQUESTED )) && [[ "$WORK_STATUS" -eq 138 ]]; then
    echo "Initial wait was interrupted by SIGUSR1; waiting for builder safe-stop"
    wait "$WORK_PID"
    WORK_STATUS=$?
fi
set -e
WORK_PID=

if [[ "$WORK_STATUS" -eq 75 ]]; then
    python scripts/build_community_forensics_validation_v2.py \
        --data-root data/raw/community_forensics \
        --state-database data/state/community_forensics_validation_v2.sqlite3 \
        --verify-state
    if (( RESTART_COUNT >= MAX_RESTARTS )); then
        echo "Maximum validation-v2 restart count reached" >&2
        exit 75
    fi
    MARKER="data/state/cf_val2_resubmitted_${SLURM_JOB_ID}"
    if mkdir "$MARKER" 2>/dev/null; then
        submit_on_head "sbatch --dependency=afterany:${SLURM_JOB_ID} --export=NONE,CF_VAL2_RESTART_COUNT=$((RESTART_COUNT + 1)) scripts/slurm/prepare_community_forensics_validation_v2.sbatch"
    else
        echo "Validation-v2 job ${SLURM_JOB_ID} was already resubmitted"
    fi
    exit 75
fi
if [[ "$WORK_STATUS" -ne 0 ]]; then
    echo "Validation-v2 preparation failed with status $WORK_STATUS" >&2
    exit "$WORK_STATUS"
fi

test -s data/raw/community_forensics/VALIDATION_V2_COMPLETE
python scripts/build_community_forensics_validation_v2.py \
    --data-root data/raw/community_forensics \
    --state-database data/state/community_forensics_validation_v2.sqlite3 \
    --verify-state

for manifest in \
    data/manifests/community_forensics_val_external_exact_seen_generator.csv \
    data/manifests/community_forensics_val_hard_hourglass.csv \
    data/manifests/community_forensics_val_hard_dfgan.csv \
    data/manifests/community_forensics_val_hard_galip.csv; do
    test -s "$manifest"
done

du -sh data/raw/community_forensics/validation_v2
df -h "$PROJECT_ROOT"
