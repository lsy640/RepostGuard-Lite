#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=05:50:00
#SBATCH --mem=16G
#SBATCH --job-name=rg_cf_ext_v3_data
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --signal=B:USR1@300

set -euo pipefail

PROJECT_ROOT=/home/msai/lius0131/AGI/repostguard-lite
ENV_PREFIX=/home/msai/lius0131/.conda/envs/repostguard
RESTART_COUNT=${CF_V3_EXT_RESTART_COUNT:-0}
MAX_RESTARTS=8
BUILD_PID=
TIMEOUT_REQUESTED=0

forward_usr1() {
    TIMEOUT_REQUESTED=1
    if [[ -n "$BUILD_PID" ]] && kill -0 "$BUILD_PID" 2>/dev/null; then
        kill -USR1 "$BUILD_PID"
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
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_ETAG_TIMEOUT=30
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_XET_HIGH_PERFORMANCE=1
mkdir -p logs data/manifests data/state data/raw/community_forensics

test -s data/raw/community_forensics/TRAIN_V3_COMPLETE
test -s data/manifests/community_forensics_train_v3.csv
test -s data/manifests/community_forensics_test_external_unseen_generator.csv

python -m pytest -q tests/test_config.py
python scripts/build_community_forensics_external_unseen_v3.py &
BUILD_PID=$!
set +e
wait "$BUILD_PID"
BUILD_STATUS=$?
if (( TIMEOUT_REQUESTED )) && [[ "$BUILD_STATUS" -eq 138 ]]; then
    wait "$BUILD_PID"
    BUILD_STATUS=$?
fi
set -e
BUILD_PID=

if [[ "$BUILD_STATUS" -eq 75 ]]; then
    python scripts/build_community_forensics_external_unseen_v3.py --verify-state
    if (( RESTART_COUNT >= MAX_RESTARTS )); then
        echo "Maximum expanded-test restart count reached" >&2
        exit 75
    fi
    MARKER="data/state/community_forensics_external_unseen_v3_resubmitted_${SLURM_JOB_ID}"
    if mkdir "$MARKER" 2>/dev/null; then
        submit_on_head "sbatch --dependency=afterany:${SLURM_JOB_ID} --export=NONE,CF_V3_EXT_RESTART_COUNT=$((RESTART_COUNT + 1)) scripts/slurm/prepare_community_forensics_external_unseen_v3.sbatch"
    fi
    exit 75
fi
if [[ "$BUILD_STATUS" -ne 0 ]]; then
    exit "$BUILD_STATUS"
fi

test -s data/raw/community_forensics/EXTERNAL_UNSEEN_V3_COMPLETE
test -s data/manifests/community_forensics_test_external_unseen_generator_v3_expanded.csv
test -s data/manifests/community_forensics_external_unseen_v3_additions.csv
test -s data/manifests/community_forensics_external_unseen_v3_audit.json
jq -e '.output.rows == 4000' data/manifests/community_forensics_external_unseen_v3_audit.json >/dev/null
jq -e '.output.class_counts == {"aigi":2000,"real":2000}' data/manifests/community_forensics_external_unseen_v3_audit.json >/dev/null
jq -e '.additions.exact_generator_count == 12' data/manifests/community_forensics_external_unseen_v3_audit.json >/dev/null
jq -e '.additions.real_source_counts == {"COCO":250,"FFHQ":250,"LAION":250,"RAISE":250}' data/manifests/community_forensics_external_unseen_v3_audit.json >/dev/null
jq -e '.integrity.phash_near_duplicate_count == 0' data/manifests/community_forensics_external_unseen_v3_audit.json >/dev/null

sha256sum \
    data/manifests/community_forensics_test_external_unseen_generator_v3_expanded.csv \
    data/manifests/community_forensics_external_unseen_v3_additions.csv \
    data/manifests/community_forensics_external_unseen_v3_audit.json
