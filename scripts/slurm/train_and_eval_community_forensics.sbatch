#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --nodelist=TC2N08
#SBATCH --time=05:50:00
#SBATCH --mem=30G
#SBATCH --job-name=rg_cf_v2_model
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --signal=B:USR1@300

set -euo pipefail

PROJECT_ROOT=/home/msai/lius0131/AGI/repostguard-lite
ENV_PREFIX=/home/msai/lius0131/.conda/envs/repostguard
EXPERIMENTS=(b0 b1 b2 m2 m3)
EXPERIMENT_INDEX=${CF_EXPERIMENT_INDEX:?CF_EXPERIMENT_INDEX is required}
PHASE=${CF_PHASE:-train}
RESTART_COUNT=${CF_RESTART_COUNT:-0}
MAX_RESTARTS=4
TRAIN_PID=
TIMEOUT_REQUESTED=0

if (( EXPERIMENT_INDEX < 0 || EXPERIMENT_INDEX >= ${#EXPERIMENTS[@]} )); then
    echo "Invalid experiment index: $EXPERIMENT_INDEX" >&2
    exit 2
fi
if [[ "$PHASE" != "train" && "$PHASE" != "eval" ]]; then
    echo "Invalid phase: $PHASE" >&2
    exit 2
fi

EXPERIMENT=${EXPERIMENTS[$EXPERIMENT_INDEX]}
CONFIG="configs/community_forensics/$EXPERIMENT.yaml"
OUTPUT_DIRECTORY="outputs/community_forensics_v2/$EXPERIMENT"

forward_usr1() {
    TIMEOUT_REQUESTED=1
    if [[ -n "$TRAIN_PID" ]] && kill -0 "$TRAIN_PID" 2>/dev/null; then
        echo "Forwarding SIGUSR1 to training PID $TRAIN_PID"
        kill -USR1 "$TRAIN_PID"
    fi
}
trap forward_usr1 USR1

submit_on_head() {
    local command=$1
    ssh -o BatchMode=yes -o ConnectTimeout=30 CCDS-TC2 \
        "cd '$PROJECT_ROOT' && $command"
}

submit_phase() {
    local index=$1
    local phase=$2
    local restart=$3
    submit_on_head "sbatch --dependency=afterany:${SLURM_JOB_ID} --export=NONE,CF_EXPERIMENT_INDEX=${index},CF_PHASE=${phase},CF_RESTART_COUNT=${restart} scripts/slurm/train_and_eval_community_forensics.sbatch"
}

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
test -s data/raw/community_forensics/VALIDATION_V2_COMPLETE
test -s data/raw/community_forensics/TRAIN_V2_COMPLETE
test -s data/manifests/community_forensics_train_v2.csv
nvidia-smi

if [[ "$PHASE" == "train" ]]; then
    python -m repostguard.train --config "$CONFIG" &
    TRAIN_PID=$!
    set +e
    wait "$TRAIN_PID"
    TRAIN_STATUS=$?
    if (( TIMEOUT_REQUESTED )) && [[ "$TRAIN_STATUS" -eq 138 ]]; then
        echo "Initial wait was interrupted by SIGUSR1; waiting for training safe-stop"
        wait "$TRAIN_PID"
        TRAIN_STATUS=$?
    fi
    set -e
    TRAIN_PID=

    if [[ "$TRAIN_STATUS" -eq 75 ]]; then
        python scripts/validate_training_checkpoint.py \
            --config "$CONFIG" \
            --checkpoint "$OUTPUT_DIRECTORY/latest.pt"
        if (( RESTART_COUNT >= MAX_RESTARTS )); then
            echo "Maximum restart count reached for $EXPERIMENT" >&2
            exit 75
        fi
        MARKER="$OUTPUT_DIRECTORY/resubmitted_${SLURM_JOB_ID}"
        if mkdir "$MARKER" 2>/dev/null; then
            submit_phase "$EXPERIMENT_INDEX" train "$((RESTART_COUNT + 1))"
        else
            echo "Training job ${SLURM_JOB_ID} was already resubmitted"
        fi
        exit 75
    fi
    if [[ "$TRAIN_STATUS" -ne 0 ]]; then
        echo "$EXPERIMENT training failed with status $TRAIN_STATUS" >&2
        exit "$TRAIN_STATUS"
    fi
    test -s "$OUTPUT_DIRECTORY/DONE"
    test -s "$OUTPUT_DIRECTORY/best.pt"
    MARKER="$OUTPUT_DIRECTORY/eval_submitted_${SLURM_JOB_ID}"
    if mkdir "$MARKER" 2>/dev/null; then
        submit_phase "$EXPERIMENT_INDEX" eval 0
    fi
    exit 0
fi

python -m repostguard.evaluate \
    --config "$CONFIG" \
    --checkpoint "$OUTPUT_DIRECTORY/best.pt"

THRESHOLD=$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["threshold_from_clean_validation"])' "$OUTPUT_DIRECTORY/summary.json")

VALIDATION_SPECS=(
    "external_exact_seen_generator|data/manifests/community_forensics_val_external_exact_seen_generator.csv"
    "hard_hourglass_exact_seen|data/manifests/community_forensics_val_hard_hourglass_v2_exact_seen.csv"
    "hard_dfgan_exact_seen|data/manifests/community_forensics_val_hard_dfgan_v2_exact_seen.csv"
    "hard_galip_exact_seen|data/manifests/community_forensics_val_hard_galip_v2_exact_seen.csv"
)
for SPEC in "${VALIDATION_SPECS[@]}"; do
    IFS='|' read -r VALIDATION_SLICE VALIDATION_MANIFEST <<< "$SPEC"
    test -s "$VALIDATION_MANIFEST"
    python -m repostguard.evaluate \
        --config "$CONFIG" \
        --checkpoint "$OUTPUT_DIRECTORY/best.pt" \
        --matrix configs/clean_only.yaml \
        --val-manifest "$VALIDATION_MANIFEST" \
        --threshold "$THRESHOLD" \
        --output-directory "$OUTPUT_DIRECTORY/val_${VALIDATION_SLICE}"
done

# Unseen-generator means both the exact identity and architecture family are unseen.
python -m repostguard.evaluate \
    --config "$CONFIG" \
    --checkpoint "$OUTPUT_DIRECTORY/best.pt" \
    --matrix configs/clean_only.yaml \
    --val-manifest data/manifests/community_forensics_test_external_unseen_generator.csv \
    --threshold "$THRESHOLD" \
    --output-directory "$OUTPUT_DIRECTORY/test_external_unseen_generator"

NEXT_INDEX=$((EXPERIMENT_INDEX + 1))
if (( NEXT_INDEX < ${#EXPERIMENTS[@]} )); then
    MARKER="$OUTPUT_DIRECTORY/next_model_submitted_${SLURM_JOB_ID}"
    if mkdir "$MARKER" 2>/dev/null; then
        submit_phase "$NEXT_INDEX" train 0
    fi
else
    echo "Community Forensics train-v2 B0/B1/B2/M2/M3 chain complete"
fi
