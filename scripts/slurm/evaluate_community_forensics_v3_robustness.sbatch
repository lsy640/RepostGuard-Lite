#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=TC2N08
#SBATCH --time=05:50:00
#SBATCH --mem=24G
#SBATCH --job-name=rg_cf_v3_robust
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/home/msai/lius0131/AGI/repostguard-lite
ENV_PREFIX=/home/msai/lius0131/.conda/envs/repostguard
MATRIX=configs/community_forensics_robustness_v2.yaml
OUTPUT_ROOT=outputs/community_forensics_v3_robustness_v2
EXPERIMENTS=(b0 b1 b2 m2 m3)
MODEL_INDICES=${CF_V3_MODEL_INDICES:?CF_V3_MODEL_INDICES is required, for example 0:1:2}
SPLIT_SPECS=(
    "exact_seen_generator|data/manifests/community_forensics_val_external_exact_seen_generator.csv"
    "hard_hourglass_exact_seen|data/manifests/community_forensics_val_hard_hourglass_v2_exact_seen.csv"
    "hard_dfgan_exact_seen|data/manifests/community_forensics_val_hard_dfgan_v2_exact_seen.csv"
    "hard_galip_exact_seen|data/manifests/community_forensics_val_hard_galip_v2_exact_seen.csv"
    "unseen_generator_expanded|data/manifests/community_forensics_test_external_unseen_generator_v3_expanded.csv"
)

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
mkdir -p logs "$OUTPUT_ROOT"
test -s data/raw/community_forensics/TRAIN_V3_COMPLETE
test -s data/raw/community_forensics/EXTERNAL_UNSEEN_V3_COMPLETE
test -s "$MATRIX"
nvidia-smi

IFS=':' read -r -a REQUESTED_INDICES <<< "$MODEL_INDICES"
for MODEL_INDEX in "${REQUESTED_INDICES[@]}"; do
    if [[ ! "$MODEL_INDEX" =~ ^[0-4]$ ]]; then
        echo "Invalid model index: $MODEL_INDEX" >&2
        exit 2
    fi
    EXPERIMENT=${EXPERIMENTS[$MODEL_INDEX]}
    CHECKPOINT="outputs/community_forensics_v3/$EXPERIMENT/best.pt"
    CONFIG="outputs/community_forensics_v3/$EXPERIMENT/resolved_config.yaml"
    SOURCE_SUMMARY="outputs/community_forensics_v3/$EXPERIMENT/summary.json"
    MODEL_OUTPUT="$OUTPUT_ROOT/$EXPERIMENT"
    test -s "$CHECKPOINT"
    test -s "$CONFIG"
    test -s "$SOURCE_SUMMARY"
    mkdir -p "$MODEL_OUTPUT"
    THRESHOLD=$(jq -er '.threshold_from_clean_validation' "$SOURCE_SUMMARY")

    for SPEC in "${SPLIT_SPECS[@]}"; do
        IFS='|' read -r SPLIT MANIFEST <<< "$SPEC"
        SPLIT_OUTPUT="$MODEL_OUTPUT/$SPLIT"
        if [[ -s "$SPLIT_OUTPUT/COMPLETE" ]]; then
            echo "Skipping completed split: $EXPERIMENT/$SPLIT"
            continue
        fi
        mkdir -p "$SPLIT_OUTPUT"
        python -m repostguard.evaluate \
            --config "$CONFIG" \
            --checkpoint "$CHECKPOINT" \
            --matrix "$MATRIX" \
            --val-manifest "$MANIFEST" \
            --threshold "$THRESHOLD" \
            --output-directory "$SPLIT_OUTPUT"
        jq -e '.conditions == 21' "$SPLIT_OUTPUT/summary.json" >/dev/null
        printf '{"completed_at_utc":"%s","experiment":"%s","slurm_job_id":"%s","split":"%s"}\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$EXPERIMENT" "$SLURM_JOB_ID" "$SPLIT" \
            > "$SPLIT_OUTPUT/COMPLETE.tmp.${SLURM_JOB_ID}"
        mv "$SPLIT_OUTPUT/COMPLETE.tmp.${SLURM_JOB_ID}" "$SPLIT_OUTPUT/COMPLETE"
    done
done
