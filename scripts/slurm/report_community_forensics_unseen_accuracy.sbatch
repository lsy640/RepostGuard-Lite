#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --job-name=rg_cf_unseen_acc
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/home/msai/lius0131/AGI/repostguard-lite
ENV_PREFIX=/home/msai/lius0131/.conda/envs/repostguard
NODE_BIN=/home/msai/lius0131/.cursor-server/bin/linux-x64/6b2afae0257df2bb5e1835f15165dc2f0de056b0/node
NODE_BIN_FALLBACK=/home/msai/lius0131/.antigravity-server/bin/1.21.9-cc6cd32816d350ee4a1ea2b4694b43f749418957/node
DELIVERY_SCRIPT=${REPORT_BUILDER:-}
REPORT_HTML=reports/evaluations/unseen_generator/COMMUNITY_FORENSICS_UNSEEN_GENERATOR_ACCURACY.html
ARTIFACT_JSON=reports/evaluations/unseen_generator/community_forensics_unseen_generator_accuracy_artifact.json
AUDIT_JSON=reports/evaluations/unseen_generator/community_forensics_unseen_generator_accuracy_notes.json
DELIVERY_RECEIPT=reports/evaluations/unseen_generator/community_forensics_unseen_generator_accuracy_delivery_receipt.log

cd "$PROJECT_ROOT"
module load anaconda
set +u
eval "$(conda shell.bash hook)"
conda activate "$ENV_PREFIX"
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p logs reports/evaluations/unseen_generator

if [[ -z "$DELIVERY_SCRIPT" ]]; then
    shopt -s nullglob
    BUILDER_CANDIDATES=(
        /home/msai/lius0131/.codex/plugins/cache/openai-curated-remote/data-analytics/*/skills/build-report/scripts/deliver_portable_artifact.mjs
        /home/msai/lius0131/.codex/.tmp/plugins/plugins/data-analytics/skills/build-report/scripts/deliver_portable_artifact.mjs
    )
    shopt -u nullglob
    for CANDIDATE in "${BUILDER_CANDIDATES[@]}"; do
        if [[ -s "$CANDIDATE" ]]; then
            DELIVERY_SCRIPT=$CANDIDATE
            break
        fi
    done
fi
if [[ ! -x "$NODE_BIN" ]]; then
    NODE_BIN=$NODE_BIN_FALLBACK
fi

for EXPERIMENT in b0 b1 b2 m2 m3; do
    test -s "outputs/community_forensics_robustness_v2/$EXPERIMENT/unseen_generator/COMPLETE"
    test -s "outputs/community_forensics_robustness_v2/$EXPERIMENT/unseen_generator/predictions.jsonl"
done

python scripts/generate_community_forensics_unseen_accuracy_report.py \
    --bootstrap-replicates 1000 \
    --bootstrap-seed 20260829 \
    --artifact-json "$ARTIFACT_JSON" \
    --audit-json "$AUDIT_JSON"

test -s reports/evaluations/unseen_generator/community_forensics_unseen_generator_all_metrics.csv
test -s reports/evaluations/unseen_generator/community_forensics_unseen_generator_clean_metrics.csv
test -s reports/evaluations/unseen_generator/community_forensics_unseen_generator_slice_metrics.csv
test -s "$ARTIFACT_JSON"
test -s "$AUDIT_JSON"

test -x "$NODE_BIN"
test -s "$DELIVERY_SCRIPT"
echo "Portable report builder: $DELIVERY_SCRIPT"
echo "Node executable: $NODE_BIN"
"$NODE_BIN" --version
"$NODE_BIN" "$DELIVERY_SCRIPT" \
    --input "$PROJECT_ROOT/$ARTIFACT_JSON" \
    --output "$PROJECT_ROOT/$REPORT_HTML" \
    --screenshot "$PROJECT_ROOT/reports/evaluations/unseen_generator/community_forensics_unseen_generator_accuracy_delivery_failure.png" \
    | tee "$PROJECT_ROOT/$DELIVERY_RECEIPT"

cd "$PROJECT_ROOT"
test -s "$REPORT_HTML"
grep -q '"ok":true' "$DELIVERY_RECEIPT"
grep -Eq '"verification":"(passed|structural_only)"' "$DELIVERY_RECEIPT"
sha256sum \
    "$REPORT_HTML" \
    reports/evaluations/unseen_generator/community_forensics_unseen_generator_all_metrics.csv \
    reports/evaluations/unseen_generator/community_forensics_unseen_generator_clean_metrics.csv \
    reports/evaluations/unseen_generator/community_forensics_unseen_generator_slice_metrics.csv \
    "$ARTIFACT_JSON" \
    "$AUDIT_JSON"
