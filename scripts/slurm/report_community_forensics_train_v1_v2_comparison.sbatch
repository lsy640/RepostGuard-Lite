#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:45:00
#SBATCH --mem=8G
#SBATCH --job-name=rg_cf_v1v2_html
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/home/msai/lius0131/AGI/repostguard-lite
ENV_PREFIX=/home/msai/lius0131/.conda/envs/repostguard
NODE_BIN=/home/msai/lius0131/.cursor-server/bin/linux-x64/6b2afae0257df2bb5e1835f15165dc2f0de056b0/node
NODE_BIN_FALLBACK=/home/msai/lius0131/.antigravity-server/bin/1.21.9-cc6cd32816d350ee4a1ea2b4694b43f749418957/node
DELIVERY_SCRIPT=${REPORT_BUILDER:-}
REPORT_ROOT=reports/evaluations/train_v1_v2_comparison
REPORT_HTML=reports/summaries/COMMUNITY_FORENSICS_TRAIN_V1_V2_COMPARISON.html
ARTIFACT_JSON="$REPORT_ROOT/community_forensics_train_v1_v2_artifact.json"
AUDIT_JSON="$REPORT_ROOT/community_forensics_train_v1_v2_notes.json"
DELIVERY_RECEIPT="$REPORT_ROOT/community_forensics_train_v1_v2_delivery_receipt.log"

cd "$PROJECT_ROOT"
module load anaconda
set +u
eval "$(conda shell.bash hook)"
conda activate "$ENV_PREFIX"
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p logs "$REPORT_ROOT"

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

test -s reports/summaries/COMMUNITY_FORENSICS_TRAIN_V1_V2_COMPARISON.md
test -s data/manifests/community_forensics_train.csv
test -s data/manifests/community_forensics_train_v2.csv
for EXPERIMENT in b0 b1 b2 m2 m3; do
    test -s "outputs/community_forensics_robustness_v2/$EXPERIMENT/COMPLETE"
    test -s "outputs/community_forensics_v2_robustness_v2/$EXPERIMENT/COMPLETE"
    test -s "outputs/community_forensics_v2_robustness_v2/$EXPERIMENT/unseen_generator/predictions.jsonl"
done

python scripts/generate_community_forensics_train_v1_v2_comparison_report.py \
    --bootstrap-replicates 1000 \
    --bootstrap-seed 20260830 \
    --artifact-json "$ARTIFACT_JSON" \
    --audit-json "$AUDIT_JSON"

test -s "$ARTIFACT_JSON"
test -s "$AUDIT_JSON"
test -s "$REPORT_ROOT/community_forensics_train_v1_v2_split_comparison.csv"
test -s "$REPORT_ROOT/community_forensics_strict_unseen_all_metrics.csv"
test -s "$REPORT_ROOT/community_forensics_strict_unseen_clean_metrics.csv"
test -s "$REPORT_ROOT/community_forensics_strict_unseen_generator_metrics.csv"
test -s "$REPORT_ROOT/community_forensics_strict_unseen_real_source_metrics.csv"
test -s "$REPORT_ROOT/community_forensics_m3_unseen_generator_roc_metrics.csv"
jq -e '.manifest.charts | length >= 11' "$ARTIFACT_JSON" >/dev/null
jq -e '.manifest.tables | length >= 16' "$ARTIFACT_JSON" >/dev/null
jq -e '.snapshot.datasets.condition_metrics | length == 105' "$ARTIFACT_JSON" >/dev/null
jq -e '.snapshot.datasets.clean_metrics | length == 5' "$ARTIFACT_JSON" >/dev/null
jq -e '.snapshot.datasets.m3_generator_roc_detail | length == 36' "$ARTIFACT_JSON" >/dev/null
jq -e '.snapshot.datasets.m3_generator_roc_wide | length == 12' "$ARTIFACT_JSON" >/dev/null

test -x "$NODE_BIN"
test -s "$DELIVERY_SCRIPT"
echo "Portable report builder: $DELIVERY_SCRIPT"
echo "Node executable: $NODE_BIN"
"$NODE_BIN" --version
"$NODE_BIN" "$DELIVERY_SCRIPT" \
    --input "$PROJECT_ROOT/$ARTIFACT_JSON" \
    --output "$PROJECT_ROOT/$REPORT_HTML" \
    --screenshot "$PROJECT_ROOT/$REPORT_ROOT/community_forensics_train_v1_v2_delivery_failure.png" \
    | tee "$PROJECT_ROOT/$DELIVERY_RECEIPT"

cd "$PROJECT_ROOT"
test -s "$REPORT_HTML"
grep -q 'Community Forensics train-v1 / train-v2' "$REPORT_HTML"
grep -q 'Strict unseen-generator' "$REPORT_HTML"
grep -q 'roc_chart' "$REPORT_HTML"
grep -q 'strict_multistage_chart' "$REPORT_HTML"
grep -q 'm3_generator_roc_chart' "$REPORT_HTML"
grep -q 'M3 标识' "$REPORT_HTML"
grep -q '"ok":true' "$DELIVERY_RECEIPT"
grep -Eq '"verification":"(passed|structural_only)"' "$DELIVERY_RECEIPT"
sha256sum \
    "$REPORT_HTML" \
    "$ARTIFACT_JSON" \
    "$AUDIT_JSON" \
    "$REPORT_ROOT/community_forensics_train_v1_v2_split_comparison.csv" \
    "$REPORT_ROOT/community_forensics_strict_unseen_all_metrics.csv" \
    "$REPORT_ROOT/community_forensics_strict_unseen_clean_metrics.csv" \
    "$REPORT_ROOT/community_forensics_m3_unseen_generator_roc_metrics.csv"
