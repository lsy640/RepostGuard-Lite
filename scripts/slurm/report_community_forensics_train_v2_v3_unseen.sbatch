#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --nodelist=TC2N08
#SBATCH --time=01:30:00
#SBATCH --mem=16G
#SBATCH --job-name=rg_cf_v2v3_intersection_report
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/home/msai/lius0131/AGI/repostguard-lite
ENV_PREFIX=/home/msai/lius0131/.conda/envs/repostguard
NODE_BIN=/home/msai/lius0131/.cursor-server/bin/linux-x64/6b2afae0257df2bb5e1835f15165dc2f0de056b0/node
NODE_BIN_FALLBACK=/home/msai/lius0131/.antigravity-server/bin/1.21.9-cc6cd32816d350ee4a1ea2b4694b43f749418957/node
DELIVERY_SCRIPT=${REPORT_BUILDER:-}
REPORT_ROOT=reports/evaluations/train_v2_v3_unseen_intersection_comparison
REPORT_MD=reports/summaries/COMMUNITY_FORENSICS_TRAIN_V2_V3_UNSEEN_INTERSECTION_COMPARISON.md
REPORT_HTML=reports/summaries/COMMUNITY_FORENSICS_TRAIN_V2_V3_UNSEEN_INTERSECTION_COMPARISON.html
ARTIFACT_JSON="$REPORT_ROOT/community_forensics_train_v2_v3_unseen_intersection_artifact.json"
AUDIT_JSON="$REPORT_ROOT/community_forensics_train_v2_v3_unseen_intersection_audit.json"
DELIVERY_RECEIPT="$REPORT_ROOT/community_forensics_train_v2_v3_unseen_intersection_delivery_receipt.log"

cd "$PROJECT_ROOT"
module load anaconda
set +u
eval "$(conda shell.bash hook)"
conda activate "$ENV_PREFIX"
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p logs "$REPORT_ROOT" reports/summaries

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

test -s data/manifests/community_forensics_test_external_unseen_generator.csv
test -s data/manifests/community_forensics_test_external_unseen_generator_v3_expanded.csv
test -s data/manifests/community_forensics_train_v2.csv
test -s data/manifests/community_forensics_train_v3.csv
test -s configs/community_forensics_robustness_v2.yaml
for EXPERIMENT in b0 b1 b2 m2 m3; do
    test -s "outputs/community_forensics_v2_robustness_v2/$EXPERIMENT/unseen_generator/COMPLETE"
    test -s "outputs/community_forensics_v3_robustness_v2/$EXPERIMENT/unseen_generator_expanded/COMPLETE"
done

python -m py_compile scripts/generate_community_forensics_train_v2_v3_unseen_report.py
python scripts/generate_community_forensics_train_v2_v3_unseen_report.py \
    --bootstrap-replicates 1000 \
    --bootstrap-seed 20260830

test -s "$REPORT_MD"
test -s "$ARTIFACT_JSON"
test -s "$AUDIT_JSON"
test -s "$REPORT_ROOT/community_forensics_train_v2_v3_unseen_intersection_clean.csv"
test -s "$REPORT_ROOT/community_forensics_train_v2_v3_unseen_intersection_all_conditions.csv"
test -s "$REPORT_ROOT/community_forensics_train_v2_v3_unseen_intersection_multistage.csv"
test -s "$REPORT_ROOT/community_forensics_train_v2_v3_unseen_intersection_generator.csv"
test -s "$REPORT_ROOT/community_forensics_train_v2_v3_unseen_intersection_real_source.csv"
test -s "$REPORT_ROOT/community_forensics_train_v2_v3_unseen_intersection_lineage.csv"
jq -e '.snapshot.datasets.clean_comparison | length == 5' "$ARTIFACT_JSON" >/dev/null
jq -e '.snapshot.datasets.condition_comparison | length == 105' "$ARTIFACT_JSON" >/dev/null
jq -e '.snapshot.datasets.multistage_comparison | length == 15' "$ARTIFACT_JSON" >/dev/null
jq -e '.snapshot.datasets.generator_comparison | length == 60' "$ARTIFACT_JSON" >/dev/null
jq -e '.snapshot.datasets.real_source_comparison | length == 20' "$ARTIFACT_JSON" >/dev/null
jq -e '.snapshot.datasets.lineage | length == 10' "$ARTIFACT_JSON" >/dev/null
jq -e '.manifest.charts | length >= 6' "$ARTIFACT_JSON" >/dev/null
jq -e '.manifest.tables | length >= 10' "$ARTIFACT_JSON" >/dev/null

test -x "$NODE_BIN"
test -s "$DELIVERY_SCRIPT"
echo "Portable report builder: $DELIVERY_SCRIPT"
echo "Node executable: $NODE_BIN"
"$NODE_BIN" --version
"$NODE_BIN" "$DELIVERY_SCRIPT" \
    --input "$PROJECT_ROOT/$ARTIFACT_JSON" \
    --output "$PROJECT_ROOT/$REPORT_HTML" \
    --screenshot "$PROJECT_ROOT/$REPORT_ROOT/community_forensics_train_v2_v3_unseen_intersection_delivery_failure.png" \
    | tee "$PROJECT_ROOT/$DELIVERY_RECEIPT"

cd "$PROJECT_ROOT"
test -s "$REPORT_HTML"
grep -q 'train-v2 / train-v3' "$REPORT_HTML"
grep -q 'Strict Unseen-generator 交集' "$REPORT_HTML"
grep -q 'condition_delta_chart' "$REPORT_HTML"
grep -q 'generator_delta_chart' "$REPORT_HTML"
grep -q '"ok":true' "$DELIVERY_RECEIPT"
grep -Eq '"verification":"(passed|structural_only)"' "$DELIVERY_RECEIPT"
sha256sum \
    "$REPORT_MD" \
    "$REPORT_HTML" \
    "$ARTIFACT_JSON" \
    "$AUDIT_JSON" \
    "$REPORT_ROOT/community_forensics_train_v2_v3_unseen_intersection_clean.csv" \
    "$REPORT_ROOT/community_forensics_train_v2_v3_unseen_intersection_all_conditions.csv" \
    "$REPORT_ROOT/community_forensics_train_v2_v3_unseen_intersection_multistage.csv" \
    "$REPORT_ROOT/community_forensics_train_v2_v3_unseen_intersection_generator.csv" \
    "$REPORT_ROOT/community_forensics_train_v2_v3_unseen_intersection_real_source.csv"
