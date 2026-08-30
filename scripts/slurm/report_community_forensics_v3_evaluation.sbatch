#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --job-name=rg_cf_v3_report
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/home/msai/lius0131/AGI/repostguard-lite
ENV_PREFIX=/home/msai/lius0131/.conda/envs/repostguard
NODE_BIN=/home/msai/lius0131/.cursor-server/bin/linux-x64/6b2afae0257df2bb5e1835f15165dc2f0de056b0/node
NODE_BIN_FALLBACK=/home/msai/lius0131/.antigravity-server/bin/1.21.9-cc6cd32816d350ee4a1ea2b4694b43f749418957/node
DELIVERY_SCRIPT=${REPORT_BUILDER:-}
REPORT_ROOT=reports/evaluations/community_forensics_v3_evaluation
REPORT_MD=reports/summaries/COMMUNITY_FORENSICS_V3_EVALUATION_REPORT.md
REPORT_HTML=reports/summaries/COMMUNITY_FORENSICS_V3_EVALUATION_REPORT.html
METRICS_CSV="$REPORT_ROOT/community_forensics_v3_evaluation_metrics.csv"
ARTIFACT_JSON="$REPORT_ROOT/community_forensics_v3_evaluation_artifact.json"
AUDIT_JSON="$REPORT_ROOT/community_forensics_v3_evaluation_audit.json"
DELIVERY_RECEIPT="$REPORT_ROOT/community_forensics_v3_evaluation_delivery_receipt.log"

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

test -s data/manifests/community_forensics_train_v3.csv
test -s data/manifests/community_forensics_test_external_unseen_generator_v3_expanded.csv
test -s configs/community_forensics_robustness_v2.yaml
for EXPERIMENT in b0 b1 b2 m2 m3; do
    test -s "outputs/community_forensics_v3/$EXPERIMENT/DONE"
    test -s "outputs/community_forensics_v3_robustness_v2/$EXPERIMENT/exact_seen_generator/COMPLETE"
    test -s "outputs/community_forensics_v3_robustness_v2/$EXPERIMENT/hard_hourglass_exact_seen/COMPLETE"
    test -s "outputs/community_forensics_v3_robustness_v2/$EXPERIMENT/hard_dfgan_exact_seen/COMPLETE"
    test -s "outputs/community_forensics_v3_robustness_v2/$EXPERIMENT/hard_galip_exact_seen/COMPLETE"
    test -s "outputs/community_forensics_v3_robustness_v2/$EXPERIMENT/unseen_generator_expanded/COMPLETE"
done

python -m py_compile \
    scripts/generate_community_forensics_robustness_v2_report.py \
    scripts/generate_community_forensics_v3_evaluation_report.py
python scripts/generate_community_forensics_v3_evaluation_report.py

test -s "$REPORT_MD"
test -s "$METRICS_CSV"
test -s "$ARTIFACT_JSON"
test -s "$AUDIT_JSON"
jq -e '.snapshot.status == "ready"' "$ARTIFACT_JSON" >/dev/null
jq -e '.snapshot.datasets.headline_metrics | length == 1' "$ARTIFACT_JSON" >/dev/null
jq -e '.snapshot.datasets.macro_detail | length == 5' "$ARTIFACT_JSON" >/dev/null
jq -e '.snapshot.datasets.split_definitions | length == 5' "$ARTIFACT_JSON" >/dev/null
jq -e '.snapshot.datasets.split_summary | length == 25' "$ARTIFACT_JSON" >/dev/null
jq -e '.snapshot.datasets.all_metrics | length == 525' "$ARTIFACT_JSON" >/dev/null
jq -e '.snapshot.datasets.model_lineage | length == 5' "$ARTIFACT_JSON" >/dev/null
jq -e '.snapshot.datasets.perturbation_matrix | length == 21' "$ARTIFACT_JSON" >/dev/null
jq -e '.snapshot.datasets.m3_gates | length == 15' "$ARTIFACT_JSON" >/dev/null
jq -e '.snapshot.datasets.full_unseen_headline | length == 1' "$ARTIFACT_JSON" >/dev/null
jq -e '.snapshot.datasets.full_unseen_clean_chart | length == 5' "$ARTIFACT_JSON" >/dev/null
jq -e '.snapshot.datasets.full_unseen_multistage_chart | length == 15' "$ARTIFACT_JSON" >/dev/null
jq -e '.dataset_integrity.slices.unseen_generator_expanded.rows == 4000' "$AUDIT_JSON" >/dev/null
jq -e '.dataset_integrity.slices.unseen_generator_expanded.exact_generator_overlap_with_train_v3 | length == 0' "$AUDIT_JSON" >/dev/null
jq -e '.dataset_integrity.slices.unseen_generator_expanded.architecture_overlap_with_train_v3 | length == 0' "$AUDIT_JSON" >/dev/null
jq -e '[.dataset_integrity.slices[] | .sample_id_overlap_with_train_v3] | add == 0' "$AUDIT_JSON" >/dev/null
jq -e '[.dataset_integrity.slices[] | .sha256_overlap_with_train_v3] | add == 0' "$AUDIT_JSON" >/dev/null
jq -e '.manifest.charts | length == 5' "$ARTIFACT_JSON" >/dev/null
jq -e '.manifest.tables | length >= 7' "$ARTIFACT_JSON" >/dev/null

test -x "$NODE_BIN"
test -s "$DELIVERY_SCRIPT"
echo "Portable report builder: $DELIVERY_SCRIPT"
echo "Node executable: $NODE_BIN"
"$NODE_BIN" --version
"$NODE_BIN" "$DELIVERY_SCRIPT" \
    --input "$PROJECT_ROOT/$ARTIFACT_JSON" \
    --output "$PROJECT_ROOT/$REPORT_HTML" \
    --screenshot "$PROJECT_ROOT/$REPORT_ROOT/community_forensics_v3_evaluation_delivery_failure.png" \
    | tee "$PROJECT_ROOT/$DELIVERY_RECEIPT"

cd "$PROJECT_ROOT"
test -s "$REPORT_HTML"
grep -q 'Community Forensics train-v3' "$REPORT_HTML"
grep -q 'Full unseen-generator (4,000)' "$REPORT_HTML"
grep -q 'Hard Hourglass' "$REPORT_HTML"
grep -q 'Hard DFGAN' "$REPORT_HTML"
grep -q 'Hard GALIP' "$REPORT_HTML"
grep -q 'macro_auroc_chart' "$REPORT_HTML"
grep -q 'strict_six_chart' "$REPORT_HTML"
grep -q 'worst_new_chart' "$REPORT_HTML"
grep -q 'full_unseen_clean_auroc_chart' "$REPORT_HTML"
grep -q 'full_unseen_multistage_auroc_chart' "$REPORT_HTML"
if grep -qi 'train-v2 / train-v3\|v2.*v3.*comparison' "$REPORT_HTML"; then
    echo "Unexpected v2 comparison language in standalone v3 report" >&2
    exit 1
fi
grep -q '"ok":true' "$DELIVERY_RECEIPT"
grep -Eq '"verification":"(passed|structural_only)"' "$DELIVERY_RECEIPT"
sha256sum "$REPORT_MD" "$REPORT_HTML" "$METRICS_CSV" "$ARTIFACT_JSON" "$AUDIT_JSON"
