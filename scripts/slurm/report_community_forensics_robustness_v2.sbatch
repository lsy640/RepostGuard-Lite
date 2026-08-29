#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --job-name=rg_cf_rv2_report
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/home/msai/lius0131/AGI/repostguard-lite
ENV_PREFIX=/home/msai/lius0131/.conda/envs/repostguard
PLUGIN_ROOT=/home/msai/lius0131/.codex/plugins/cache/openai-curated-remote/data-analytics/0.2.8-13ceeea1f599
NODE_BIN=/home/msai/lius0131/.cursor-server/bin/linux-x64/6b2afae0257df2bb5e1835f15165dc2f0de056b0/node
DELIVERY_SCRIPT="$PLUGIN_ROOT/skills/build-report/scripts/deliver_portable_artifact.mjs"
OUTPUT_ROOT=outputs/community_forensics_robustness_v2
REPORT_HTML=reports/COMMUNITY_FORENSICS_B0_B1_B2_M2_M3_ROBUSTNESS_V2.html
ARTIFACT_JSON=reports/community_forensics_robustness_v2_report_artifact.json
AUDIT_JSON=reports/community_forensics_robustness_v2_report_notes.json
DELIVERY_RECEIPT=reports/community_forensics_robustness_v2_delivery_receipt.log

cd "$PROJECT_ROOT"
module load anaconda
set +u
eval "$(conda shell.bash hook)"
conda activate "$ENV_PREFIX"
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p logs reports

for EXPERIMENT in b0 b1 b2 m2 m3; do
    test -s "$OUTPUT_ROOT/$EXPERIMENT/COMPLETE"
done

python scripts/generate_community_forensics_robustness_v2_report.py \
    --evaluation-root "$OUTPUT_ROOT" \
    --matrix configs/community_forensics_robustness_v2.yaml \
    --metrics-csv reports/community_forensics_robustness_v2_metrics.csv \
    --artifact-json "$ARTIFACT_JSON" \
    --audit-json "$AUDIT_JSON"

test -s reports/community_forensics_robustness_v2_metrics.csv
test -s "$ARTIFACT_JSON"
test -s "$AUDIT_JSON"

test -x "$NODE_BIN"
test -s "$DELIVERY_SCRIPT"
"$NODE_BIN" --version
"$NODE_BIN" "$DELIVERY_SCRIPT" \
    --input "$PROJECT_ROOT/$ARTIFACT_JSON" \
    --output "$PROJECT_ROOT/$REPORT_HTML" \
    --screenshot "$PROJECT_ROOT/reports/community_forensics_robustness_v2_delivery_failure.png" \
    | tee "$PROJECT_ROOT/$DELIVERY_RECEIPT"

cd "$PROJECT_ROOT"
test -s "$REPORT_HTML"
grep -q '"ok":true' "$DELIVERY_RECEIPT"
grep -Eq '"verification":"(passed|structural_only)"' "$DELIVERY_RECEIPT"
sha256sum "$REPORT_HTML" \
    reports/community_forensics_robustness_v2_metrics.csv \
    "$ARTIFACT_JSON" \
    "$AUDIT_JSON"
