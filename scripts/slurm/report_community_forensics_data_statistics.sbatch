#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --job-name=rg_cf_data_report
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/home/msai/lius0131/AGI/repostguard-lite
ENV_PREFIX=/home/msai/lius0131/.conda/envs/repostguard
PLUGIN_ROOT=/home/msai/lius0131/.codex/plugins/cache/openai-curated-remote/data-analytics/0.2.8-13ceeea1f599
NODE_BIN=/home/msai/lius0131/.cursor-server/bin/linux-x64/6b2afae0257df2bb5e1835f15165dc2f0de056b0/node
DELIVERY_SCRIPT="$PLUGIN_ROOT/skills/build-report/scripts/deliver_portable_artifact.mjs"
REPORT_HTML=reports/COMMUNITY_FORENSICS_DATA_STATISTICS.html
ARTIFACT_JSON=reports/community_forensics_data_statistics_artifact.json
AUDIT_JSON=reports/community_forensics_data_statistics_notes.json
SPLIT_CSV=reports/community_forensics_data_statistics.csv
GENERATOR_CSV=reports/community_forensics_generator_statistics.csv
DISTRIBUTION_CSV=reports/community_forensics_distribution_statistics.csv
DELIVERY_RECEIPT=reports/community_forensics_data_statistics_delivery_receipt.log

cd "$PROJECT_ROOT"
module load anaconda
set +u
eval "$(conda shell.bash hook)"
conda activate "$ENV_PREFIX"
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p logs reports

test -s data/raw/community_forensics/COMPLETE
test -s data/raw/community_forensics/VALIDATION_V2_COMPLETE
test -s reports/community_forensics_tiff_integrity.json

python scripts/generate_community_forensics_data_statistics_report.py \
    --data-root data/raw/community_forensics \
    --tiff-audit reports/community_forensics_tiff_integrity.json \
    --split-csv "$SPLIT_CSV" \
    --generator-csv "$GENERATOR_CSV" \
    --distribution-csv "$DISTRIBUTION_CSV" \
    --artifact-json "$ARTIFACT_JSON" \
    --audit-json "$AUDIT_JSON"

test -s "$SPLIT_CSV"
test -s "$GENERATOR_CSV"
test -s "$DISTRIBUTION_CSV"
test -s "$ARTIFACT_JSON"
test -s "$AUDIT_JSON"

test -x "$NODE_BIN"
test -s "$DELIVERY_SCRIPT"
"$NODE_BIN" --version
"$NODE_BIN" "$DELIVERY_SCRIPT" \
    --input "$PROJECT_ROOT/$ARTIFACT_JSON" \
    --output "$PROJECT_ROOT/$REPORT_HTML" \
    --screenshot "$PROJECT_ROOT/reports/community_forensics_data_statistics_delivery_failure.png" \
    | tee "$PROJECT_ROOT/$DELIVERY_RECEIPT"

cd "$PROJECT_ROOT"
test -s "$REPORT_HTML"
grep -q '"ok":true' "$DELIVERY_RECEIPT"
grep -Eq '"verification":"(passed|structural_only)"' "$DELIVERY_RECEIPT"
sha256sum \
    "$REPORT_HTML" \
    "$SPLIT_CSV" \
    "$GENERATOR_CSV" \
    "$DISTRIBUTION_CSV" \
    "$ARTIFACT_JSON" \
    "$AUDIT_JSON"
