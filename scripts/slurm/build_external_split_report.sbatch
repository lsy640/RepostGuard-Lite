#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --job-name=rg_cf_report
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/home/msai/lius0131/AGI/repostguard-lite
NODE_BIN=/home/msai/lius0131/.antigravity-server/bin/1.21.9-cc6cd32816d350ee4a1ea2b4694b43f749418957/node

# Prefer an explicit override, then discover the currently installed plugin
# version. Codex plugin cache versions are content-addressed and may change.
REPORT_BUILDER_OVERRIDE=${REPORT_BUILDER:-}
REPORT_BUILDER=
for candidate in \
    "$REPORT_BUILDER_OVERRIDE" \
    /home/msai/lius0131/.codex/plugins/cache/openai-curated-remote/data-analytics/*/skills/build-report/scripts/deliver_portable_artifact.mjs \
    /home/msai/lius0131/.codex/.tmp/plugins/plugins/data-analytics/skills/build-report/scripts/deliver_portable_artifact.mjs; do
    if [[ -n "$candidate" && -f "$candidate" ]]; then
        REPORT_BUILDER=$candidate
        break
    fi
done

if [[ -z "$REPORT_BUILDER" ]]; then
    echo "ERROR: deliver_portable_artifact.mjs was not found." >&2
    echo "Set REPORT_BUILDER to the installed data-analytics build-report script." >&2
    exit 2
fi

if [[ ! -x "$NODE_BIN" ]]; then
    echo "ERROR: Node executable not found or not executable: $NODE_BIN" >&2
    exit 2
fi

cd "$PROJECT_ROOT"
mkdir -p logs reports/evaluations/external_split

"$NODE_BIN" --version
echo "Report builder: $REPORT_BUILDER"
"$NODE_BIN" "$REPORT_BUILDER" \
    --input reports/evaluations/external_split/community_forensics_external_split_diagnostic_artifact.json \
    --output reports/evaluations/external_split/community_forensics_external_split_diagnostic.html
