#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodelist=TC2N03
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=05:50:00
#SBATCH --mem=30G
#SBATCH --job-name=rg_v32_external4k
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/msai/xjiang026/projects/repostguard-lite}
ENV_PREFIX=${ENV_PREFIX:-/home/msai/xjiang026/.conda/envs/repostguard}
CONFIG=configs/community_forensics_v3/student_v32_corrected.yaml
OUTPUT_DIRECTORY=outputs/community_forensics_v32/student_corrected
CHECKPOINT="$OUTPUT_DIRECTORY/best.pt"
SELECTION="$OUTPUT_DIRECTORY/dev_selection_vs_v31_baseline.json"
DEV_DIRECTORY="$OUTPUT_DIRECTORY/dev_family_unseen_robustness"
FINAL_DIRECTORY="$OUTPUT_DIRECTORY/final_external_v3_expanded_robustness_v2"

cd "$PROJECT_ROOT"
module load anaconda
set +u
eval "$(conda shell.bash hook)"
conda activate "$ENV_PREFIX"
set -u
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export REPOSTGUARD_NUM_WORKERS="$SLURM_CPUS_PER_TASK"

test -s "$CONFIG"
test -s "$CHECKPOINT"
test -s "$SELECTION"
test -s "$DEV_DIRECTORY/summary.json"
python scripts/validate_training_checkpoint.py \
    --config "$CONFIG" --checkpoint "$CHECKPOINT"

python - "$SELECTION" "$CONFIG" "$CHECKPOINT" "$OUTPUT_DIRECTORY" <<'PY'
import json
import sys

selection_path, config, checkpoint, output_directory = sys.argv[1:]
selection = json.load(open(selection_path, encoding="utf-8"))
if selection.get("selection_data") != "family_unseen_dev_only":
    raise SystemExit("Selection did not use the family-unseen dev protocol")
if selection.get("protected_external_test_used") is not False:
    raise SystemExit("Protected external data was already used during selection")
winner = selection.get("winner", {})
expected = {
    "name": "v32_corrected",
    "config": config,
    "checkpoint": checkpoint,
    "output_directory": output_directory,
}
for key, value in expected.items():
    if winner.get(key) != value:
        raise SystemExit(
            f"V3.2 final gate failed for {key}: "
            f"actual={winner.get(key)!r}, expected={value!r}"
        )
print("V3.2 internal winner gate passed")
PY

THRESHOLD=$(python -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["threshold_from_clean_validation"])' \
    "$DEV_DIRECTORY/summary.json")

if [[ -s "$FINAL_DIRECTORY/summary.json" \
      && -s "$FINAL_DIRECTORY/run_card.json" \
      && -s "$FINAL_DIRECTORY/metrics_by_transform.csv" \
      && -s "$FINAL_DIRECTORY/predictions.jsonl" ]]; then
    echo "Validated existing V3.2 protected external output; skipping duplicate evaluation."
else
    python -m repostguard.evaluate \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --matrix configs/community_forensics_robustness_v2.yaml \
        --val-manifest data/manifests/community_forensics_test_external_unseen_generator_v3_expanded.csv \
        --threshold "$THRESHOLD" \
        --output-directory "$FINAL_DIRECTORY"
fi

test -s "$FINAL_DIRECTORY/summary.json"
test -s "$FINAL_DIRECTORY/run_card.json"
test -s "$FINAL_DIRECTORY/metrics_by_transform.csv"
test -s "$FINAL_DIRECTORY/predictions.jsonl"

echo "Protected external strict 4k evaluation complete; mobile export intentionally skipped."
