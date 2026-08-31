#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodelist=TC2N03
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=05:50:00
#SBATCH --mem=30G
#SBATCH --job-name=rg_v31_final_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-/home/msai/xjiang026/projects/repostguard-lite}
ENV_PREFIX=${ENV_PREFIX:-/home/msai/xjiang026/.conda/envs/repostguard}
SELECTION=outputs/community_forensics_v31/ablation_selection.json

cd "$PROJECT_ROOT"
module load anaconda
set +u
eval "$(conda shell.bash hook)"
conda activate "$ENV_PREFIX"
set -u
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
readarray -t WINNER < <(python - <<'PY'
import json
selection=json.load(open('outputs/community_forensics_v31/ablation_selection.json',encoding='utf-8'))
assert selection['protected_external_test_used'] is False
winner=selection['winner']
print(winner['config'])
print(winner['checkpoint'])
print(winner['output_directory'])
PY
)
CONFIG=${WINNER[0]}
CHECKPOINT=${WINNER[1]}
OUTPUT_DIRECTORY=${WINNER[2]}
DEV_SUMMARY="$OUTPUT_DIRECTORY/dev_family_unseen_robustness/summary.json"
THRESHOLD=$(python -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["threshold_from_clean_validation"])' "$DEV_SUMMARY")
python -m repostguard.evaluate \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --matrix configs/community_forensics_robustness_v2.yaml \
    --val-manifest data/manifests/community_forensics_test_external_unseen_generator_v3_expanded.csv \
    --threshold "$THRESHOLD" \
    --output-directory "$OUTPUT_DIRECTORY/final_external_v3_expanded_robustness_v2"
mkdir -p "$OUTPUT_DIRECTORY/mobile"
python scripts/export_student_mobile.py \
    --config "$CONFIG" --checkpoint "$CHECKPOINT" \
    --output-directory "$OUTPUT_DIRECTORY/mobile" --format both
python scripts/validate_mobile_onnx.py \
    --config "$CONFIG" --checkpoint "$CHECKPOINT" \
    --onnx "$OUTPUT_DIRECTORY/mobile/student_mnv3_fp32.onnx" \
    | tee "$OUTPUT_DIRECTORY/mobile/onnx_parity.json"
