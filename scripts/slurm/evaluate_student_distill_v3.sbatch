#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodelist=TC2N03
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=05:50:00
#SBATCH --mem=30G
#SBATCH --job-name=rg_v3_student_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/msai/xjiang026/projects/repostguard-lite}
ENV_PREFIX=${ENV_PREFIX:-/home/msai/xjiang026/.conda/envs/repostguard}
CONFIG=${STUDENT_CONFIG:-configs/community_forensics_v3/student_mnv3_dual_teacher_frozen.yaml}
OUTPUT_DIRECTORY=outputs/community_forensics_v3/student_mnv3_dual_teacher
CHECKPOINT="$OUTPUT_DIRECTORY/best.pt"

cd "$PROJECT_ROOT"
module load anaconda
set +u
eval "$(conda shell.bash hook)"
conda activate "$ENV_PREFIX"
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p logs "$OUTPUT_DIRECTORY/mobile"
test -s data/raw/community_forensics/TRAIN_V3_COMPLETE
test -s data/raw/community_forensics/EXTERNAL_UNSEEN_V3_COMPLETE
test -s "$CONFIG"
test -s "$CHECKPOINT"
nvidia-smi

python -m repostguard.evaluate \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output-directory "$OUTPUT_DIRECTORY/internal_validation"
THRESHOLD=$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["threshold_from_clean_validation"])' "$OUTPUT_DIRECTORY/internal_validation/summary.json")

python -m repostguard.evaluate \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --matrix configs/clean_only.yaml \
    --val-manifest data/manifests/community_forensics_val_external_exact_seen_generator.csv \
    --threshold "$THRESHOLD" \
    --output-directory "$OUTPUT_DIRECTORY/calibration_exact_seen_tuned"

for SPEC in \
    "hard_hourglass:data/manifests/community_forensics_val_hard_hourglass_v2_exact_seen.csv" \
    "hard_dfgan:data/manifests/community_forensics_val_hard_dfgan_v2_exact_seen.csv" \
    "hard_galip:data/manifests/community_forensics_val_hard_galip_v2_exact_seen.csv"; do
    NAME=${SPEC%%:*}
    MANIFEST=${SPEC#*:}
    python -m repostguard.evaluate \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --matrix configs/clean_only.yaml \
        --val-manifest "$MANIFEST" \
        --threshold "$THRESHOLD" \
        --output-directory "$OUTPUT_DIRECTORY/diagnostic_${NAME}"
done

python -m repostguard.evaluate \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --matrix configs/clean_only.yaml \
    --val-manifest data/manifests/community_forensics_test_external_unseen_generator.csv \
    --threshold "$THRESHOLD" \
    --output-directory "$OUTPUT_DIRECTORY/test_external_unseen_generator_v1_historical"

python -m repostguard.evaluate \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --matrix configs/community_forensics_robustness_v2.yaml \
    --val-manifest data/manifests/community_forensics_test_external_unseen_generator_v3_expanded.csv \
    --threshold "$THRESHOLD" \
    --output-directory "$OUTPUT_DIRECTORY/test_external_unseen_generator_v3_expanded_robustness_v2"

python scripts/export_student_mobile.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output-directory "$OUTPUT_DIRECTORY/mobile" \
    --format both
python scripts/validate_mobile_onnx.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --onnx "$OUTPUT_DIRECTORY/mobile/student_mnv3_fp32.onnx" \
    | tee "$OUTPUT_DIRECTORY/mobile/onnx_parity.json"
test -s "$OUTPUT_DIRECTORY/mobile/student_mnv3_fp32.onnx"
test -s "$OUTPUT_DIRECTORY/mobile/export_metadata.json"
test -s "$OUTPUT_DIRECTORY/mobile/onnx_parity.json"
