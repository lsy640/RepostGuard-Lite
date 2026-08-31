#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodelist=TC2N03
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=05:50:00
#SBATCH --mem=30G
#SBATCH --array=0-1%1
#SBATCH --job-name=rg_v3_teacher_gate
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/msai/xjiang026/projects/repostguard-lite}
ENV_PREFIX=${ENV_PREFIX:-/home/msai/xjiang026/.conda/envs/repostguard}
EXPERIMENTS=(m2 m3)
EXPERIMENT=${TEACHER_EXPERIMENT:-${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}}
CONFIG="configs/community_forensics_v3/${EXPERIMENT}.yaml"
OUTPUT_DIRECTORY="outputs/community_forensics_v3/${EXPERIMENT}"
CHECKPOINT="$OUTPUT_DIRECTORY/best.pt"
INTERNAL="$OUTPUT_DIRECTORY/acceptance_internal"
UNSEEN="$OUTPUT_DIRECTORY/acceptance_unseen_v3_expanded"

cd "$PROJECT_ROOT"
module load anaconda
set +u
eval "$(conda shell.bash hook)"
conda activate "$ENV_PREFIX"
set -u

export PYTHONUNBUFFERED=1
export TORCH_HOME="$PROJECT_ROOT/data/cache/torch"
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p logs "$OUTPUT_DIRECTORY"
test -s data/raw/community_forensics/TRAIN_V3_COMPLETE
test -s data/raw/community_forensics/EXTERNAL_UNSEEN_V3_COMPLETE
test -s data/manifests/community_forensics_test_external_unseen_generator_v3_expanded.csv
test -s "$CONFIG"
test -s "$CHECKPOINT"
nvidia-smi

python -m repostguard.evaluate \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output-directory "$INTERNAL"
THRESHOLD=$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["threshold_from_clean_validation"])' "$INTERNAL/summary.json")
python -m repostguard.evaluate \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --matrix configs/clean_only.yaml \
    --val-manifest data/manifests/community_forensics_test_external_unseen_generator_v3_expanded.csv \
    --threshold "$THRESHOLD" \
    --output-directory "$UNSEEN"
python scripts/check_distillation_teacher_acceptance.py \
    --experiment "$EXPERIMENT" \
    --internal-summary "$INTERNAL/summary.json" \
    --unseen-summary "$UNSEEN/summary.json" \
    | tee "$OUTPUT_DIRECTORY/teacher_acceptance_v3.json"
test -s "$OUTPUT_DIRECTORY/teacher_acceptance_v3.json"
