#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodelist=TC2N03
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --mem=30G
#SBATCH --job-name=rg_v32_full_smoke
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/msai/xjiang026/projects/repostguard-lite}
ENV_PREFIX=${ENV_PREFIX:-/home/msai/xjiang026/.conda/envs/repostguard}
CONFIG=configs/community_forensics_v3/student_v32_full_refit_e20_smoke.yaml
OUTPUT_DIRECTORY=outputs/community_forensics_v32_full_refit_e20/student_corrected_smoke

cd "$PROJECT_ROOT"
module load anaconda
set +u
eval "$(conda shell.bash hook)"
conda activate "$ENV_PREFIX"
set -u
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export REPOSTGUARD_NUM_WORKERS="$SLURM_CPUS_PER_TASK"
mkdir -p logs

python scripts/validate_student_v32_full_refit_data.py \
    --train data/manifests/community_forensics_train_v3.csv \
    --val data/manifests/community_forensics_val_v32_full_refit_e20.csv \
    --protected data/manifests/community_forensics_test_external_unseen_generator_v3_expanded.csv \
    --expected-train-rows 24000 \
    --expected-train-sha256 fc0a7ab732faeb604ed1e77281fada715d7cffb353974a4985820548d871d9d6
python scripts/validate_teacher_affine_calibration.py --config "$CONFIG"
python scripts/validate_teacher_cache_shard.py \
    --config "$CONFIG" --shard-index 0 --num-shards 1

if [[ -s "$OUTPUT_DIRECTORY/DONE" ]]; then
    python scripts/validate_training_checkpoint.py \
        --config "$CONFIG" --checkpoint "$OUTPUT_DIRECTORY/best.pt"
    python scripts/validate_training_checkpoint.py \
        --config "$CONFIG" --checkpoint "$OUTPUT_DIRECTORY/latest.pt"
    echo "Validated existing full-refit smoke output; refusing duplicate work."
    exit 0
fi

python -m repostguard.train --config "$CONFIG"
python scripts/validate_training_checkpoint.py \
    --config "$CONFIG" --checkpoint "$OUTPUT_DIRECTORY/best.pt"
python scripts/validate_training_checkpoint.py \
    --config "$CONFIG" --checkpoint "$OUTPUT_DIRECTORY/latest.pt"
test -s "$OUTPUT_DIRECTORY/DONE"
echo "Full-refit one-epoch smoke gate complete; mobile export intentionally skipped."
