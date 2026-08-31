#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodelist=TC2N03
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --mem=30G
#SBATCH --job-name=rg_v32_full_cache
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/msai/xjiang026/projects/repostguard-lite}
ENV_PREFIX=${ENV_PREFIX:-/home/msai/xjiang026/.conda/envs/repostguard}
CONFIG=configs/community_forensics_v3/student_v32_full_refit_e20.yaml
CACHE_DIRECTORY=data/cache/teacher_logits/community_forensics_m3_v32_full_refit_e20

cd "$PROJECT_ROOT"
module load anaconda
set +u
eval "$(conda shell.bash hook)"
conda activate "$ENV_PREFIX"
set -u
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export REPOSTGUARD_NUM_WORKERS="$SLURM_CPUS_PER_TASK"
mkdir -p logs "$CACHE_DIRECTORY"

python scripts/validate_student_v32_full_refit_data.py \
    --train data/manifests/community_forensics_train_v3.csv \
    --val data/manifests/community_forensics_val_v32_full_refit_e20.csv \
    --protected data/manifests/community_forensics_test_external_unseen_generator_v3_expanded.csv \
    --expected-train-rows 24000 \
    --expected-train-sha256 fc0a7ab732faeb604ed1e77281fada715d7cffb353974a4985820548d871d9d6
python scripts/validate_teacher_affine_calibration.py --config "$CONFIG"

if [[ -s "$CACHE_DIRECTORY/teacher_cache_000-of-001.pt" ]]; then
    python scripts/validate_teacher_cache_shard.py \
        --config "$CONFIG" --shard-index 0 --num-shards 1
    echo "Validated existing full-refit cache; refusing duplicate cache work."
    exit 0
fi

python scripts/cache_teacher_logits.py \
    --config "$CONFIG" \
    --shard-index 0 \
    --num-shards 1 \
    --batch-size 8
python scripts/validate_teacher_cache_shard.py \
    --config "$CONFIG" --shard-index 0 --num-shards 1
