#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodelist=TC2N03
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --mem=30G
#SBATCH --job-name=rg_v31_feature_cache
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/msai/xjiang026/projects/repostguard-lite}
ENV_PREFIX=${ENV_PREFIX:-/home/msai/xjiang026/.conda/envs/repostguard}
CONFIG=configs/community_forensics_v3/student_v31_ablation_feature_forensic.yaml
CACHE_DIRECTORY=data/cache/teacher_logits/community_forensics_m3_v31_features

cd "$PROJECT_ROOT"
module load anaconda
set +u
eval "$(conda shell.bash hook)"
conda activate "$ENV_PREFIX"
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p logs "$CACHE_DIRECTORY"
python scripts/validate_family_unseen_dev_split.py \
    --train data/manifests/community_forensics_train_v31_family_holdout.csv \
    --dev data/manifests/community_forensics_val_family_unseen_dev_v1.csv \
    --protected data/manifests/community_forensics_test_external_unseen_generator_v3_expanded.csv
if [[ -s "$CACHE_DIRECTORY/teacher_cache_000-of-001.pt" ]]; then
    python scripts/validate_teacher_cache_shard.py \
        --config "$CONFIG" --shard-index 0 --num-shards 1
    echo "Validated existing v3.1 feature cache; refusing duplicate work."
    exit 0
fi
python scripts/cache_teacher_logits.py \
    --config "$CONFIG" \
    --shard-index 0 \
    --num-shards 1 \
    --batch-size 8
python scripts/validate_teacher_cache_shard.py \
    --config "$CONFIG" --shard-index 0 --num-shards 1
