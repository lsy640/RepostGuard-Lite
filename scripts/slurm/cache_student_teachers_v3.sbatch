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
#SBATCH --job-name=rg_v3_teacher_cache
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/msai/xjiang026/projects/repostguard-lite}
ENV_PREFIX=${ENV_PREFIX:-/home/msai/xjiang026/.conda/envs/repostguard}
SHARD_INDEX=${TEACHER_SHARD_INDEX:-${SLURM_ARRAY_TASK_ID}}
NUM_SHARDS=${TEACHER_NUM_SHARDS:-2}
CONFIG=${STUDENT_CONFIG:-configs/community_forensics_v3/student_mnv3_dual_teacher.yaml}

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
mkdir -p logs data/cache/teacher_logits/community_forensics_m2_m3_v3
test -s data/raw/community_forensics/TRAIN_V3_COMPLETE
test -s data/manifests/community_forensics_train_v3.csv
test -s outputs/community_forensics_v3/m2/best.pt
test -s outputs/community_forensics_v3/m3/best.pt
test -s "$CONFIG"
nvidia-smi

CACHE_PATH=$(printf \
    "data/cache/teacher_logits/community_forensics_m2_m3_v3/teacher_cache_%03d-of-%03d.pt" \
    "$SHARD_INDEX" "$NUM_SHARDS")
if [[ ! -s "$CACHE_PATH" ]]; then
    python scripts/cache_teacher_logits.py \
        --config "$CONFIG" \
        --shard-index "$SHARD_INDEX" \
        --num-shards "$NUM_SHARDS" \
        --batch-size "${TEACHER_CACHE_BATCH_SIZE:-8}"
fi
python scripts/validate_teacher_cache_shard.py \
    --config "$CONFIG" \
    --shard-index "$SHARD_INDEX" \
    --num-shards "$NUM_SHARDS"
