#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodelist=TC2N03
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=05:50:00
#SBATCH --mem=30G
#SBATCH --array=0-1%2
#SBATCH --job-name=rg_teacher_cache
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/msai/xjiang026/projects/repostguard-lite}
ENV_PREFIX=${ENV_PREFIX:-/home/msai/xjiang026/.conda/envs/repostguard}
SHARD_INDEX=${TEACHER_SHARD_INDEX:-${SLURM_ARRAY_TASK_ID}}
NUM_SHARDS=${TEACHER_NUM_SHARDS:-2}
CONFIG=${STUDENT_CONFIG:-configs/community_forensics/student_mnv3_dual_teacher.yaml}

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
mkdir -p logs
test -s data/raw/community_forensics/COMPLETE
test -s outputs/community_forensics/distillation_teacher_m2/best.pt
test -s outputs/community_forensics/distillation_teacher_m3/best.pt
nvidia-smi

python scripts/cache_teacher_logits.py \
    --config "$CONFIG" \
    --shard-index "$SHARD_INDEX" \
    --num-shards "$NUM_SHARDS" \
    --batch-size "${TEACHER_CACHE_BATCH_SIZE:-8}"
