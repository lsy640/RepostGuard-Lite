#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodelist=TC2N03
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --mem=30G
#SBATCH --job-name=rg_student_smoke
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/msai/xjiang026/projects/repostguard-lite}
ENV_PREFIX=${ENV_PREFIX:-/home/msai/xjiang026/.conda/envs/repostguard}
CONFIG_TEMPLATE=configs/community_forensics/student_mnv3_dual_teacher_smoke.yaml
CONFIG=configs/community_forensics/student_mnv3_dual_teacher_smoke_frozen.yaml
CALIBRATION=outputs/community_forensics/student_mnv3_dual_teacher/teacher_calibration.json
CACHE=data/cache/teacher_logits/community_forensics_m2_m3_smoke/teacher_cache_000-of-001.pt
OUTPUT=outputs/community_forensics/student_mnv3_dual_teacher_smoke

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
test -s outputs/community_forensics/distillation_teacher_m2/best.pt
test -s outputs/community_forensics/distillation_teacher_m3/best.pt
test -s "$CALIBRATION"
nvidia-smi

python scripts/build_student_smoke_manifests.py \
    --train-input data/manifests/community_forensics_train.csv \
    --validation-input data/manifests/community_forensics_val_unseen_generator.csv \
    --train-output data/manifests/community_forensics_train_student_smoke.csv \
    --validation-output data/manifests/community_forensics_val_student_smoke.csv

python scripts/freeze_student_distillation_config.py \
    --config "$CONFIG_TEMPLATE" \
    --calibration "$CALIBRATION" \
    --output "$CONFIG" \
    --overwrite

if [[ ! -s "$CACHE" ]]; then
    python scripts/cache_teacher_logits.py \
        --config "$CONFIG" \
        --shard-index 0 \
        --num-shards 1 \
        --batch-size 4
fi

python -m repostguard.train --config "$CONFIG" --resume none
test -s "$OUTPUT/best.pt"
python -m repostguard.evaluate \
    --config "$CONFIG" \
    --checkpoint "$OUTPUT/best.pt" \
    --matrix configs/clean_only.yaml \
    --output-directory "$OUTPUT/evaluation"
test -s "$OUTPUT/evaluation/summary.json"
