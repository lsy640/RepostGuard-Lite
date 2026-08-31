#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodelist=TC2N03
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --mem=6G
#SBATCH --job-name=rg_v32_smoke
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-/home/msai/xjiang026/projects/repostguard-lite}
ENV_PREFIX=${ENV_PREFIX:-/home/msai/xjiang026/.conda/envs/repostguard}
CONFIG=configs/community_forensics_v3/student_v32_corrected_smoke.yaml

cd "$PROJECT_ROOT"
module load anaconda
set +u
eval "$(conda shell.bash hook)"
conda activate "$ENV_PREFIX"
set -u
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p logs

python scripts/validate_family_unseen_dev_split.py \
    --train data/manifests/community_forensics_train_v31_family_holdout.csv \
    --dev data/manifests/community_forensics_val_family_unseen_dev_v1.csv \
    --protected data/manifests/community_forensics_test_external_unseen_generator_v3_expanded.csv
python scripts/validate_teacher_affine_calibration.py --config "$CONFIG"
python scripts/validate_teacher_cache_shard.py \
    --config "$CONFIG" --shard-index 0 --num-shards 1
python -m repostguard.train --config "$CONFIG"
python scripts/export_student_mobile.py \
    --config "$CONFIG" \
    --checkpoint outputs/community_forensics_v32/student_corrected_smoke/best.pt \
    --output-directory outputs/community_forensics_v32/student_corrected_smoke/mobile_export \
    --format both
python scripts/validate_mobile_onnx.py \
    --config "$CONFIG" \
    --checkpoint outputs/community_forensics_v32/student_corrected_smoke/best.pt \
    --onnx outputs/community_forensics_v32/student_corrected_smoke/mobile_export/student_mnv3_fp32.onnx
