#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodelist=TC2N03
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --mem=30G
#SBATCH --job-name=rg_data_verify
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/msai/xjiang026/projects/repostguard-lite}
ENV_PREFIX=${ENV_PREFIX:-/home/msai/xjiang026/.conda/envs/repostguard}
AUDIT=outputs/community_forensics/student_mnv3_dual_teacher/data_validation.json

cd "$PROJECT_ROOT"
module load anaconda
set +u
eval "$(conda shell.bash hook)"
conda activate "$ENV_PREFIX"
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p logs "$(dirname "$AUDIT")"
test -s data/raw/community_forensics/COMPLETE
nvidia-smi

python scripts/validate_student_data.py \
    --data-root data/raw/community_forensics \
    --manifest data/manifests/community_forensics_train.csv \
    --manifest data/manifests/community_forensics_val_unseen_generator.csv \
    --manifest data/manifests/community_forensics_val_external_exact_seen_generator.csv \
    --manifest data/manifests/community_forensics_val_hard_hourglass.csv \
    --manifest data/manifests/community_forensics_val_hard_dfgan.csv \
    --manifest data/manifests/community_forensics_val_hard_galip.csv \
    --manifest data/manifests/community_forensics_test_external_seen_family.csv \
    --manifest data/manifests/community_forensics_test_external_unseen_generator.csv \
    --workers "$SLURM_CPUS_PER_TASK" \
    --output "$AUDIT"

test -s "$AUDIT"
sha256sum "$AUDIT"
