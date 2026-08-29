#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --job-name=rg_cf_v2_rv2_check
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/home/msai/lius0131/AGI/repostguard-lite
ENV_PREFIX=/home/msai/lius0131/.conda/envs/repostguard

cd "$PROJECT_ROOT"
module load anaconda
set +u
eval "$(conda shell.bash hook)"
conda activate "$ENV_PREFIX"
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p logs
test -s data/raw/community_forensics/TRAIN_V2_COMPLETE

pytest -q \
    tests/test_transforms.py \
    tests/test_evaluate.py \
    tests/test_config.py
for EXPERIMENT in b0 b1 b2 m2 m3; do
    python scripts/validate_training_checkpoint.py \
        --config "outputs/community_forensics_v2/${EXPERIMENT}/resolved_config.yaml" \
        --checkpoint "outputs/community_forensics_v2/${EXPERIMENT}/best.pt"
done
python scripts/generate_community_forensics_robustness_v2_report.py --help
