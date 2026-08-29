#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --job-name=rg_cf_check
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

python scripts/build_community_forensics.py --help
python -m repostguard.train --help
python -m repostguard.evaluate --help
python -m pytest -q \
    tests/test_community_forensics.py \
    tests/test_config.py \
    tests/test_evaluate.py \
    tests/test_transforms.py \
    tests/test_forensic.py \
    tests/test_inference_schema.py \
    tests/test_losses.py \
    tests/test_quality_gate.py
