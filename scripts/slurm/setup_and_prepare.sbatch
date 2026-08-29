#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --mem=16G
#SBATCH --job-name=rg_setup
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/home/msai/lius0131/AGI/repostguard-lite
ENV_PREFIX=/home/msai/lius0131/.conda/envs/repostguard
cd "$PROJECT_ROOT"

module load anaconda
set +u
eval "$(conda shell.bash hook)"
set -u

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    conda env create --prefix "$ENV_PREFIX" --file environment.yml
else
    echo "Using existing project environment: $ENV_PREFIX"
fi

set +u
conda activate "$ENV_PREFIX"
set -u

export PYTHONUNBUFFERED=1
export TORCH_HOME="$PROJECT_ROOT/data/cache/torch"
mkdir -p data/cache/open_clip data/manifests logs
python -m pip install --no-deps --editable .

df -h "$PROJECT_ROOT"
du -sh "$ENV_PREFIX" data 2>/dev/null || true
python scripts/download_cifake.py --output-dir data/raw/cifake
python scripts/build_cifake_manifest.py \
    --data-root data/raw/cifake \
    --output-dir data/manifests \
    --train-per-class 5000 \
    --test-per-class 1000 \
    --seed 20260827
python scripts/precache_models.py --clip-cache data/cache/open_clip
python -m pytest
du -sh "$ENV_PREFIX" data
