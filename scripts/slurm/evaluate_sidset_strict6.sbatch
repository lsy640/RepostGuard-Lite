#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --nodelist=TC2N08
#SBATCH --time=03:00:00
#SBATCH --mem=24G
#SBATCH --job-name=rg_sid_strict6
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/home/msai/lius0131/AGI/repostguard-lite
ENV_PREFIX=/home/msai/lius0131/.conda/envs/repostguard
EXPERIMENTS=(b0 b1 b2 m2 m3)
MATRIX="configs/sidset_strict6.yaml"

cd "$PROJECT_ROOT"
module load anaconda
set +u
eval "$(conda shell.bash hook)"
conda activate "$ENV_PREFIX"
set -u

export PYTHONUNBUFFERED=1
export TORCH_HOME="$PROJECT_ROOT/data/cache/torch"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p logs outputs/sidset_strict6

nvidia-smi
for EXPERIMENT in "${EXPERIMENTS[@]}"; do
    CONFIG="configs/sidset/$EXPERIMENT.yaml"
    CHECKPOINT="outputs/sidset/$EXPERIMENT/best.pt"
    OUTPUT_DIRECTORY="outputs/sidset_strict6/$EXPERIMENT"
    if [[ ! -s "$CHECKPOINT" ]]; then
        echo "Missing checkpoint: $CHECKPOINT" >&2
        exit 2
    fi
    mkdir -p "$OUTPUT_DIRECTORY"
    echo "Evaluating $EXPERIMENT with strict_random_six"
    python -m repostguard.evaluate \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --matrix "$MATRIX" \
        --output-directory "$OUTPUT_DIRECTORY"
done
