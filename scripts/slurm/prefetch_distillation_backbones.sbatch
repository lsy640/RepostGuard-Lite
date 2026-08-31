#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodelist=TC2N03
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=30G
#SBATCH --job-name=rg_kd_prefetch
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/msai/xjiang026/projects/repostguard-lite}
ENV_PREFIX=${ENV_PREFIX:-/home/msai/xjiang026/.conda/envs/repostguard}

cd "$PROJECT_ROOT"
module load anaconda
set +u
eval "$(conda shell.bash hook)"
conda activate "$ENV_PREFIX"
set -u

export PYTHONUNBUFFERED=1
export TORCH_HOME="$PROJECT_ROOT/data/cache/torch"
export HF_HOME="$PROJECT_ROOT/data/cache/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p logs "$TORCH_HOME" "$HF_HOME"
nvidia-smi

python - <<'PY'
import gc

from repostguard.config import load_config
from repostguard.models import build_model

for path in (
    "configs/community_forensics/distillation_teacher_m2.yaml",
    "configs/community_forensics/student_mnv3_dual_teacher.yaml",
):
    model = build_model(load_config(path), load_pretrained=True)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"prefetched {path}: parameters={parameters}", flush=True)
    del model
    gc.collect()
PY

find data/cache/torch data/cache/open_clip data/cache/huggingface \
    -maxdepth 3 -type f -printf '%p %s\n' 2>/dev/null | sort
