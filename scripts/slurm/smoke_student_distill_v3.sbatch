#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodelist=TC2N03
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --mem=30G
#SBATCH --job-name=rg_v3_student_smoke
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/msai/xjiang026/projects/repostguard-lite}
ENV_PREFIX=${ENV_PREFIX:-/home/msai/xjiang026/.conda/envs/repostguard}
DUAL_TEMPLATE=configs/community_forensics_v3/student_mnv3_dual_teacher_smoke.yaml
DUAL_CONFIG=configs/community_forensics_v3/student_mnv3_dual_teacher_smoke_frozen.yaml
M3_TEMPLATE=configs/community_forensics_v3/student_mnv3_m3_primary_smoke.yaml
M3_CONFIG=configs/community_forensics_v3/student_mnv3_m3_primary_smoke_frozen.yaml
FULL_TEMPLATE=configs/community_forensics_v3/student_mnv3_dual_teacher.yaml
FULL_CONFIG=configs/community_forensics_v3/student_mnv3_dual_teacher_frozen.yaml
CALIBRATION=outputs/community_forensics_v3/student_mnv3_dual_teacher/teacher_calibration.json
CACHE=data/cache/teacher_logits/community_forensics_m2_m3_smoke_v3/teacher_cache_000-of-001.pt
DUAL_OUTPUT=outputs/community_forensics_v3/student_mnv3_dual_teacher_smoke
M3_OUTPUT=outputs/community_forensics_v3/student_mnv3_m3_primary_smoke
DECISION=outputs/community_forensics_v3/student_mnv3_dual_teacher/student_smoke_selection.json
DUAL_M2_WEIGHT=${STUDENT_DUAL_M2_WEIGHT:-0.3}

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
mkdir -p logs "$(dirname "$DECISION")"
test -s data/raw/community_forensics/TRAIN_V3_COMPLETE
test -s outputs/community_forensics_v3/m2/best.pt
test -s outputs/community_forensics_v3/m3/best.pt
test -s "$CALIBRATION"
if [[ -e outputs/community_forensics_v3/student_mnv3_dual_teacher/latest.pt ]] || \
   [[ -e outputs/community_forensics_v3/student_mnv3_dual_teacher/DONE ]]; then
    echo "Student full training has already started; refusing to change its frozen config."
    exit 3
fi
nvidia-smi

python scripts/build_student_smoke_manifests.py \
    --train-input data/manifests/community_forensics_train_v3.csv \
    --validation-input data/manifests/community_forensics_val_unseen_generator.csv \
    --train-output data/manifests/community_forensics_train_v3_student_smoke.csv \
    --validation-output data/manifests/community_forensics_val_student_smoke_v3.csv \
    --train-per-label 500 \
    --validation-per-label 250

python scripts/freeze_student_distillation_config.py \
    --config "$DUAL_TEMPLATE" \
    --calibration "$CALIBRATION" \
    --output "$DUAL_CONFIG" \
    --m2-weight "$DUAL_M2_WEIGHT" \
    --overwrite
python scripts/freeze_student_distillation_config.py \
    --config "$M3_TEMPLATE" \
    --calibration "$CALIBRATION" \
    --output "$M3_CONFIG" \
    --m2-weight 0 \
    --overwrite

if [[ ! -s "$CACHE" ]]; then
    python scripts/cache_teacher_logits.py \
        --config "$DUAL_CONFIG" \
        --shard-index 0 \
        --num-shards 1 \
        --batch-size 8
fi

python -m repostguard.train --config "$DUAL_CONFIG" --resume none
test -s "$DUAL_OUTPUT/best.pt"
python -m repostguard.evaluate \
    --config "$DUAL_CONFIG" \
    --checkpoint "$DUAL_OUTPUT/best.pt" \
    --output-directory "$DUAL_OUTPUT/evaluation"
test -s "$DUAL_OUTPUT/evaluation/summary.json"

python -m repostguard.train --config "$M3_CONFIG" --resume none
test -s "$M3_OUTPUT/best.pt"
python -m repostguard.evaluate \
    --config "$M3_CONFIG" \
    --checkpoint "$M3_OUTPUT/best.pt" \
    --output-directory "$M3_OUTPUT/evaluation"
test -s "$M3_OUTPUT/evaluation/summary.json"

python scripts/select_student_smoke_candidate.py \
    --calibration "$CALIBRATION" \
    --dual-summary "$DUAL_OUTPUT/evaluation/summary.json" \
    --m3-summary "$M3_OUTPUT/evaluation/summary.json" \
    --output "$DECISION" \
    --dual-m2-weight "$DUAL_M2_WEIGHT" \
    --minimum-robust-auroc-gain 0.002 \
    --maximum-clean-auroc-regression 0.001
WINNER=$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["winner"])' "$DECISION")
if [[ "$WINNER" == "m3_primary" ]]; then
    python scripts/freeze_student_distillation_config.py \
        --config "$FULL_TEMPLATE" \
        --calibration "$CALIBRATION" \
        --output "$FULL_CONFIG" \
        --m2-weight 0 \
        --overwrite
else
    python scripts/freeze_student_distillation_config.py \
        --config "$FULL_TEMPLATE" \
        --calibration "$CALIBRATION" \
        --output "$FULL_CONFIG" \
        --m2-weight "$DUAL_M2_WEIGHT" \
        --overwrite
fi
test -s "$FULL_CONFIG"
sha256sum "$DECISION" "$FULL_CONFIG"
