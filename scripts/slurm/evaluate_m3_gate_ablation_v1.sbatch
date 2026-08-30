#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=TC2N08
#SBATCH --time=03:00:00
#SBATCH --mem=24G
#SBATCH --job-name=rg_m3_gate_ablate
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/home/msai/lius0131/AGI/repostguard-lite
ENV_PREFIX=/home/msai/lius0131/.conda/envs/repostguard
PROTOCOL=configs/community_forensics_m3_gate_ablation_v1.yaml
OUTPUT_ROOT=outputs/community_forensics_v3_m3_gate_ablation_v1

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
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p logs "$OUTPUT_ROOT"

test -s "$PROTOCOL"
test -s outputs/community_forensics_v3/m3/best.pt
test -s outputs/community_forensics_v3/m3/DONE
test -s data/manifests/community_forensics_test_external_unseen_generator_v3_expanded.csv
test -s configs/community_forensics_robustness_v2.yaml
test -s outputs/community_forensics_v3_robustness_v2/m3/unseen_generator_expanded/COMPLETE

python -m py_compile \
    src/repostguard/models/detectors.py \
    scripts/evaluate_m3_gate_ablation.py
nvidia-smi
python scripts/evaluate_m3_gate_ablation.py --protocol "$PROTOCOL"

test -s "$OUTPUT_ROOT/COMPLETE"
test -s "$OUTPUT_ROOT/summary.json"
test -s "$OUTPUT_ROOT/run_card.json"
test -s "$OUTPUT_ROOT/metrics_by_mode_and_transform.csv"
test -s "$OUTPUT_ROOT/paired_bootstrap_key_conditions.csv"
test -s "$OUTPUT_ROOT/predictions_by_mode.jsonl"
jq -e '.protocol_id == "community_forensics_train_v3_m3_gate_ablation_v1"' "$OUTPUT_ROOT/summary.json" >/dev/null
jq -e '.modes | length == 4' "$OUTPUT_ROOT/summary.json" >/dev/null
jq -e '.learned_reconciliation.maximum_absolute_difference <= 1e-6' "$OUTPUT_ROOT/summary.json" >/dev/null
test "$(($(wc -l < "$OUTPUT_ROOT/metrics_by_mode_and_transform.csv") - 1))" -eq 84
test "$(($(wc -l < "$OUTPUT_ROOT/paired_bootstrap_key_conditions.csv") - 1))" -eq 12
test "$(wc -l < "$OUTPUT_ROOT/predictions_by_mode.jsonl")" -eq 84000
sha256sum \
    "$OUTPUT_ROOT/summary.json" \
    "$OUTPUT_ROOT/run_card.json" \
    "$OUTPUT_ROOT/metrics_by_mode_and_transform.csv" \
    "$OUTPUT_ROOT/paired_bootstrap_key_conditions.csv" \
    "$OUTPUT_ROOT/predictions_by_mode.jsonl"
