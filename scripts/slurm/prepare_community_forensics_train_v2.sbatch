#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --job-name=rg_cf_train_v2
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
mkdir -p logs data/manifests data/raw/community_forensics

test -s data/raw/community_forensics/COMPLETE
test -s data/raw/community_forensics/VALIDATION_V2_COMPLETE
test -s data/manifests/community_forensics_train.csv
test -s data/manifests/community_forensics_test_external_seen_family.csv
test -s data/manifests/community_forensics_test_external_unseen_generator.csv
test -s data/manifests/community_forensics_audit.json

python -m pytest -q tests/test_config.py
python scripts/build_community_forensics_train_v2.py

test -s data/manifests/community_forensics_train_v2.csv
test -s data/manifests/community_forensics_train_v2_audit.json
test -s data/raw/community_forensics/TRAIN_V2_COMPLETE

jq -e '.output.rows == 20000' \
    data/manifests/community_forensics_train_v2_audit.json >/dev/null
jq -e '.output.class_counts == {"aigi":10000,"real":10000}' \
    data/manifests/community_forensics_train_v2_audit.json >/dev/null
jq -e '.output.exact_generator_count == 909' \
    data/manifests/community_forensics_train_v2_audit.json >/dev/null
jq -e '.output.promoted_exact_generators | length == 9' \
    data/manifests/community_forensics_train_v2_audit.json >/dev/null
jq -e '.policy.excluded_from_future_evaluation == true' \
    data/manifests/community_forensics_train_v2_audit.json >/dev/null
jq -e '.integrity.train_v2_strict_unseen_generator_overlap == []' \
    data/manifests/community_forensics_train_v2_audit.json >/dev/null
jq -e '.integrity.train_v2_strict_unseen_architecture_overlap == []' \
    data/manifests/community_forensics_train_v2_audit.json >/dev/null
jq -e '[.hard_slices[] | (.train_v2_overlap | all(. == 0))] | all' \
    data/manifests/community_forensics_train_v2_audit.json >/dev/null
jq -e '[.hard_slices[] | (.duplicates | all(. == 0))] | all' \
    data/manifests/community_forensics_train_v2_audit.json >/dev/null
jq -e '[.hard_slices[].materialized_paths_and_sizes_verified] | all' \
    data/manifests/community_forensics_train_v2_audit.json >/dev/null
jq -e '.integrity.quality_profile.required_field_completeness_rate == 1' \
    data/manifests/community_forensics_train_v2_audit.json >/dev/null
jq -e '.integrity.quality_profile.role_consistency_rate == 1' \
    data/manifests/community_forensics_train_v2_audit.json >/dev/null
jq -e '.inputs.base_audit.cross_split_phash_near_duplicate_count == 0' \
    data/manifests/community_forensics_train_v2_audit.json >/dev/null

sha256sum \
    data/manifests/community_forensics_train_v2.csv \
    data/manifests/community_forensics_train_v2_audit.json \
    data/manifests/community_forensics_val_hard_hourglass_v2_exact_seen.csv \
    data/manifests/community_forensics_val_hard_dfgan_v2_exact_seen.csv \
    data/manifests/community_forensics_val_hard_galip_v2_exact_seen.csv
