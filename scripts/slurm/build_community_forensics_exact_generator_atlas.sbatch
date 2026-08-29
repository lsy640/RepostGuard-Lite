#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --job-name=rg_cf_gen_atlas
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/home/msai/lius0131/AGI/repostguard-lite
ENV_PREFIX=/home/msai/lius0131/.conda/envs/repostguard
ATLAS_ROOT=reports/atlases/train_v2
MASTER="$ATLAS_ROOT/community_forensics_exact_generators_atlas.jpg"
TRAIN="$ATLAS_ROOT/community_forensics_train_exact_generators_atlas.jpg"
TEST="$ATLAS_ROOT/community_forensics_test_exact_generators_atlas.jpg"
INDEX="$ATLAS_ROOT/community_forensics_exact_generators_atlas_index.csv"
AUDIT="$ATLAS_ROOT/community_forensics_exact_generators_atlas_audit.json"

cd "$PROJECT_ROOT"
module load anaconda
set +u
eval "$(conda shell.bash hook)"
conda activate "$ENV_PREFIX"
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p logs "$ATLAS_ROOT"

test -s data/raw/community_forensics/COMPLETE
test -s data/raw/community_forensics/VALIDATION_V2_COMPLETE
test -s data/manifests/community_forensics_train_v2.csv
test -s data/manifests/community_forensics_test_external_unseen_generator.csv

python scripts/build_community_forensics_exact_generator_atlas.py \
    --data-root data/raw/community_forensics \
    --output "$MASTER" \
    --train-output "$TRAIN" \
    --test-output "$TEST" \
    --index-csv "$INDEX" \
    --audit-json "$AUDIT" \
    --train-generator-limit 78 \
    --selection-seed 20260829 \
    --quality 92

test -s "$MASTER"
test -s "$TRAIN"
test -s "$TEST"
test -s "$INDEX"
test -s "$AUDIT"
test "$(wc -l < "$INDEX")" -eq 91
jq -e '
  .counts.total == 90 and
  .counts.train == 78 and
  .counts.unseen_family == 12 and
  .checks.expected_generator_counts_match == true and
  .checks.unique_generator_ids == 90 and
  .checks.unique_sample_ids == 90 and
  .checks.missing_selected_files == 0 and
  .checks.byte_size_mismatches == 0 and
  .train_generator_sampling.available_by_architecture.GAN == 4 and
  .train_generator_sampling.available_by_architecture.LatDiff == 903 and
  .train_generator_sampling.available_by_architecture.PixDiff == 2 and
  .train_generator_sampling.selected_by_architecture.GAN == 4 and
  .train_generator_sampling.selected_by_architecture.LatDiff == 72 and
  .train_generator_sampling.selected_by_architecture.PixDiff == 2
' "$AUDIT" >/dev/null

file "$MASTER" "$TRAIN" "$TEST"
sha256sum "$MASTER" "$TRAIN" "$TEST" "$INDEX" "$AUDIT"
