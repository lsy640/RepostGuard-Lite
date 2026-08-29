#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --job-name=rg_cf_v2_test
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module load anaconda
set +u
eval "$(conda shell.bash hook)"
conda activate /home/msai/lius0131/.conda/envs/repostguard
set -u

export PYTHONUNBUFFERED=1
PROJECT_ROOT=/home/msai/lius0131/AGI/repostguard-lite
cd "$PROJECT_ROOT"
mkdir -p logs

test -f data/raw/community_forensics/TRAIN_V2_COMPLETE
test -f data/manifests/community_forensics_train_v2.csv

python -m pytest -q \
  tests/test_config.py \
  tests/test_community_forensics.py \
  tests/test_transforms.py \
  tests/test_evaluate.py

python scripts/build_community_forensics_train_v2.py --help >/dev/null
python scripts/generate_community_forensics_data_statistics_report.py --help >/dev/null
python scripts/generate_community_forensics_robustness_v2_report.py --help >/dev/null
python scripts/generate_community_forensics_unseen_accuracy_report.py --help >/dev/null
python scripts/generate_community_forensics_train_v1_v2_comparison_report.py --help >/dev/null
python scripts/build_community_forensics_exact_generator_atlas.py --help >/dev/null

echo "TRAIN_V2_WORKFLOW_TESTS_PASSED"
