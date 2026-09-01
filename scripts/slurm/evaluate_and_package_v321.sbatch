#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodelist=TC2N03
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=30G
#SBATCH --job-name=rg_v321_complete_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/msai/xjiang026/projects/repostguard-lite}
ENV_PREFIX=${ENV_PREFIX:-/home/msai/xjiang026/.conda/envs/repostguard}
CONFIG=configs/community_forensics_v3/student_v32_corrected.yaml
CHECKPOINT=outputs/releases/v32_corrected_family_unseen_epoch3/best.pt
CHECKPOINT_SHA256=05edd57825cf28608d26fe77db92f364f37dcf7f070d757e82f85b9e2711cfea
MANIFEST=data/manifests/community_forensics_val_v32_full_refit_e20.csv
SOURCE=outputs/community_forensics_v32/student_corrected
RELEASE=outputs/releases/v32_corrected_family_unseen_epoch3
FIXED_OUTPUT="$RELEASE/fixed1500_clean"
FROZEN_THRESHOLD=0.060516357421875

cd "$PROJECT_ROOT"
module load anaconda
set +u
eval "$(conda shell.bash hook)"
conda activate "$ENV_PREFIX"
set -u
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export REPOSTGUARD_NUM_WORKERS="$SLURM_CPUS_PER_TASK"

test -s "$CONFIG"
test -s "$CHECKPOINT"
test -s "$MANIFEST"
test "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" = "$CHECKPOINT_SHA256"
python scripts/validate_training_checkpoint.py \
    --config "$CONFIG" --checkpoint "$CHECKPOINT"

python - "$CHECKPOINT" "$MANIFEST" <<'PY'
import csv
import sys
import torch

checkpoint_path, manifest_path = sys.argv[1:]
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
expected = {"epoch": 3, "global_step": 1545, "best_metric": 0.8369652909749363}
for key, value in expected.items():
    actual = checkpoint.get(key)
    if isinstance(value, float):
        accepted = actual is not None and abs(float(actual) - value) <= 1e-12
    else:
        accepted = actual == value
    if not accepted:
        raise SystemExit(f"V3.2.1 checkpoint mismatch: {key}={actual!r}")
with open(manifest_path, newline="", encoding="utf-8-sig") as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != 1500 or sum(int(row["label"]) == 0 for row in rows) != 750:
    raise SystemExit("Fixed validation manifest is not the expected balanced 1500 set")
print("V3.2.1 checkpoint and fixed-1500 manifest gates passed")
PY

if [[ -s "$FIXED_OUTPUT/summary.json" \
      && -s "$FIXED_OUTPUT/run_card.json" \
      && -s "$FIXED_OUTPUT/metrics_by_transform.csv" \
      && -s "$FIXED_OUTPUT/predictions.jsonl" ]]; then
    echo "Validated existing V3.2.1 fixed-1500 inference; skipping duplicate inference."
else
    python -m repostguard.evaluate \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --matrix configs/clean_only.yaml \
        --val-manifest "$MANIFEST" \
        --threshold "$FROZEN_THRESHOLD" \
        --output-directory "$FIXED_OUTPUT"
fi

test -s "$FIXED_OUTPUT/summary.json"
test -s "$FIXED_OUTPUT/run_card.json"
test -s "$FIXED_OUTPUT/metrics_by_transform.csv"
test -s "$FIXED_OUTPUT/predictions.jsonl"

python scripts/summarize_v321_fixed1500.py \
    --manifest "$MANIFEST" \
    --predictions "$FIXED_OUTPUT/predictions.jsonl" \
    --output-directory "$FIXED_OUTPUT"

# Archive existing completed evaluations without rerunning them.
test -s "$SOURCE/latest.pt"
test -s "$SOURCE/dev_family_unseen_robustness/predictions.jsonl"
test -s "$SOURCE/dev_family_unseen_robustness/quality_gate_audit.json"
test -s "$SOURCE/final_external_v3_expanded_robustness_v2/summary.json"
test -s "$SOURCE/final_external_v3_expanded_robustness_v2/predictions.jsonl"
cp -p "$SOURCE/latest.pt" "$RELEASE/latest.pt"
cp -p "$SOURCE/dev_family_unseen_robustness/predictions.jsonl" \
    "$RELEASE/dev_family_unseen_robustness/predictions.jsonl"
cp -p "$SOURCE/dev_family_unseen_robustness/quality_gate_audit.json" \
    "$RELEASE/dev_family_unseen_robustness/quality_gate_audit.json"
mkdir -p "$RELEASE/final_external_v3_expanded_robustness_v2"
cp -p "$SOURCE/final_external_v3_expanded_robustness_v2/"* \
    "$RELEASE/final_external_v3_expanded_robustness_v2/"

python - "$RELEASE" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

release = Path(sys.argv[1])
dev = json.loads((release / "dev_family_unseen_robustness/summary.json").read_text())
external = json.loads(
    (release / "final_external_v3_expanded_robustness_v2/summary.json").read_text()
)
fixed = json.loads((release / "fixed1500_clean/fixed1500_v321_metrics.json").read_text())
overall = fixed["overall"]["frozen_threshold"]

readme = f"""# Student V3.2.1 release

V3.2.1 is the frozen family-unseen V3.2 corrected Student. It uses the epoch-3 checkpoint selected without the protected external 4k set.

## Frozen checkpoint

- epoch: 3
- global step: 1545
- parameters: 7,955,038
- teacher: M3 100%, M2 0%
- checkpoint selection: family-unseen development set

## Evaluation summary

| Protocol | Clean AUROC | Clean BA | Robust mean AUROC | Robust worst AUROC |
|---|---:|---:|---:|---:|
| Family-unseen dev | {dev['clean_auroc']:.6f} | {dev['clean_balanced_accuracy']:.6f} | {dev['robust_mean_auroc']:.6f} | {dev['robust_worst_auroc']:.6f} |
| Protected external 4k | {external['clean_auroc']:.6f} | {external['clean_balanced_accuracy']:.6f} | {external['robust_mean_auroc']:.6f} | {external['robust_worst_auroc']:.6f} |
| Fixed 1500 hard-mixture clean | {overall['auroc']:.6f} | {overall['balanced_accuracy']:.6f} | n/a | n/a |

The protected external 4k and family-unseen results were archived from their existing completed runs; they were not rerun during release completion. See `fixed1500_clean/FIXED1500_EVALUATION.md` for SD1.4, DFGAN, GALIP, and Hourglass slices.
"""
(release / "README.md").write_text(readme, encoding="utf-8")

included = [
    path
    for path in release.rglob("*")
    if path.is_file() and path.name not in {"SHA256SUMS", "RELEASE_AUDIT.json"}
]
hashes = {}
for path in sorted(included):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    hashes[path.relative_to(release).as_posix()] = digest.hexdigest()
(release / "SHA256SUMS").write_text(
    "".join(f"{digest}  {path}\n" for path, digest in hashes.items()),
    encoding="utf-8",
)
audit = {
    "schema_version": 1,
    "release_name": "v3.2.1",
    "checkpoint_sha256": hashes["best.pt"],
    "fixed1500_inference_completed": True,
    "family_unseen_rerun": False,
    "external4k_rerun": False,
    "files_hashed": len(hashes),
}
(release / "RELEASE_AUDIT.json").write_text(
    json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(audit, sort_keys=True))
PY

test -s "$RELEASE/README.md"
test -s "$RELEASE/SHA256SUMS"
test -s "$RELEASE/RELEASE_AUDIT.json"
echo "V3.2.1 fixed-1500 evaluation and release completion finished."
