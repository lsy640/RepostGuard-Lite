# RepostGuard-Lite

This repository implements the AIGC image robustness experiments defined in
`../AIGC图像鲁棒性检测项目可行性方案.md`. The current primary training and
evaluation track is the frozen Community Forensics dataset; CIFAKE and SID-Set
are retained as historical prototype lineages only.

## Current status and consolidated summary

The canonical project summary is
[`reports/summaries/COMMUNITY_FORENSICS_PROJECT_SUMMARY.md`](reports/summaries/COMMUNITY_FORENSICS_PROJECT_SUMMARY.md).
The complete report directory map and regeneration entry points are listed in
[`reports/README_reports.md`](reports/README_reports.md).
It documents the model architectures, training lineage, the protocol-v1 data
and results, format debiasing, internal checkpoint selection, external slices,
21 clean/perturbation conditions, fixed-threshold metrics,
stage-by-stage improvements, limitations, and artifact locations.

The metrics below are **protocol-v1 historical results** from checkpoints trained
before the seen-family cohort was promoted into training. They remain valid for
that frozen lineage, but are not results for the new train-v2 protocol. On the
balanced 2,000-image strict unseen-generator test:

- **M3** is the best fixed-threshold detector: clean Accuracy 79.30%, F1
  80.12%, AIGI Recall 83.40%, AUROC 0.8631 and AP 0.8206. Across the 20
  perturbations it retains 76.90% mean Accuracy and 71.05% worst Accuracy.
- **M2** is slightly more conservative, with 77.93% clean Precision and 77.20%
  Real Specificity.
- **B2** had the strongest cross-slice ranking: 0.7315 clean macro-AUROC across
  exact-seen, three hard generators, seen-family/exact-unseen and strict
  unseen-generator. Its frozen operating threshold has low AIGI recall, so this
  AUROC advantage does not make it the best current fixed-threshold detector.
- Hourglass, DFGAN and GALIP exposed a material M2/M3 generalization gap; the v1
  aggregate strict-unseen performance must not be treated as universal unseen-
  generator performance.

The five evaluated models are:

- **B0**: pretrained EfficientNet-B0, clean-only training.
- **B1**: the same EfficientNet-B0 with class-symmetric JPEG, blur, resize,
  Gaussian-noise, colour-jitter, and centre-crop augmentation.
- **B2**: frozen OpenCLIP image tower with a trainable linear detector.
- **M2**: frozen OpenCLIP plus a trainable DCT-ranked, SRM-inspired/NPR
  ResNet-18 forensic branch and paired clean/degraded consistency losses.
- **M3**: M2 plus a label-agnostic quality-aware gate that dynamically weights
  semantic and forensic features using blur, blockiness, noise, effective
  resolution and dynamic-range proxies.

The historical initial run used a deterministic 10,000-image CIFAKE training
subset and a 2,000-image official-test validation subset (balanced by class).
CIFAKE remains pipeline/pilot evidence only. A second, higher-resolution
SID-Set track used 10,000 real + 10,000 fully synthetic official-training
images and 2,000 real + 2,000 fully synthetic official-validation images.
SID-Set's tampered class is deliberately excluded from this binary whole-image
generation task. Their raw data and trained checkpoints have been removed to
free storage; the reproducibility scripts, configs and historical reports remain.

## Data contract

Manifests are CSV files with these required columns:

```text
sample_id,path,label,split,source_dataset,generator_id
```

`path` is relative to `data.root` in each experiment config. `label=1` means
AIGC/FAKE and `label=0` means authentic/REAL. Dataset code never uses file or
directory names as model inputs.

## Historical CIFAKE cluster workflow

All Python commands run through SLURM. From the TC2 Head Node:

```bash
cd /home/msai/lius0131/AGI/repostguard-lite
sbatch scripts/slurm/setup_and_prepare.sbatch
sbatch --dependency=afterok:<SETUP_JOB_ID> --array=0-1 \
  scripts/slurm/train_and_eval_pilot.sbatch
# TC2 normal currently limits submitted jobs to two; after the first batch ends:
sbatch --array=2-3 scripts/slurm/train_and_eval_pilot.sbatch
```

The array indices map to `B0`, `B1`, `B2`, and `M2`. Logs are written under
`logs/`; checkpoints, metrics, predictions and run cards are written beneath
`outputs/<experiment>/`.

The pilot observed a 12.22 GB maximum resident set and a 3:29 maximum wall
time, so the reusable training script requests 16 GB and one hour. The current
QoS permits only one GPU per user, so array tasks execute sequentially even
though two submitted jobs are allowed.

## Historical SID-Set workflow (not currently materialized)

The public SID-Set repository is about 140 GB. The historical builder streamed
and materialised only the requested 24,000 images. The downloader records the
resolved Hugging Face revision, original image IDs, SHA-256 hashes, dimensions,
formats, class counts, and exact train/validation overlap. It is deterministic
and resumable; an interrupted job can be submitted again without discarding
completed images.

```bash
cd /home/msai/lius0131/AGI/repostguard-lite
sbatch scripts/slurm/prepare_sidset.sbatch

# After preparation succeeds, submit at most two array elements at once under
# the current normal QoS. Only one GPU task can run at a time.
sbatch --array=0-1 scripts/slurm/train_and_eval_sidset.sbatch
sbatch --array=2-3 scripts/slurm/train_and_eval_sidset.sbatch
sbatch --array=4 scripts/slurm/train_and_eval_sidset.sbatch
```

Array indices 0--4 map to B0, B1, B2, M2, and M3. SID-Set manifests and the audit are
written to `data/manifests/sidset_train.csv`,
`data/manifests/sidset_validation.csv`, and
`data/manifests/sidset_subset_audit.json`. The CIFAKE and SID-Set configs and
scripts remain for reproducibility, but their raw subsets and trained output
directories are no longer present. Re-running this workflow will download or
materialise SID-Set again and retrain the models.

### SID-Set format debiasing

SID-Set's real subset is stored almost entirely as JPEG/MPO while its fully
synthetic subset is PNG, creating a severe label-format shortcut. SID-Set
configs therefore apply mandatory **on-the-fly** format equalisation during
image reading; no duplicate dataset is written:

1. decode the source and discard container metadata;
2. bicubic-resize every class to 224x224, disrupting the original JPEG block
   grid;
3. JPEG-roundtrip every class with the same settings;
4. during training, sample quality from 70/80/90/95 independently of label;
5. during validation, use fixed quality 90 for deterministic metrics;
6. apply B1/M2 robustness augmentation only after this common format path.

The original CIFAKE configs keep this feature disabled, preserving the pilot's
reproducibility. Equalisation materially reduces the shortcut but cannot prove
that every trace of an image's prior compression history has been removed, so
source-format-stratified error analysis remains necessary.

## Community Forensics train-v2 track

The materialized source pool follows
`../Community_Forensics训练与测试集构建方案.md` and pins the public
CommunityForensics-Small and CommunityForensics-Eval repositories to resolved
revisions.  It scans only Parquet metadata columns first, freezes a deterministic
selection plan, and materialises 24,000 base images. Protocol v2 does not copy or
redownload images: it promotes every row from the former 2,000-image
`test_external_seen_family` cohort into a new training manifest.

```text
train_v2:                        10,000 real + 10,000 AIGI
val_unseen_generator:             1,000 real +  1,000 AIGI
test_external_unseen_generator:   1,000 real +  1,000 AIGI
```

The original manifests remain immutable for lineage. The active training manifest
is `data/manifests/community_forensics_train_v2.csv`; its 20,000 rows contain the
original 18,000-image Small train plus all 1,000 Real and 1,000 AIGI images from
the former seen-family test. Promoted rows retain their source paths and sample
IDs, but their `split`, `project_split`, and training exposure fields are changed
in the derived manifest. File and directory names are never model inputs.

`test_external_seen_family` is retired and forbidden for all future model
evaluation. Its nine exact generators are now train-seen. The remaining strict
unseen-generator test keeps both its 12 exact identities and its Commercial/Other
architecture families disjoint from train-v2.

The builder stores SHA-256, a 64-bit perceptual hash,
source locators, revisions and a complete audit in `data/manifests/`.  SQLite state
under `data/state/` makes metadata scanning and image materialisation resumable.
Exact duplicate hashes discovered after materialisation are repaired before the
final manifest is frozen by deterministic replacement within the same split and
the same generator (or real source).  Replaced files are moved to
`data/quarantine/community_forensics_duplicates/`, and the complete lineage is
recorded in `community_forensics_exact_dedup_repairs.json`.
Cross-split pHash conflicts are repaired similarly, preserving train before
validation and external splits and recording every replacement in
`community_forensics_phash_repairs.json`.
The pinned Small revision reports `real_source=N/A` for every real row, so its
10,000 real images are selected globally and deterministically from the official
train split with `real_source=UNSPECIFIED`; the audit records that the requested
four-source Small balance cannot be verified.  Eval retains its explicit
RAISE/COCO/FFHQ/LAION balance.

```bash
sbatch scripts/slurm/validate_community_forensics.sbatch
sbatch --dependency=afterok:<CHECK_JOB_ID> \
  --export=NONE,CF_PREP_RESTART_COUNT=0 \
  scripts/slurm/prepare_community_forensics.sbatch
```

After data construction and audit succeed, the preparation script starts a
single-job chain for B0, B1, B2, M2 and M3.  Each model has a separate train and
evaluation phase.  Model selection uses `val_unseen_generator`; the clean
validation threshold is then frozen for the diagnostic slices and strict-unseen
test. New training outputs are isolated under
`outputs/community_forensics_v2/<experiment>/`; existing protocol-v1 checkpoints
under `outputs/community_forensics/<experiment>/` are not resumed or overwritten.

Build and audit the derived manifests before training:

```bash
sbatch scripts/slurm/prepare_community_forensics_train_v2.sbatch
```

The job writes `community_forensics_train_v2.csv`, three relabelled hard-slice
manifests, `community_forensics_train_v2_audit.json`, and the atomic
`TRAIN_V2_COMPLETE` marker. It verifies 10,000/10,000 class balance, unique
sample/path/SHA/source locators, all materialized paths and sizes, and continued
strict-unseen generator/family disjointness.

### Community Forensics train-v3 expansion

Train-v3 keeps every train-v2 row and adds 4,000 previously unused images from
the pinned CommunityForensics-Small revision:

```text
GAN AIGI:                    1,000 images / all 12 exact generators
pixel-diffusion AIGI:        1,000 images / all  3 exact generators
GAN-matched real quota:      1,000 images
pixel-diffusion real quota:  1,000 images
train_v3 total:             12,000 real + 12,000 AIGI
```

Within each AIGI family, the deterministic sampler first covers every exact
generator and then keeps per-generator contributions within one image of each
other (GAN 83--84; PixDiff 333--334). The pinned Small metadata reports
`architecture=Real`, `subset=Real`, and `real_source=N/A` for every authentic
image, so no unsupported real-source type is invented. Instead, the two
requested 1,000-image counterpart quotas are drawn from distinct deterministic
Parquet row groups and this limitation is recorded in the v3 audit.

The materializer reads only row groups needed by the chosen images, resumes
from `data/state/community_forensics_train_v3.sqlite3`, and verifies SHA/source
identity disjointness plus pHash distance against every frozen train,
validation, hard and strict-unseen manifest. Build it with:

```bash
sbatch --export=ALL,CF_V3_RESTART_COUNT=0 \
  scripts/slurm/prepare_community_forensics_train_v3.sbatch
```

Successful construction writes:

- `data/manifests/community_forensics_train_v3_additions.csv`;
- `data/manifests/community_forensics_train_v3.csv`;
- `data/manifests/community_forensics_train_v3_selection_plan.csv`;
- `data/manifests/community_forensics_train_v3_audit.json`;
- `data/raw/community_forensics/TRAIN_V3_COMPLETE`.

The independent configs live under `configs/community_forensics_v3/`, and all
new checkpoints/results are isolated under `outputs/community_forensics_v3/`.
After the v3 completion marker exists, start the five-model chain on the
required GPU node `TC2N08`:

```bash
sbatch --export=NONE,CF_V3_EXPERIMENT_INDEX=0,CF_V3_PHASE=train,CF_V3_RESTART_COUNT=0 \
  scripts/slurm/train_and_eval_community_forensics_v3.sbatch
```

For the clean plus 20-perturbation matrix, submit no more than two jobs under
the normal QoS:

```bash
sbatch --export=NONE,CF_V3_MODEL_INDICES=0:1:2 \
  scripts/slurm/evaluate_community_forensics_v3_robustness.sbatch
sbatch --export=NONE,CF_V3_MODEL_INDICES=3:4 \
  scripts/slurm/evaluate_community_forensics_v3_robustness.sbatch
```

The retired external seen-family manifest remains excluded. Checkpoint
selection still uses the internal Small validation split, and its frozen
threshold is reused for the exact-seen, hard-generator and strict-unseen tests.

Train-v3 uses an expanded strict unseen-generator test while preserving the
original 2,000-image manifest as an immutable baseline. The expanded manifest
adds 1,000 Eval AIGI images and 1,000 Eval real images, for 2,000/2,000 class
balance. AIGI additions cover all 12 exact generators available under the
unseen `Commercial` and `Other` families with 83--84 samples each. The pinned
Eval revision contains no additional exact generator identities beyond those
12, so the expansion increases sample support rather than claiming new
generator categories. Real additions are balanced at 250 each across COCO,
FFHQ, LAION and RAISE.

The expansion is constructed only after train-v3 succeeds, so SHA/source-locator
and pHash checks include the new training additions:

```bash
sbatch --dependency=afterok:<TRAIN_V3_DATA_JOB_ID> \
  --export=NONE,CF_V3_EXT_RESTART_COUNT=0 \
  scripts/slurm/prepare_community_forensics_external_unseen_v3.sbatch
```

It writes
`community_forensics_test_external_unseen_generator_v3_expanded.csv`, an
additions-only manifest, a selection plan, a JSON audit, and the atomic
`EXTERNAL_UNSEEN_V3_COMPLETE` marker. V3 training/evaluation scripts require
that marker and use the expanded 4,000-image test; v1/v2 scripts continue to use
their frozen 2,000-image strict-unseen manifest.

If a COCO val2017/DALL-E Advanced reserved-image hash manifest is available, pass
it to the original corpus builder with `--reserved-hash-manifest`. The derived
train-v2 builder reuses that frozen audit. Without such a manifest, the audit
records that official split/source constraints were enforced but reserved-image
hash overlap could not be verified.

### Community Forensics validation-v2 extension

The validation-v2 extension originally added a true exact-seen cohort and three
exact-unseen hard slices without rewriting the frozen base manifests. It pins
[`TheKernel01/AIGIBench`](https://huggingface.co/datasets/TheKernel01/AIGIBench)
at revision `f125eabc5ac34a4729d74adc1aa1214540f91947`.  The dataset card identifies
its `SD14` class as Stable Diffusion 1.4; Small train independently contains the
exact model identity `CompVis/stable-diffusion-v1-4`.  The canonical mapping is
therefore explicit and source-backed rather than inferred from a broad family.

```text
val_external_exact_seen_generator: 1,000 ImageNet real + 1,000 SD14 AIGI
val_hard_hourglass:                   250 shared Eval real + 250 Hourglass AIGI
val_hard_dfgan:                       250 shared Eval real + 250 DFGAN AIGI
val_hard_galip:                       250 shared Eval real + 250 GALIP AIGI
```

The three hard manifests share one fixed 250-image real reference panel so their
generator-specific AUROCs are directly comparable. Their AIGI rows and real
panel were selected from Eval locators unused by the frozen external tests.
SHA-256 and pHash checks forbid image overlap with every frozen base manifest.
Under train-v2, other Hourglass/DFGAN/GALIP images are now part of training, so
the three hard slices are relabelled as **exact-seen hard validation** while
remaining image-disjoint. The original Small `val_unseen_generator` remains the
checkpoint-selection endpoint.

Build the extension before starting the model chain:

```bash
sbatch scripts/slurm/prepare_community_forensics_validation_v2.sbatch
sbatch --dependency=afterok:<VALIDATION_V2_JOB_ID> \
  scripts/slurm/prepare_community_forensics_train_v2.sbatch
```

The job is resumable through
`data/state/community_forensics_validation_v2.sqlite3`, caps new materialized
data at 8 GiB, and writes `VALIDATION_V2_COMPLETE` only after all manifests and
the audit have been atomically finalized.  Seeded selection ranks remote
row-groups before rows so the chosen images remain deterministic without
scattering a small subset across hundreds of redundant Parquet reads.

### Community Forensics data statistics report

The existing self-contained report at
`reports/data_statistics/COMMUNITY_FORENSICS_DATA_STATISTICS.html` is the
protocol-v1 snapshot and remains unchanged as historical evidence. After the
train-v2 manifest is built, the current report job writes the seven-active-
manifest snapshot under `reports/data_statistics/train_v2/`. It distinguishes manifest
references from unique physical images because the three hard slices reuse one
250-image real panel.  The report includes class balance, exact-generator and
architecture-family exposure, real sources, formats, resolution and storage,
all 21 active pairwise split-overlap checks, manifest lineage, and build-audit counts.

Supporting machine-readable outputs are:

- `reports/data_statistics/train_v2/community_forensics_data_statistics.csv`;
- `reports/data_statistics/train_v2/community_forensics_generator_statistics.csv`;
- `reports/data_statistics/train_v2/community_forensics_distribution_statistics.csv`;
- `reports/data_statistics/train_v2/community_forensics_data_statistics_notes.json`;
- `reports/data_statistics/train_v2/community_forensics_data_statistics_artifact.json`.

Regenerate and validate the package on a Compute Node with:

```bash
sbatch scripts/slurm/report_community_forensics_data_statistics.sbatch
```

The existing atlas under `reports/atlases/` is the protocol-v1 snapshot. The
current atlas job writes train-v2 outputs under `reports/atlases/train_v2/` and
contains one deterministically selected AIGI representative for 78 training
generators and all 12 strict-unseen test generators. The compact training sample keeps every
generator from rare architecture families (at most five generators) and fills
the remaining slots proportionally with a fixed hash rank.  Separate train/test
atlases and the complete tile-to-manifest mapping are stored alongside it as
`community_forensics_{train,test}_exact_generators_atlas.jpg` and
`community_forensics_exact_generators_atlas_index.csv`.  Validation manifests
and real images are intentionally excluded from this visualization.

```bash
sbatch scripts/slurm/build_community_forensics_exact_generator_atlas.sbatch
```

## Reproduce one current experiment

Inside an allocated SLURM job, after activating `repostguard`:

```bash
python -m repostguard.train --config configs/community_forensics/b1.yaml
python -m repostguard.evaluate \
  --config configs/community_forensics/b1.yaml \
  --checkpoint outputs/community_forensics_v2/b1/best.pt

python -m repostguard.infer \
  --config configs/community_forensics/m2.yaml \
  --checkpoint outputs/community_forensics_v2/m2/best.pt \
  --input-dir ./images \
  --output ./predictions.json \
  --diagnostics ./diagnostics.json
```

Training checkpoints are atomically replaced and contain model, optimizer,
scheduler, AMP scaler, epoch/global step, best metric, RNG state and the full
resolved config. `SIGUSR1` requests a safe-boundary checkpoint and clean exit.

## Outputs

Evaluation writes:

- `metrics_by_transform.csv`: clean plus the fixed robustness matrix;
- `summary.json`: clean/robust mean/worst metrics and AUROC drop;
- `predictions.jsonl`: per-image probabilities and transform metadata;
- `run_card.json`: config and checkpoint hashes, environment and parameter count.

The threshold is selected from clean validation predictions by maximising
balanced accuracy, then held fixed for every transformed condition.

### Community Forensics robustness matrix v2

`configs/community_forensics_robustness_v2.yaml` preserves the original clean
plus 17 perturbation conditions and appends three stricter composed conditions:

- four-stage platform repost: center crop, resize, Gaussian blur, JPEG;
- four-stage edit repost: color jitter, resize, Gaussian noise, JPEG;
- six-stage random composition: crop, resize, color jitter, blur, noise and
  JPEG, with deterministic per-sample strengths from the full training ranges.

Future train-v2 B0/B1/B2/M2/M3 checkpoints are evaluated on five active
slices: external exact-seen-generator, exact-seen hard Hourglass, exact-seen
hard DFGAN, exact-seen hard GALIP, and strict unseen-generator. The retired
seen-family manifest is not present in the evaluation script.

The normal QoS admits only two submitted jobs, so run two ordinary jobs rather
than a five-task array:

```bash
sbatch --export=NONE,CF_MODEL_INDICES=0:1:2 scripts/slurm/evaluate_community_forensics_robustness_v2.sbatch
sbatch --export=NONE,CF_MODEL_INDICES=3:4 scripts/slurm/evaluate_community_forensics_robustness_v2.sbatch
```

Each model uses the probability threshold selected from the internal
Small clean validation split. The external splits never select their own
threshold. Evaluation loads the immutable `resolved_config.yaml` stored beside
each checkpoint, then applies manifest/matrix/threshold overrides only after the
checkpoint digest has been validated. Outputs are isolated under
`outputs/community_forensics_v2_robustness_v2/`. The existing
`reports/evaluations/robustness_v2/` package remains the protocol-v1 report;
future train-v2 reports are written under
`reports/evaluations/robustness_v2_train_v2/`.

### Unseen-generator detailed accuracy report

The existing package under `reports/evaluations/unseen_generator/` analyzes
protocol-v1 checkpoints. Future train-v2 predictions on the same balanced
2,000-image strict unseen-generator test are written under
`reports/evaluations/unseen_generator_train_v2/`. In addition to
ROC curves and AUROC, the report includes precision-recall curves, Accuracy,
Precision, AIGI Recall, Real Specificity, NPV, F1/Macro-F1, Balanced Accuracy,
MCC, AP, FPR/FNR, Brier, ECE-15, low-FPR TPR, confusion counts, stratified
bootstrap intervals, all 21 clean/perturbed conditions, exact-generator recall,
and real-source specificity. Per-model thresholds remain frozen from the
internal Small clean validation split.

Supporting machine-readable outputs are:

- `reports/evaluations/unseen_generator_train_v2/community_forensics_unseen_generator_clean_metrics.csv`;
- `reports/evaluations/unseen_generator_train_v2/community_forensics_unseen_generator_all_metrics.csv`;
- `reports/evaluations/unseen_generator_train_v2/community_forensics_unseen_generator_slice_metrics.csv`;
- `reports/evaluations/unseen_generator_train_v2/community_forensics_unseen_generator_accuracy_notes.json`;
- `reports/evaluations/unseen_generator_train_v2/community_forensics_unseen_generator_accuracy_artifact.json`.

Regenerate the metrics and portable report on a Compute Node with:

```bash
sbatch scripts/slurm/report_community_forensics_unseen_accuracy.sbatch
```

## Scope and caveats

The forensic filters are a deterministic 30-kernel **SRM-inspired** high-pass
bank, not the proprietary/bit-exact filter implementation from another codebase.
M2 uses DCT energy to select high- and low-frequency patches and shares one
ResNet-18 over RGB, high-pass residual and NPR channels. The reserved COCO
val2017/DALL-E Advanced hash manifest was not supplied, so official source rules
were enforced but reserved-image hash exclusion has not been verified.
