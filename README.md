# RepostGuard-Lite pilot

This repository implements the initial experiments defined in
`../AIGC图像鲁棒性检测项目可行性方案.md`:

- **B0**: pretrained EfficientNet-B0, clean-only training.
- **B1**: the same EfficientNet-B0 with class-symmetric JPEG, blur, resize,
  Gaussian-noise, colour-jitter, and centre-crop augmentation.
- **B2**: frozen OpenCLIP image tower with a trainable linear detector.
- **M2**: frozen OpenCLIP plus a trainable DCT-ranked, SRM-inspired/NPR
  ResNet-18 forensic branch and paired clean/degraded consistency losses.
- **M3**: M2 plus a label-agnostic quality-aware gate that dynamically weights
  semantic and forensic features using blur, blockiness, noise, effective
  resolution and dynamic-range proxies.

The completed initial run uses a deterministic 10,000-image CIFAKE training
subset and a 2,000-image official-test validation subset (balanced by class).
CIFAKE remains a pipeline/pilot dataset only. A second, higher-resolution
SID-Set track uses 10,000 real + 10,000 fully synthetic official-training
images and 2,000 real + 2,000 fully synthetic official-validation images.
SID-Set's tampered class is deliberately excluded from this binary whole-image
generation task.

## Data contract

Manifests are CSV files with these required columns:

```text
sample_id,path,label,split,source_dataset,generator_id
```

`path` is relative to `data.root` in each experiment config. `label=1` means
AIGC/FAKE and `label=0` means authentic/REAL. Dataset code never uses file or
directory names as model inputs.

## Cluster workflow

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

## Prepare and run SID-Set

The public SID-Set repository is about 140 GB, so this project streams and
materialises only the requested 24,000 images. The downloader records the
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
`data/manifests/sidset_subset_audit.json`. The original CIFAKE configs and
outputs are preserved; SID-Set uses `configs/sidset/` and
`outputs/sidset/<experiment>/`.

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

## Community Forensics 24k track

The frozen primary track follows
`../Community_Forensics训练与测试集构建方案.md` and pins the public
CommunityForensics-Small and CommunityForensics-Eval repositories to resolved
revisions.  It scans only Parquet metadata columns first, freezes a deterministic
selection plan, and then materialises the selected 24,000 images:

```text
train:                           9,000 real + 9,000 AIGI
val_unseen_generator:            1,000 real + 1,000 AIGI
test_external_seen_family:       1,000 real + 1,000 AIGI
test_external_unseen_generator:  1,000 real + 1,000 AIGI
```

The Small AIGI split uses equal generator contributions and disjoint train/validation
generator identities.  In the pinned dataset revisions, Small and Eval have no
eligible exact generator intersection.  The two frozen Eval tests are retained
with strict, non-overlapping meanings:

- `test_external_seen_family`: the architecture family occurs in Small train,
  but every exact Eval generator identity is absent from Small train and validation;
- `test_external_unseen_generator`: both the exact generator identity and its
  architecture family are absent from Small train and validation.  A same-family
  fallback is forbidden.

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
validation threshold is then frozen for the external seen-family/strict-unseen clean tests.
Training outputs are written under `outputs/community_forensics/<experiment>/`.

If a COCO val2017/DALL-E Advanced reserved-image hash manifest is available, pass
it to the builder with `--reserved-hash-manifest`.  Without it, the audit records
that official split/source constraints were enforced but reserved-image hash
overlap could not be verified.

### Community Forensics validation-v2 extension

The validation-v2 extension adds a true exact-seen cohort without replacing or
rewriting either frozen Eval test.  It pins
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
generator-specific AUROCs are directly comparable.  Their AIGI rows and real
panel are selected only from Eval locators unused by the frozen external tests.
SHA-256 and pHash checks also forbid overlap with every frozen base manifest.
These are diagnostic validation slices; the original Small
`val_unseen_generator` remains the checkpoint-selection endpoint.

Build the extension before starting the model chain:

```bash
sbatch scripts/slurm/prepare_community_forensics_validation_v2.sbatch
```

The job is resumable through
`data/state/community_forensics_validation_v2.sqlite3`, caps new materialized
data at 8 GiB, and writes `VALIDATION_V2_COMPLETE` only after all manifests and
the audit have been atomically finalized.  Seeded selection ranks remote
row-groups before rows so the chosen images remain deterministic without
scattering a small subset across hundreds of redundant Parquet reads.

### Community Forensics data statistics report

The current training split, checkpoint-selection validation split, four
external/hard validation slices, and two external test splits are summarized in
the self-contained report
`reports/COMMUNITY_FORENSICS_DATA_STATISTICS.html`.  It distinguishes manifest
references from unique physical images because the three hard slices reuse one
250-image real panel.  The report includes class balance, exact-generator and
architecture-family exposure, real sources, formats, resolution and storage,
all 28 pairwise split-overlap checks, manifest lineage, and build-audit counts.

Supporting machine-readable outputs are:

- `reports/community_forensics_data_statistics.csv`: one row per split;
- `reports/community_forensics_generator_statistics.csv`: complete exact-generator inventory;
- `reports/community_forensics_distribution_statistics.csv`: class, architecture, format, real-source, and resolution distributions;
- `reports/community_forensics_data_statistics_notes.json`: report contract, audit details, and SQL/chart lineage;
- `reports/community_forensics_data_statistics_artifact.json`: canonical portable-report artifact.

Regenerate and validate the package on a Compute Node with:

```bash
sbatch scripts/slurm/report_community_forensics_data_statistics.sbatch
```

An annotated exact-generator sample atlas is available at
`reports/community_forensics_exact_generators_atlas.jpg`.  It contains one
deterministically selected AIGI representative for 69 training generators and
all 21 external-test generators.  The compact training sample keeps every
generator from rare architecture families (at most five generators) and fills
the remaining slots proportionally with a fixed hash rank.  Separate train/test
atlases and the complete tile-to-manifest mapping are stored alongside it as
`community_forensics_{train,test}_exact_generators_atlas.jpg` and
`community_forensics_exact_generators_atlas_index.csv`.  Validation manifests
and real images are intentionally excluded from this visualization.

```bash
sbatch scripts/slurm/build_community_forensics_exact_generator_atlas.sbatch
```

## Reproduce one experiment

Inside an allocated SLURM job, after activating `repostguard`:

```bash
python -m repostguard.train --config configs/b1.yaml
python -m repostguard.evaluate \
  --config configs/b1.yaml \
  --checkpoint outputs/b1/best.pt

python -m repostguard.infer \
  --config configs/m2.yaml \
  --checkpoint outputs/m2/best.pt \
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

The frozen B0/B1/B2/M2/M3 checkpoints are evaluated on exact-seen-generator,
Hourglass, DFGAN, GALIP, seen-family and strict unseen-generator splits by:

The normal QoS admits only two submitted jobs, so run two ordinary jobs rather
than a five-task array:

```bash
sbatch --export=NONE,CF_MODEL_INDICES=0:1:2 scripts/slurm/evaluate_community_forensics_robustness_v2.sbatch
sbatch --export=NONE,CF_MODEL_INDICES=3:4 scripts/slurm/evaluate_community_forensics_robustness_v2.sbatch
```

Each model uses the probability threshold previously selected from the internal
Small clean validation split. The external splits never select their own
threshold. Evaluation loads the immutable `resolved_config.yaml` stored beside
each checkpoint, then applies manifest/matrix/threshold overrides only after the
checkpoint digest has been validated. The report job writes a self-contained,
validated HTML report at
`reports/COMMUNITY_FORENSICS_B0_B1_B2_M2_M3_ROBUSTNESS_V2.html`, plus the
canonical report artifact JSON, a complete machine-readable CSV, audit notes,
and a portable-delivery verification receipt.

## Scope and caveats

The forensic filters are a deterministic 30-kernel **SRM-inspired** high-pass
bank, not the proprietary/bit-exact filter implementation from another codebase.
M2 uses DCT energy to select high- and low-frequency patches and shares one
ResNet-18 over RGB, high-pass residual and NPR channels. The COCO val2017 and
DALL-E Advanced reserved demonstration sets are not downloaded or referenced by
this pilot.
