# Student V3.2 full-data 20-epoch refit plan

## Objective

Train one final competition-oriented Student V3.2 experiment from scratch on the
full 24,000-row V3 training manifest, including the 2,004 samples from the 19
families previously reserved for family-unseen development. Preserve the current
family-unseen-selected epoch-3 model as the rollback and evidence model.

This is one 20-epoch experiment split across resumable SLURM stages to respect
TC2's per-job time limit. It is not an epoch-20 continuation of the current
family-holdout experiment.

## Frozen rollback model

- Remote snapshot: `outputs/releases/v32_corrected_family_unseen_epoch3`
- Checkpoint: `best.pt` (epoch 3, global step 1545)
- Checkpoint SHA-256:
  `05edd57825cf28608d26fe77db92f364f37dcf7f070d757e82f85b9e2711cfea`
- Keep the original output directory and the snapshot read-only by convention;
  the full refit must use a new output directory.

## Fixed experiment definition

Create a new config, `configs/community_forensics_v3/student_v32_full_refit_e20.yaml`,
that inherits the corrected V3.2 method and changes only the data/output/runtime
fields required for the full refit:

- train manifest: `data/manifests/community_forensics_train_v3.csv`
- expected train rows: 24,000
- expected manifest SHA-256:
  `fc0a7ab732faeb604ed1e77281fada715d7cffb353974a4985820548d871d9d6`
- output directory:
  `outputs/community_forensics_v32_full_refit_e20/student_corrected`
- teacher cache directory:
  `data/cache/teacher_logits/community_forensics_m3_v32_full_refit_e20`
- epochs: 20
- seed, architecture, pretrained weights, augmentations, sampling, affine teacher
  calibration, T=1 losses, reliability weighting and feature/forensic distillation:
  unchanged from `student_v32_corrected.yaml`
- validation: a frozen, train-disjoint, non-protected validation manifest. Verify
  sample/path disjointness before training. Do not use the 19-family dev or the
  protected expanded external 4k for checkpoint selection.

The full refit starts with random Student initialization plus the same declared
pretrained backbone. It must not load the current `best.pt` or `latest.pt`.

## Stage 1: full teacher cache

Rebuild the M3 logits, semantic/forensic/fused features and quality-gate targets
for all 24,000 training rows. Do not retrain M3 and do not overwrite the current
21,996-row cache.

Gate before training:

- cache sample count is exactly 24,000;
- manifest SHA-256 matches the full manifest;
- four configured views are present;
- preprocessing and teacher checkpoint digests match V3.2;
- logit, feature and gate shapes are valid;
- sample IDs are unique and source bounds are valid.

Expected L40S time is roughly 5-10 minutes based on the prior 21,996-row prepare
job. Treat this as an estimate, not a guarantee.

## Stage 2: formal epoch-1 gate

Start the final experiment itself with 1 GPU, 10 CPU and 30 GB, using
`STOP_AFTER_EPOCH=1`. Epoch 1 is the smoke gate and writes directly to the final
output directory; no separate throwaway training run is used. Confirm finite
losses, deterministic startup, checkpoint lineage, sampling proportions and a
successful resumable checkpoint. No mobile export and no protected external
evaluation are part of this stage.

The cache was already fully validated by its producing job. The training script
only checks that the frozen cache artifact exists; `CachedDistillationDataset`
performs its own manifest, lineage, shape and coverage checks while loading it,
so the standalone cache validator is not redundantly executed before every
training stage.

## Stage 3: epochs 2-10

After the epoch-1 gate passes, resume the same experiment with
`STOP_AFTER_EPOCH=10`. Do not clear the final output directory or reset any
training state.

Gate at epoch 10:

- job is COMPLETED with ExitCode 0;
- `latest.pt` reports epoch 10 and retains optimizer/scheduler/RNG state;
- `best.pt` exists and records its selected validation epoch;
- all losses and validation metrics are finite;
- class/architecture/generator sampling remains within the predetermined
  tolerances;
- no DONE marker is required yet.

## Stage 4: epochs 11-20

Resume the same experiment from the same `latest.pt` in a second L40S job. Do
not clear the output directory and do not reset the optimizer, scheduler,
global step or best-metric state.

The epoch-1 gate, epochs 2-10 continuation and epochs 11-20 continuation
constitute one 20-epoch training run. Splitting them avoids wasted smoke work and
keeps every job comfortably below TC2's six-hour limit. Based on the previous
21,996-row run, the total training estimate is about 5-6 hours.

Final gate:

- job is COMPLETED with ExitCode 0;
- `latest.pt` reports epoch 20;
- DONE exists;
- `best.pt` and `latest.pt` both pass config/checkpoint lineage validation;
- every epoch and checkpoint metric is finite.

## Model selection and evaluation policy

- `best.pt` is updated by the frozen, train-disjoint, non-protected validation
  metric; it may come from any epoch 1-20.
- Always retain epoch-20 `latest.pt`, but do not automatically deploy it if
  `best.pt` comes from an earlier epoch.
- Do not use the former 19-family dev as unseen evidence after adding those
  families to training.
- Do not use the protected expanded external 4k to choose an epoch, alter the
  config or decide whether to continue training.
- The already frozen epoch-3 model remains the rollback model and the valid
  family-unseen evidence model.
- A later external-4k comparison of the refit would be a second look at that
  test set and must be reported as such; it is not part of training or model
  selection.

## Stop and rollback rules

Stop without overwriting the frozen model if cache validation fails, validation
overlaps training, checkpoint lineage changes, losses become non-finite, resume
state is incomplete, or the job fails for a non-timeout reason. A safe timeout
may requeue/resume from `latest.pt`; all other failures require diagnosis before
retrying the failed stage.
