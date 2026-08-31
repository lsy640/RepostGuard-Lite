# M3-primary / M2-secondary mobile Student plan

## Frozen first-version design

- Student: ImageNet-initialized MobileNetV3-Large, one binary logit.
- Teachers: M3 weight `0.7`, M2 weight `0.3`.
- Teacher probabilities: scalar-temperature calibrated on the internal
  `val_unseen_generator` split, then mixed in probability space.
- Loss: `0.5 * hard BCE + 0.4 * teacher KD + 0.1 * clean/aug consistency`.
- Teacher disagreement: when calibrated probability disagreement exceeds
  `0.25`, that sample's KD contribution is multiplied by `0.25`; hard-label
  supervision remains unchanged.
- Cached views: clean, JPEG Q50, resize 0.5 + JPEG Q70, and deterministic
  strict six-transform composition.
- Checkpoint selection: internal clean validation AUROC only. External slices
  are diagnostic and cannot select a checkpoint or threshold.

## Reproducibility gates

1. Validate the dataset marker, all manifest paths, and the train-manifest
   SHA-256.
2. Validate M2/M3 checkpoint/config digests. Cache shards record both teacher
   checkpoint SHA-256 hashes, manifest SHA-256, transforms, and runtime lineage.
3. Fit teacher scalar temperatures on the internal validation cache, never on
   the Student training labels.
4. Generate a fully resolved frozen Student YAML containing the temperatures
   and teacher hashes. Training checkpoints record the resulting config digest.
5. Refuse caches with incomplete sample coverage, wrong image hashes, mixed
   teacher lineage, or view mismatches.

## TC2 execution order

The account's `normal` QoS permits at most one GPU per job, two simultaneous
jobs, and six hours per job. Every long training job requests 5 h 50 min and
receives `SIGUSR1` five minutes before the limit, so it can atomically write a
resumable checkpoint. It then asks Slurm to requeue the same job; if requeue is
disabled by the cluster, it exits with code 75 and requires manual resubmission.

1. Create an isolated environment and run the full unit test suite plus
   `bash -n` on all new SLURM scripts.
2. Transfer Community Forensics data, then verify the byte size and SHA-256 of
   every unique image referenced by all training and evaluation manifests.
3. Prefetch the shared OpenCLIP and MobileNetV3 pretrained backbones once, then
   if the published M2/M3 checkpoint files are unavailable, reproduce them in
   parallel with `train_distillation_teachers.sbatch`, then evaluate them before
   accepting them as teachers.
4. Submit `cache_student_teachers.sbatch` as a two-task array for train logits.
5. Run `cache_and_calibrate_student_teachers.sbatch` to cache the internal
   validation split, fit temperatures, and create
   `student_mnv3_dual_teacher_frozen.yaml`.
6. Run a small end-to-end smoke job.
7. Submit `train_student_distill.sbatch`; resubmit the same job after a verified
   safe stop until `DONE` exists.
8. Run `evaluate_student_distill.sbatch` on internal, exact-seen, hard-generator,
   seen-family, and unseen-generator robustness slices.
9. Export the accepted checkpoint, measure model size and latency on the target
   Android/iOS devices, then choose FP16 or INT8 deployment from measured
accuracy/latency rather than desktop estimates.

After code, environment, and data transfer are verified, the dependency DAG is
submitted with `bash scripts/submit_student_pipeline.sh`. Its immutable Slurm
job IDs are recorded under the Student output directory.

The reproducible FP32 export command is:

```bash
python scripts/export_student_mobile.py \
  --config configs/community_forensics/student_mnv3_dual_teacher_frozen.yaml \
  --checkpoint outputs/community_forensics/student_mnv3_dual_teacher/best.pt \
  --output-directory outputs/community_forensics/student_mnv3_dual_teacher/mobile \
  --format both
```

## Acceptance gate

The first mobile Student is accepted only if it preserves the competition
split's frozen operating-point quality within the team-agreed tolerance and
does not introduce a material regression on strict unseen-generator or worst
repost perturbations. App size and phone latency are reported only from an
exported artifact on representative devices.
