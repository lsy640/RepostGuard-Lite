# Student V3.2 corrected distillation

## Why this is a separate version

V3.1 baseline and T=1 remain diagnostic runs. They share two implementation
problems and must not be treated as final candidates:

1. The old `WeightedRandomSampler` gives every
   `(label, source_dataset, generator_id)` group equal total mass. The V3.1
   train manifest has 902 AIGI groups but only 2 Real groups. Its expected
   distribution is therefore 99.7788% AIGI and 0.2212% Real; roughly 75.3% of
   batch-128 batches contain no Real image.
2. Teacher calibration was `sigmoid(raw_logit / temperature)` with no
   intercept. On the dedicated 2,000-image calibration cache, 99.85-100% of
   M3 targets remain above 0.5. Temperature cannot move the binary decision
   boundary.

V3.2 fixes these before adding more feature capacity.

## Corrected training distribution

Target probability mass:

```text
Real       50%
AIGI       50%
  LatDiff  25% of the whole batch (50% within AIGI)
  GAN      12.5% of the whole batch (25% within AIGI)
  PixDiff  12.5% of the whole batch (25% within AIGI)
```

Within architecture `a`, generator `g` receives mass proportional to
`N(a,g)^0.5`; each row then receives its generator mass divided by `N(a,g)`.
This is between row-uniform and generator-uniform sampling and avoids extreme
duplication of the two PixDiff groups.

The Real half is also hierarchical. Its two sources contain 10,082 and 916
rows, so source mass is proportional to `N(source)^0.5` (about 76.8% and
23.2% within Real) instead of either the row-uniform 91.7%/8.3% split or an
over-aggressive 50%/50% source split. `source_dataset` is never a flat group
competing against hundreds of AIGI generators.

The loader prints the exact expected distribution and a SHA-256 of the weight
vector. Training prints actual per-epoch Real/LatDiff/GAN/PixDiff counts.

## Corrected teacher targets

For every teacher view independently:

```text
z_calibrated(view) = a_view * z_raw + b_view, a_view > 0
p_teacher = sigmoid(z_calibrated)
```

The dedicated calibration artifact passes the following M3 gates:

| View | NLL before | NLL after | BA at 0.5 | Positive rate | Real mean p | AIGI mean p |
|---|---:|---:|---:|---:|---:|---:|
| clean | 2.3873 | 0.4654 | 0.7795 | 0.5175 | 0.3029 | 0.6973 |
| jpeg50 | 2.5812 | 0.4439 | 0.7920 | 0.5220 | 0.2857 | 0.7145 |
| resize50+jpeg70 | 2.3644 | 0.4574 | 0.7780 | 0.5160 | 0.2969 | 0.7034 |
| strict6 | 2.5144 | 0.5203 | 0.7495 | 0.5195 | 0.3462 | 0.6539 |

M3 remains the only logit teacher. Each logit KD sample is weighted by its
calibrated confidence. A teacher prediction on the wrong side of ground truth
receives only 10% of that confidence weight; ground-truth BCE is never reduced.
The reported soft loss is Bernoulli KL (soft BCE minus teacher entropy), so it
does not contain the old large constant.

## Representation transfer

The deployable Student contains 7,955,038 parameters and keeps the two-branch
MobileNetV3-Large + EfficientNet-B0 design. Using the distilled semantic
projection in fusion removes 361,856 parameters (4.35%) from the earlier
8,316,894-parameter V3.2 draft. V3.2 adds only:

- three NPR channels, `RGB - nearest_down_up(RGB)`, to the existing forensic
  adapter; there is no third backbone;
- a small quality-aware semantic/forensic gate;
- direct 256-D semantic/forensic fusion, so the distilled semantic projection
  and the quality gate both sit on the deployed classification path;
- gate-fraction distillation from M3;
- pointwise semantic/forensic/fused feature distillation;
- batch relational distillation on forensic and fused 256-D features;
- teacher/Student cross-view feature-delta alignment.

Relational and cross-view losses exist only during training. The corrected
random-weight graph passes TorchScript and ONNX opset-18 export plus ONNX
Runtime parity on zero, one, and random batch inputs. Its FP32 artifacts are
31,070,503 bytes (TorchScript) and 31,167,485 bytes (ONNX) before quantization.

The method follows the motivation of
[Relational Knowledge Distillation](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html),
[Contrastive Representation Distillation](https://openreview.net/references/pdf?id=Ltki-ab4x),
and the local residual used by
[NPR](https://openaccess.thecvf.com/content/CVPR2024/html/Tan_Rethinking_the_Up-Sampling_Operations_in_CNN-based_Generative_Network_for_Generalizable_CVPR_2024_paper.html).
Teacher-score calibration matters particularly for robust/worst-group
distillation; see
[Robust distillation for worst-class performance](https://proceedings.mlr.press/v216/wang23e.html).

## Loss and schedule

```text
hard ground-truth BCE       0.50
confidence-gated logit KD   0.15
clean/degraded consistency  0.05
feature bundle              0.30
```

Inside the feature bundle:

```text
pointwise branch features   0.65
relational/cross-view       0.25
quality gate                0.10
```

All distillation terms ramp over the first three epochs. Hard BCE remains
active at full weight from the first batch.

## Resource-safe execution order

Every GPU job is pinned to TC2N03 (L40S), requests one GPU, four CPU, and 6 GB
host memory, and uses no tmux.

1. `prepare_student_v32.sbatch`: validate affine calibration and add only M3
   gate fractions to the existing feature cache. It does not rerun the large
   teacher branches.
2. `smoke_student_v32.sbatch`: two epochs, then TorchScript/ONNX export and
   ONNX Runtime parity.
3. `train_student_v32.sbatch` with `STOP_AFTER_EPOCH=10`: same final config and
   checkpoint lineage, clean stop at epoch 10 without writing `DONE`.
4. `evaluate_student_v32_dev.sbatch`: evaluate the protected Student-train
   holdout and compute pooled plus architecture/hierarchical-generator metrics.
   It compares against the already-completed V3.1 T=3 baseline artifacts; it
   does not retrain the baseline and excludes the known-biased T=1 diagnostic.
5. Only if the epoch-10 gate passes, rerun `train_student_v32.sbatch` without
   `STOP_AFTER_EPOCH`; it resumes the same checkpoint to epoch 20.
6. Run the external strict 4k test once, only after the internal winner is
   frozen.

## Winner criteria

Checkpoint selection must not use pooled AUROC alone:

1. pooled clean AUROC and architecture-macro clean AUROC may regress by at
   most 0.01;
2. primary: robust architecture-macro AUROC;
3. secondary: worst architecture × transform AUROC;
4. tie-break: hierarchical generator macro, then pooled robust mean.

The current holdout is a Student-direct generator holdout. M3 saw these data
during teacher training, so reports must not call it fully teacher-unseen. The
external 4k set remains the final strict unseen test.
