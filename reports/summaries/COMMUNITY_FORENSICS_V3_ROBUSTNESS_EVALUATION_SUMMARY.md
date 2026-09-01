# Community Forensics train-v3 Robustness Evaluation Summary

> **English** | [中文](COMMUNITY_FORENSICS_V3_ROBUSTNESS_EVALUATION_SUMMARY_Chinese.md)

## Conclusions

On the complete 4,000-image strict unseen-generator test set, **M2 is currently the most balanced model in terms of robustness**: its Clean AUROC is 0.9308, its equally weighted mean AUROC across the 20 transformed conditions is 0.9163, and its worst-condition AUROC remains 0.8525. At the frozen threshold, its Clean Accuracy is 85.78%, while its mean transformed Accuracy is 83.70%. M3 performs nearly identically to M2 and exceeds it by 0.11 percentage points in mean transformed Accuracy; with only a single training seed, this difference should not be interpreted as a statistical advantage.

Relative to the architecturally identical B0, B1 substantially narrows the performance gap between Clean and transformed inputs, indicating that class-symmetric augmentation improves robustness to composite processing. B2 does not decline in mean transformed AUROC, yet its Accuracy/BA decreases markedly, suggesting that its principal limitation is transfer of the frozen threshold rather than a wholesale loss of ranking ability.

## Evaluation Protocol

| Item | Frozen definition |
|---|---|
| Training version | Community Forensics train-v3; 24,000 images, comprising 12,000 Real and 12,000 AIGI images |
| Test role | External strict unseen-generator test; not used for checkpoint, threshold, or model selection |
| Test size | 4,000 images, comprising 2,000 Real and 2,000 AIGI images |
| AIGI coverage | 12 exact generators unseen during training; the Commercial and Other supercategories are likewise absent from training |
| Real-image coverage | COCO, FFHQ, LAION, and RAISE, with 500 images from each source |
| Conditions | 1 Clean condition + 17 original perturbations + 2 four-stage compositions + 1 six-stage random composition, for 21 conditions in total |
| Checkpoint | Selected for each model exclusively by Clean AUROC on the internal validation set |
| Threshold | Frozen for each model exclusively by Clean balanced accuracy on the internal validation set |
| Aggregation | The transformed mean is an equally weighted average over the 20 non-Clean conditions; every condition uses the same 4,000 images |

Test manifest SHA256: `59ca2e4ca966dac9fa4fb55281153f93e5becdd3e25da83bc2dff3fad36126cd`

Perturbation matrix SHA256: `69531f3f7111651808c99f14f89723bf631345878b1cbd0cbe0eee8531dde83c`

Official evaluation job: `32885`; its terminal state was recorded as complete.

## Compact Comparison of Clean and Transformed Performance

Because the test set is exactly class-balanced, Accuracy and balanced accuracy are numerically identical. `Δ` is defined as the transformed mean minus the Clean result; a negative value denotes degradation.

| Model | Clean AUROC | Mean AUROC over 20 transformed conditions | Δ AUROC | Worst transformed AUROC | Clean Acc./BA | Mean transformed Acc./BA |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.8125 | 0.7505 | -0.0620 | 0.4846 | 74.10% | 68.85% |
| B1 | 0.8117 | 0.7850 | -0.0267 | 0.6665 | 74.33% | 71.48% |
| B2 | 0.7707 | 0.7722 | +0.0015 | 0.6743 | 69.80% | 59.83% |
| **M2** | **0.9308** | **0.9163** | **-0.0145** | **0.8525** | **85.78%** | 83.70% |
| M3 | 0.9305 | 0.9154 | -0.0152 | 0.8489 | 85.35% | **83.81%** |

![Comparison of AUROC and Accuracy on Clean and transformed images](../evaluations/community_forensics_v3_evaluation/robustness_clean_vs_transformed.svg)

## AUROC by Perturbation Group

| Model | Clean | Mean over the original 17 perturbations | 4-stage A | 4-stage B | 6-stage | Mean over all 20 transformed conditions |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.8125 | 0.7582 | 0.7447 | 0.7173 | 0.6597 | 0.7505 |
| B1 | 0.8117 | 0.7874 | 0.7872 | 0.7828 | 0.7435 | 0.7850 |
| B2 | 0.7707 | 0.7826 | 0.7743 | 0.6910 | 0.6743 | 0.7722 |
| **M2** | **0.9308** | **0.9218** | **0.9153** | 0.8877 | **0.8525** | **0.9163** |
| M3 | 0.9305 | 0.9210 | 0.9132 | **0.8888** | 0.8489 | 0.9154 |

The original 17 perturbations comprise:

- JPEG compression at Q90, Q70, Q50, and Q30;
- Gaussian blur with σ=0.5, 1.0, and 2.0;
- resizing via bicubic interpolation at scale 0.5 and bilinear interpolation at scale 0.25;
- Gaussian noise with σ=0.02, 0.05, and 0.1;
- color jitter at 0.8/0.8/0.8 and 1.2/1.2/1.2;
- center cropping at ratio 0.8; and
- two-stage compositions: resize 0.5 + JPEG Q70, and crop 0.8 + JPEG Q50.

The three additional stringent conditions are:

- 4-stage A platform repost: crop 0.85 → bicubic resize 0.5 → blur σ=1.0 → JPEG Q50;
- 4-stage B edit repost: color jitter 1.15/1.15/0.85 → bilinear resize 0.5 → noise σ=0.05 → JPEG Q50; and
- 6-stage random composition: all six perturbation categories are applied, with the intensity of each independently sampled from its training range.

## Model-Level Interpretation

### B0 and B1

B0 attains a Clean AUROC of 0.8125, but its transformed mean falls to 0.7505, and its worst result—under Gaussian noise with σ=0.1—is only 0.4846. B1 adds no parameters and changes only the training augmentation. Its transformed mean rises to 0.7850, its worst AUROC improves to 0.6665, and its six-stage AUROC increases from 0.6597 to 0.7435. The B1/B0 comparison therefore provides a direct ablation demonstrating that augmentation improves robustness to composite perturbations.

### B2

B2's mean transformed AUROC is 0.0015 higher than its Clean AUROC, whereas its mean transformed Accuracy/BA falls from 69.80% to 59.83%. This does not mean that perturbations improve deployment performance; instead, the score distribution shifts relative to the frozen threshold. AUROC-based ranking performance and performance at a fixed operating point must therefore be reported jointly.

### M2 and M3

M2 and M3 clearly outperform all three baselines on Clean inputs, the original 17 perturbations, and the additional composite conditions. M2 achieves the highest Clean, transformed-mean, 4-stage A, six-stage, and worst-condition AUROC; M3 is marginally higher on 4-stage B AUROC and mean transformed Accuracy/BA. Because these differences are small and only one training seed is available, the current deployment choice prioritizes M2, while M3 remains a candidate gating strategy whose value may depend on dataset scale. Neither model should be claimed to be statistically superior to the other.

## Principal Failure Modes and Deployment Implications

1. **Sequential processing pipelines remain the most challenging.** The six-stage composition is the worst condition for M2 and M3, while strong Gaussian noise is the worst condition for B0 and B1.
2. **AUROC cannot substitute for threshold transfer analysis.** B2 provides the clearest example; deployment requires threshold calibration on an independent target-domain dataset.
3. **Low-FPR tail performance remains inadequate.** On the current full-unseen Clean set, M2 and M3 both have TPR@1%FPR = 0. Despite high aggregate AUROC, they therefore do not satisfy stringent low-false-positive requirements.
4. **The preferred model depends on the objective.** M2 is preferred when jointly considering strict-unseen ranking and performance at the frozen threshold; B2 remains stronger for generator-level ranking on the difficult Hourglass, DFGAN, and GALIP slices.

## Scope of Evidence

- The results are based on a single training seed; small inter-model differences do not establish statistical significance.
- The transformed mean equally weights 20 predefined conditions and does not represent the probability distribution of processing pipelines on future real-world platforms.
- The test prior is 50% AIGI; Accuracy, Precision, and NPV do not transfer directly to real-platform class priors.
- The strict-unseen set covers only 12 generators and four real-image sources and is not representative of all future generators or camera domains.
- All thresholds were obtained from the internal Clean validation set; external test labels were not used to select checkpoints or thresholds.

## Traceable Artifacts

The official structured results for each model are located at:

```text
outputs/community_forensics_v3_robustness_v2/<model>/unseen_generator_expanded/
├── COMPLETE
├── metrics_by_transform.csv
├── predictions.jsonl
├── run_card.json
└── summary.json
```

For the more comprehensive five-slice analysis, see [`COMMUNITY_FORENSICS_V3_EVALUATION_REPORT.md`](COMMUNITY_FORENSICS_V3_EVALUATION_REPORT.md).
