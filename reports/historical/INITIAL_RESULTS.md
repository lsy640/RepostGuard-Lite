# CIFAKE pilot: B0, B1, B2 and M2

Run date: 2026-08-27 (Asia/Singapore)  
Compute node: NVIDIA A40, one GPU per job  
Training: 10,000 CIFAKE official-train images, balanced 5,000/5,000  
Validation: 2,000 CIFAKE official-test images, balanced 1,000/1,000  
Epochs: 3  
Evaluation: clean plus 17 fixed single/composed degradation conditions

## Data audit

- All selected samples are 32x32 JPEG images.
- The deterministic builder skipped 9 duplicate candidates inside train.
- It skipped 4 test/cross-split duplicate candidates after fixing the train set.
- Final exact train/test SHA-256 overlap is zero.
- COCO val2017 and DALL-E Advanced reserved sets were not used.

## Results

| Experiment | Clean AUROC | Clean bal. acc. | Robust mean AUROC | Robust mean bal. acc. | Worst AUROC |
|---|---:|---:|---:|---:|---:|
| B0 | 0.9945 | 0.9705 | 0.8638 | 0.7448 | 0.5442 |
| B1 | 0.9929 | 0.9570 | **0.9730** | **0.9043** | **0.9120** |
| B2 | 0.9738 | 0.9220 | 0.9202 | 0.7693 | 0.7688 |
| M2 | 0.9904 | 0.9545 | 0.9641 | 0.8986 | 0.8971 |

The threshold for each model was selected once on clean validation by maximum
balanced accuracy, then held fixed for every transformed condition.

## Initial interpretation

1. B0 confirms the shortcut problem: excellent clean performance coexists with
   a 0.1307 AUROC drop and near-chance performance under blur sigma 2.
2. B1 is the strongest CIFAKE pilot. Class-symmetric augmentation improves
   robust mean AUROC by 0.1092 and robust mean balanced accuracy by 0.1594 over
   B0 while clean AUROC changes by only -0.0017.
3. B2's frozen CLIP semantics are useful but not naturally robust to these
   severe transformations of native 32x32 images.
4. M2 improves over B2 by 0.0439 robust mean AUROC, 0.1293 robust mean balanced
   accuracy and 0.1282 worst-case AUROC. This supports the combined forensic,
   augmentation and paired-consistency design as a repair to the semantic-only
   baseline.
5. M2 remains 0.0089 behind B1 in robust mean AUROC on CIFAKE. Without the M1
   ablation, this run cannot isolate how much of M2's gain comes from the
   forensic branch versus paired consistency.

## Scope

These values are pipeline-validation results, not in-the-wild generalisation
claims. CIFAKE has only 32x32 content and one fake generator (Stable Diffusion
1.4). The next decision-quality experiment should add source/generator-held-out
SID-Set or WildFake data and include M1 so paired consistency has a matched
architecture control.

Machine-readable aggregate metrics are in `pilot_comparison.csv`; full
per-condition metrics, predictions, checkpoint hashes and run cards are under
`../outputs/<experiment>/`.

