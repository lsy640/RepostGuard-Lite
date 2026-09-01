# RepostGuard-Lite — Project Story

> **Detect AI-generated images—even after real-world reposting and edits.**

## 1. Written Project Description

AI-generated images are often resized, cropped, recompressed, blurred, recolored, or repeatedly re-encoded before users see them. Meanwhile, new generators appear faster than detectors can be retrained. RepostGuard-Lite addresses both problems: transformation robustness and generalization to generators absent from training.

The primary M2 model combines a frozen OpenCLIP semantic representation with a forensic branch that analyzes RGB content, DCT frequency components, SRM residuals, and NPR-inspired residuals. It is trained on paired clean and degraded images with class-symmetric augmentation and consistency losses, encouraging stable predictions after JPEG compression, resizing, cropping, blur, noise, color changes, and multistage reposting pipelines.

The project evaluates generalization on a strict unseen-generator test rather than a random image holdout. The main test contains 4,000 images—2,000 Real and 2,000 AIGI—from 12 generators and two generator supercategories absent from training. M2 is also connected to a local FastAPI service and Vue interface for single-image inference, evidence inspection, robustness testing, and batch analysis. MobileNetV3-based Student models explore fully local mobile inference; the Android implementation remains a prototype rather than a finished product.

### Development stack

- **Development tools:** Git and GitHub, a Python 3.11 virtual environment, command-line workflows, pytest, Vitest, and Vite. The demo runs entirely on the local machine: Vue 3 provides the interface, FastAPI/Uvicorn serves the API, and PyTorch executes M2/M3. Local CPU inference is supported on Windows, Linux, and macOS, with optional CUDA acceleration on compatible NVIDIA-equipped Windows and Linux systems. Uploaded images never leave the computer.
- **Models and APIs:** EfficientNet-B0 baselines (B0/B1), OpenCLIP ViT-B/32 baseline (B2), semantic–forensic M2, quality-gated M3, and MobileNetV3 Student models. Pinned M2/M3 weights are released through Hugging Face Hub. Inference uses a local FastAPI API rather than a third-party cloud inference service.
- **Libraries and frameworks:** PyTorch, Torchvision, OpenCLIP, Hugging Face Datasets, NumPy, pandas, SciPy, scikit-learn, Pillow, Matplotlib, PyYAML, ONNX, ONNX Runtime, FastAPI, Uvicorn, Vue 3, TypeScript, and Vite.
- **Datasets and assets:** CommunityForensics-Small supplies the 24,000-image train-v3 set, balanced between 12,000 Real and 12,000 AIGI images and covering 921 exact generator labels. CommunityForensics-Eval supplies the main external evaluation data, while AIGIBench supports an exact-seen slice. CIFAKE and SID-Set are retained only as historical pipeline pilots. Frozen manifests, checksums, perturbation definitions, run cards, per-image predictions, model weights, and error-case thumbnails support reproducibility; raw datasets are not committed to Git.

## 2. Robustness Evaluation Summary

The same 4,000 strict-unseen images are evaluated under one Clean condition and 20 transformed conditions: 17 original perturbations, two four-stage reposting pipelines, and one six-stage random composition. The transformed mean equally weights all 20 non-Clean conditions.

| Model | Clean AUROC | Mean transformed AUROC | Worst transformed AUROC | Clean Accuracy | Mean transformed Accuracy |
|---|---:|---:|---:|---:|---:|
| B0 | 0.8125 | 0.7505 | 0.4846 | 74.10% | 68.85% |
| B1 | 0.8117 | 0.7850 | 0.6665 | 74.33% | 71.48% |
| B2 | 0.7707 | 0.7722 | 0.6743 | 69.80% | 59.83% |
| **M2** | **0.9308** | **0.9163** | **0.8525** | **85.78%** | 83.70% |
| M3 | 0.9305 | 0.9154 | 0.8489 | 85.35% | **83.81%** |

M2 provides the best overall balance: its mean transformed AUROC is only 0.0145 below Clean, although Accuracy falls to 76.53% under the six-stage condition. B1's improvement over the architecturally identical B0 shows the value of robustness augmentation. B2 demonstrates that AUROC alone is insufficient: its ranking performance remains stable after transformation, but its Accuracy declines because its scores shift relative to the frozen threshold.

## 3. Error Analysis Note

At M2's validation-frozen threshold, Clean produces **334 false positives among 2,000 Real images** and **235 false negatives among 2,000 AIGI images**. The six-stage pipeline increases these totals to **432 false positives** and **507 false negatives**.

### Representative false positives

- A LAION product photograph on a white background and a COCO figurine photograph remain high-confidence false positives across the tested multistage pipelines.
- An FFHQ portrait is a Clean false positive but becomes correctly classified after stronger processing, showing that transformations do not always push scores toward AIGI.
- False positives are concentrated in LAION: its FPR is 52.40% under Clean and 57.80% after six-stage processing. This is a descriptive source-level pattern, not evidence of a single visual cause.

### Representative false negatives

- Firefly Image 2 architectural imagery, a DALL·E 2 landscape, and a Stable Cascade seascape are persistent false negatives.
- Stable Cascade is the most difficult generator in the main test, with an FNR of 37.95% under Clean and 54.82% after six-stage processing.
- An Imagen 3 waterfall/city image is correct under Clean but becomes a false negative after multistage processing; its score drops from 0.9966 to 0.7012.

### Main trade-offs

- **Coverage vs. blind spots:** M2 is strongest on the full strict-unseen set, but B2 ranks the difficult Hourglass, DFGAN, and GALIP slices more reliably.
- **AUROC vs. threshold performance:** stable ranking does not guarantee stable Recall or Specificity at a fixed threshold.
- **False positives vs. false negatives:** different transformations move scores in different directions, so one global threshold cannot eliminate both error types.
- **Accuracy vs. deployment safety:** M2/M3 achieve about 0.931 Clean AUROC but 0% TPR at 1% FPR on the current test set. Their sigmoid scores are confidence/ranking scores, not calibrated probabilities; production deployment requires independent target-platform calibration and explicit FP/FN costs.

These results come from one training seed, 12 unseen generators, four Real sources, and a balanced 50% AIGI test prior. Representative errors are diagnostic examples rather than a random estimate of error prevalence.
