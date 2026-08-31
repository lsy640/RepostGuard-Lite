# Student Distillation 完整产物

本目录是 RepostGuard-Lite Student 蒸馏模型的版本化交付入口。已完成轮次的 checkpoint、冻结配置、评测汇总、运行审计和逐样本预测均放在对应子目录内。原名 V3.2 corrected 的 family-unseen epoch-3 winner 现正式命名为 **V3.2.1**。

## 版本索引

| 模型目录 | M2 比例 | M3 比例 | Student 参数量 | 状态与用途 |
|---|---:|---:|---:|---|
| [`v1_m2_30_m3_70/`](v1_m2_30_m3_70/) | 30% | 70% | 4,203,313 | 第一轮双教师蒸馏；含移动端导出 |
| [`v3_first_m2_0_m3_100/`](v3_first_m2_0_m3_100/) | 0% | 100% | 4,203,313 | **V3.0**：train-v3 第一版，M3-only；含移动端导出 |
| [`v3_1_t3_baseline/`](v3_1_t3_baseline/) | 0% | 100% | 4,203,313 | **V3.1 baseline**：T=3，family-unseen 基线 |
| [`v3_1_t1_diagnostic/`](v3_1_t1_diagnostic/) | 0% | 100% | 4,203,313 | **V3.1 diagnostic**：仅把 KD temperature 改为 T=1 |
| [`v3_2_1/`](v3_2_1/) | 0% | 100% | 7,955,038 | **V3.2.1**：T=1 + feature/forensic distillation；冻结的 epoch-3 winner |

V3.1/V3.2 当前完成运行没有生成 ONNX/TorchScript，因此这里只提交实际存在的全部产物，不伪造移动端文件。V1 与 V3.0 的 ONNX、TorchScript 和 parity 结果仍保留在各自的 `mobile/` 子目录。

## 同一 family-unseen 内部验证协议下的对比

以下 V3.1 T=3 与 V3.2.1 使用相同的 19-family holdout manifest 和相同的 18-condition transform matrix，可以直接 A/B。V1/V3.0 使用的是另一套内部固定验证协议，不能把其高分直接与本表比较。

| 指标 | V3.1 T=3 baseline | V3.2.1 epoch 3 | 变化 |
|---|---:|---:|---:|
| Clean AUROC | 0.791871 | 0.836965 | +0.045094 |
| Clean balanced accuracy | 0.721058 | 0.780439 | +0.059381 |
| Robust mean AUROC | 0.755244 | 0.816711 | +0.061467 |
| Robust mean balanced accuracy | 0.655072 | 0.731273 | +0.076201 |
| Robust worst AUROC | 0.689937 | 0.769403 | +0.079466 |

V3.2.1 还在一次性受保护 expanded V3 unseen 4k（21 conditions）上得到：Clean AUROC **0.906329**、Clean balanced accuracy **0.810500**、Robust mean AUROC **0.871061**、Robust mean balanced accuracy **0.732813**、Robust worst AUROC **0.812674**。该 4k 结果只作最终报告，未用于 checkpoint 选择或调参。

新增固定 1,500 张 hard-mixture clean 诊断结果：总体 AUROC **0.654344**；SD1.4 **0.911288**、DFGAN **0.639973**、GALIP **0.628123**、Hourglass **0.417696**。该结果显示 V3.2.1 的主要短板集中在 Hourglass 与两类 hard GAN；完整逐图预测和双阈值指标见 [`v3_2_1/fixed1500_clean/`](v3_2_1/fixed1500_clean/)。

V3.1 T=1 是诊断运行：best checkpoint 的内部 Clean AUROC 为 **0.783724**，低于同轮 T=3 baseline；它没有完成同口径的 18-condition robustness 评测，因此不放进上面的完整对比表。

## V3.2.2 full-refit 诊断状态（含原 19 个 holdout families）

V3.2.2 full-refit 从头训练，并将 family-unseen 开发阶段暂时剔除的 **19 个 families、2,004 张样本**放回训练集，使用完整 **24,000 行** train-v3 manifest。它已完成到 epoch 10，但固定 1,500 张验证集的最佳 checkpoint 出现在 epoch 1（AUROC **0.726831**），epoch 10 降至 **0.628780**，因此没有继续到 epoch 20。冻结的 V3.2.1 epoch-3 模型保持为正式 evidence/release 模型。

架构和蒸馏方法：

- M3-only teacher（M2 0% / M3 100%），不重训教师；
- MobileNetV3-Large semantic branch + EfficientNet-B0 lightweight forensic/NPR branch；
- projected feature fusion 与 quality gate，总参数量约 7.96M；
- 每个 view 的 affine Platt calibration，KD temperature `T=1`；
- hard / soft-KD / consistency / feature loss = `0.50 / 0.15 / 0.05 / 0.30`；
- feature 项包含 pointwise、relational 与 quality-gate distillation；
- 原计划最多 20 epochs；实际在 epoch 10 门禁后停止，`best.pt` 只由冻结、训练不重叠、非 protected validation 选择；protected external 4k 不参与选 epoch。

训练集为 24,000 行；独立 validation 为 1,500 行（750 Real + 750 AIGI，AIGI 中 LatDiff/GAN/PixDiff 各 250），并通过 `sample_id`、路径和 SHA-256 去重门禁。Full-refit 仅保留为诊断实验，不在本次 release 中发布为新 winner。完整定义见 [`../reports/plans/STUDENT_V32_FULL_REFIT_E20_PLAN.md`](../reports/plans/STUDENT_V32_FULL_REFIT_E20_PLAN.md)、[`../configs/community_forensics_v3/student_v32_full_refit_e20.yaml`](../configs/community_forensics_v3/student_v32_full_refit_e20.yaml) 和 [`../data/manifests/community_forensics_val_v32_full_refit_e20.csv`](../data/manifests/community_forensics_val_v32_full_refit_e20.csv)。

## 产物与复现边界

本目录有意提交 `best.pt`、`latest.pt`、全部现有 `predictions.jsonl`、配置、metrics、summary、run card、quality-gate audit 和必要实现代码。原因是总体数据量仍较小，完整保留可让接收方复查单样本错误、checkpoint lineage 和评测口径。每个新增版本目录提供 `SHA256SUMS.txt`。

外部测试之间的样本构成和协议并不完全一致；跨数据集数值不能视作严格 A/B。各目录中的 `resolved_config.yaml`、`run_card.json`、`summary.json` 和逐样本预测是最终审计依据。
