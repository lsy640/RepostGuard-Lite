# V3.2.1 epoch-3 winner：M3-only

这是在 19-family holdout 协议下冻结的 V3.2.1 winner（原名 V3.2 corrected）。与 V3.1 相比，它不仅采用 T=1，还修正采样与 teacher calibration，并加入真正的 feature/forensic distillation 和轻量 forensic 分支。`best.pt` 来自 epoch 3；`latest.pt` 保留同一运行的 epoch-10 终点。

## 架构与蒸馏配置

| 项目 | 值 |
|---|---|
| M2 / M3 teacher | 0% / 100% |
| Semantic branch | MobileNetV3-Large |
| Forensic/NPR branch | EfficientNet-B0 lightweight branch |
| Student 参数量 | 7,955,038 |
| KD temperature | 1.0 |
| Hard / soft-KD / consistency / feature | 0.50 / 0.15 / 0.05 / 0.30 |
| Best checkpoint | epoch 3 / global step 1545 |
| Latest checkpoint | epoch 10 / global step 5150 |

实际配置以 [`resolved_config.yaml`](resolved_config.yaml) 为准。

## 评测结果

| 测试 | Clean AUROC | Clean balanced accuracy | Robust mean AUROC | Robust mean balanced accuracy | Worst AUROC |
|---|---:|---:|---:|---:|---:|
| 19-family unseen dev，18 conditions | 0.836965 | 0.780439 | 0.816711 | 0.731273 | 0.769403 |
| Protected expanded V3 unseen 4k，21 conditions | 0.906329 | 0.810500 | 0.871061 | 0.732813 | 0.812674 |
| Fixed 1500 hard-mixture，clean only | 0.654344 | 0.578000 | — | — | — |

内部开发评测见 [`dev_family_unseen_robustness/`](dev_family_unseen_robustness/)，与 V3.1 的完整选择审计见 [`dev_selection_vs_v31_baseline.json`](dev_selection_vs_v31_baseline.json)。4k 最终评测见 [`final_external_v3_expanded_robustness_v2/`](final_external_v3_expanded_robustness_v2/)，该受保护测试未用于选择 checkpoint 或调参。

固定 1,500 张 hard-mixture clean 补充评测见 [`fixed1500_clean/`](fixed1500_clean/)，包含完整逐样本预测和 SD1.4、DFGAN、GALIP、Hourglass 四个子集结果：

| 子集 | AUROC | AP | 冻结阈值 balanced accuracy |
|---|---:|---:|---:|
| Overall | 0.654344 | 0.676611 | 0.578000 |
| SD1.4 exact-seen | 0.911288 | 0.717123 | 0.705333 |
| DFGAN hard | 0.639973 | 0.218085 | 0.591333 |
| GALIP hard | 0.628123 | 0.213629 | 0.587333 |
| Hourglass hard | 0.417696 | 0.230428 | 0.439333 |

这里的冻结阈值为原 family-unseen 阈值 `0.060516357421875`；固定1500上计算的 `0.927734375` 只作为诊断阈值，不用于 protected external 4k。

本版本没有生成 ONNX/TorchScript；移动端导出暂缓，因此目录只包含实际完成的 checkpoint、评测、审计和全部逐样本预测。

## 核心文件 SHA-256

| 文件 | Bytes | SHA-256 |
|---|---:|---|
| [`best.pt`](best.pt) | 96,347,902 | `05edd57825cf28608d26fe77db92f364f37dcf7f070d757e82f85b9e2711cfea` |
| [`latest.pt`](latest.pt) | 96,347,902 | `0f96c768a3091b59dd48cac7d36294d84b834d0933c78287cf4421f94d5dd3cf` |
| [`resolved_config.yaml`](resolved_config.yaml) | 4,097 | `228ff0b69e51cb46375e4831208536295f68d0e38826a0279fcd72ba5a9fc35d` |

全部仓库发布文件校验和见 [`SHA256SUMS.txt`](SHA256SUMS.txt)，补评测运行审计见 [`RELEASE_AUDIT.json`](RELEASE_AUDIT.json)。
