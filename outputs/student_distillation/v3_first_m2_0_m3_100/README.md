# V3 第一版 Student：M2 0% + M3 100%

这是 train-v3 上第一版 MobileNetV3-Large Student 的自包含交付。配置文件和历史输出路径虽然沿用 `dual_teacher` 名称，但实际权重为 M2 0%、M3 100%，因此这是 M3-only distillation，不是真正的双教师蒸馏。

## 蒸馏配置

| 项目 | 值 |
|---|---|
| Student backbone | `mobilenet_v3_large` |
| Student 参数量 | 4,203,313 |
| 输入 | 224×224 RGB，归一化嵌入模型 |
| M2 teacher weight | 0.0（0%） |
| M3 teacher weight | 1.0（100%） |
| Hard / KD / consistency | 0.5 / 0.4 / 0.1 |
| KD temperature | 3.0 |
| Epochs | 20 |
| 训练作业 | SLURM 32979 |
| 内部评测作业 | SLURM 32989 |

实际运行参数以 [`resolved_config.yaml`](resolved_config.yaml) 为准。本目录仅代表 V3 第一版，不包含后续 V3.1 的温度修正或 feature distillation。

## 测试结果

| 测试 | Clean AUROC | Balanced accuracy | Robust mean AUROC | Worst AUROC |
|---|---:|---:|---:|---:|
| 内部固定验证，18 conditions | 0.983660 | 0.945500 | 0.973722 | 0.953880 |
| External exact-seen clean | 0.918969 | 0.814500 | — | — |
| Historical unseen clean | 0.874707 | 0.799000 | — | — |
| Expanded V3 unseen 4k，21 conditions | 0.878353 | 0.803000 | 0.849040 | 0.793141 |
| Hard Hourglass | 0.613800 | 0.548000 | — | — |
| Hard DFGAN | 0.636584 | 0.538000 | — | — |
| Hard GALIP | 0.705568 | 0.590000 | — | — |

内部固定阈值为 `0.79248046875`。内部最差条件是 `resize__interpolation=bilinear_scale=0.25`；Expanded V3 unseen 最差条件是 `strict_random_six__profile=full_training_range_v1`。

测试入口：

- 内部验证：[`internal_validation/`](internal_validation/)；
- Exact-seen：[`calibration_exact_seen_tuned/`](calibration_exact_seen_tuned/)；
- Historical unseen：[`test_external_unseen_generator_v1_historical/`](test_external_unseen_generator_v1_historical/)；
- Expanded V3 unseen 4k：[`test_external_unseen_generator_v3_expanded_robustness_v2/`](test_external_unseen_generator_v3_expanded_robustness_v2/)；
- Hard generators：[`diagnostic_hard_hourglass/`](diagnostic_hard_hourglass/)、[`diagnostic_hard_dfgan/`](diagnostic_hard_dfgan/)、[`diagnostic_hard_galip/`](diagnostic_hard_galip/)。

外部测试的样本构成和协议不完全相同，跨测试集数值不能视作严格同集 A/B。每个测试目录中的 `run_card.json`、`summary.json`、`metrics_by_transform.csv` 和 `predictions.jsonl` 是最终审计依据。

## 模型与手机端产物

| 文件 | Bytes | SHA-256 |
|---|---:|---|
| [`best.pt`](best.pt) | 50,812,264 | `362f618e3486d82761126b9b1d959f6c08e46e53d6d4aa8f66a73ac4f035be74` |
| [`latest.pt`](latest.pt) | 50,812,264 | `e772c5799bace854c50852a5aa2755ae17d910f045cda2817b745856bb674538` |
| [`mobile/student_mnv3_fp32.onnx`](mobile/student_mnv3_fp32.onnx) | 16,808,474 | `cdfcec7c56eed55218879ae2aa2eb00f0eac248fa0922b5687e72feb672f2306` |
| [`mobile/student_mnv3_fp32.torchscript.pt`](mobile/student_mnv3_fp32.torchscript.pt) | 16,812,452 | `ae5e1acaa18ec9d214a07db30d4dbd41033459606f5e6d8884f0287215654650` |

V3 第一版 ONNX parity 状态为 `passed`，测试用例最大绝对误差为 `2.2911e-6`。详细结果见 [`mobile/onnx_parity.json`](mobile/onnx_parity.json)，导出说明见 [`mobile/export_metadata.json`](mobile/export_metadata.json)。

## 来源

- TC2 原始目录：`/home/msai/xjiang026/projects/repostguard-lite/outputs/community_forensics_v3/student_mnv3_dual_teacher`
