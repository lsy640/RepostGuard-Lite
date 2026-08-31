# Student Distillation 完整产物

本目录是 MobileNetV3-Large Student 蒸馏模型的完整交付入口。每一轮模型的权重、冻结配置、独立报告、测试结果、逐样本预测和手机端导出均放在对应模型子目录内。

| 模型目录 | M2 比例 | M3 比例 | Student 参数量 | 说明 |
|---|---:|---:|---:|---|
| [`v1_m2_30_m3_70/`](v1_m2_30_m3_70/) | 30% | 70% | 4,203,313 | 第一轮双教师蒸馏 |
| [`v3_first_m2_0_m3_100/`](v3_first_m2_0_m3_100/) | 0% | 100% | 4,203,313 | train-v3 第一版，实际为 M3-only |

## 同一内部验证协议下的简要对比

| 指标 | V1：M2 30% / M3 70% | V3 第一版：M2 0% / M3 100% |
|---|---:|---:|
| Clean AUROC | 0.975284 | 0.983660 |
| Clean balanced accuracy | 0.930500 | 0.945500 |
| Robust mean AUROC | 0.964239 | 0.973722 |
| Robust mean balanced accuracy | 0.892059 | 0.899676 |
| Robust worst AUROC | 0.927802 | 0.953880 |

两轮内部评测使用相同验证 manifest 和相同 transform matrix。外部测试的样本构成及协议并不完全相同，详细指标、限制和产物 SHA-256 请进入各模型目录查看 `README.md`。

## 为什么完整提交模型文件

本次两轮结果总量较小，单文件最大约 51 MB。为了让接收方可以直接复现评测、检查单样本错误并开展手机端测试，本目录完整保留 checkpoint、ONNX、TorchScript 和 `predictions.jsonl`。这些文件是有意覆盖仓库默认忽略规则后提交的。
