# V1 Student：M2 30% + M3 70%

这是第一轮 MobileNetV3-Large Student 双教师蒸馏的自包含交付，包含训练 checkpoint、冻结配置、校准记录、全部测试结果、逐样本预测和移动端导出。

## 蒸馏配置

| 项目 | 值 |
|---|---|
| Student backbone | `mobilenet_v3_large` |
| Student 参数量 | 4,203,313 |
| 输入 | 224×224 RGB，归一化嵌入模型 |
| M2 teacher weight | 0.3（30%） |
| M3 teacher weight | 0.7（70%） |
| Hard / KD / consistency | 0.5 / 0.4 / 0.1 |
| KD temperature | 3.0 |
| Epochs | 20 |
| 训练作业 | SLURM 32770 |
| 内部评测作业 | SLURM 32845 |

实际运行参数以 [`resolved_config.yaml`](resolved_config.yaml) 为准。

## 测试结果

| 测试 | Clean AUROC | Balanced accuracy | Robust mean AUROC | Worst AUROC |
|---|---:|---:|---:|---:|
| 内部固定验证，18 conditions | 0.975284 | 0.930500 | 0.964239 | 0.927802 |
| External seen-family clean | 0.706446 | 0.656500 | — | — |
| External exact-seen clean | 0.844743 | 0.770500 | — | — |
| Historical unseen，21 conditions | 0.846498 | 0.757000 | 0.810741 | 0.749797 |
| Hard Hourglass | 0.353232 | 0.428000 | — | — |
| Hard DFGAN | 0.376688 | 0.400000 | — | — |
| Hard GALIP | 0.439840 | 0.418000 | — | — |

内部固定阈值为 `0.79833984375`。内部最差条件是 `resize__interpolation=bilinear_scale=0.25`；Historical unseen 最差条件是 `strict_random_six__profile=full_training_range_v1`。

测试入口：

- 内部验证：[`summary.json`](summary.json)、[`metrics_by_transform.csv`](metrics_by_transform.csv)、[`predictions.jsonl`](predictions.jsonl)；
- Seen-family：[`test_external_seen_family/`](test_external_seen_family/)；
- Historical unseen robustness：[`test_external_unseen_generator_robustness_v2/`](test_external_unseen_generator_robustness_v2/)；
- Exact-seen：[`val_external_exact_seen_generator/`](val_external_exact_seen_generator/)；
- Hard generators：[`val_hard_hourglass/`](val_hard_hourglass/)、[`val_hard_dfgan/`](val_hard_dfgan/)、[`val_hard_galip/`](val_hard_galip/)。

## 模型与手机端产物

| 文件 | Bytes | SHA-256 |
|---|---:|---|
| [`best.pt`](best.pt) | 50,812,968 | `ee78e66239bb5c5a8394f4c9fe3d165c5d985d016e343728231e0f7d43a0982c` |
| [`latest.pt`](latest.pt) | 50,812,968 | `5d19f32492513d4fea459757ce27fa918bf03303d89ab6e811a811802fe8c21a` |
| [`mobile/student_mnv3_fp32.onnx`](mobile/student_mnv3_fp32.onnx) | 16,808,474 | `b44979772ae2cfa4770c555b9c965c5224e523349c0f6d5280dcd13bc212804a` |
| [`mobile/student_mnv3_fp32.torchscript.pt`](mobile/student_mnv3_fp32.torchscript.pt) | 16,812,589 | `7033aeb5c232baf222a40c7a27bc373c29f4ef8766a2f8552d9cd5bca4b20eb8` |

训练完成回执见 [`DONE`](DONE)，教师校准见 [`teacher_calibration.json`](teacher_calibration.json)，移动端导出说明见 [`mobile/export_metadata.json`](mobile/export_metadata.json)。

## 来源

- TC2 原始目录：`/home/msai/xjiang026/projects/repostguard-lite/outputs/community_forensics/student_mnv3_dual_teacher`
- 本次交付副本来源：`artifacts/tc2_run_32770/student_mnv3_dual_teacher`
