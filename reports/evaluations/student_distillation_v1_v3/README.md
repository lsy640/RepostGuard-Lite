# Student 蒸馏 V1 与 V3 第一版完整交付

本目录保存两轮 MobileNetV3-Large Student 蒸馏的完整结果快照，包括训练 checkpoint、冻结配置、教师校准、全部测试汇总、逐退化条件指标、逐样本预测，以及 ONNX/TorchScript 手机端导出。

> 完整产物纳入 Git 的原因：本次两轮结果的数据量较小，单文件最大约 51 MB，保留 checkpoint、移动端模型和 `predictions.jsonl` 能让接收方直接复现评测、检查单样本错误并开展手机端测试。它们通过 `git add -f` 明确覆盖仓库对模型和预测文件的默认忽略规则。

## 两轮快照

| 快照 | 训练作业 | Student | 教师权重 | 训练目标权重 | 说明 |
|---|---:|---:|---|---|---|
| [`v1/`](v1/) | 32770 | 4,203,313 参数 | M2 0.3 / M3 0.7 | hard 0.5 / KD 0.4 / consistency 0.1 | 第一轮 M2+M3 双教师蒸馏 |
| [`v3_first/`](v3_first/) | 32979 | 4,203,313 参数 | M2 0.0 / M3 1.0 | hard 0.5 / KD 0.4 / consistency 0.1 | train-v3 上的第一版 Student；配置名沿用 `dual_teacher`，实际为 M3-only |

两轮均使用 `mobilenet_v3_large`、224×224 RGB 输入、20 epochs、KD temperature 3.0。这里的 `v3_first` 是已完成的第一版，不包含后续 V3.1 的温度修正或 feature distillation 实验。

## 主要测试结果

### 内部固定验证集（18 conditions）

两轮 run card 的验证 manifest SHA-256 都是 `11bfa4b6d7c538ce0a3d774c3f2902ac11ffc7dbe513de354f87fbaad1d6b6ba`，评测矩阵 SHA-256 都是 `ff8b2c24739a833b2c6a970d1c2e64ee14f25af8554bd2f3f7d8aa589a4acbea`。

| 指标 | V1 | V3 第一版 | V3 - V1 |
|---|---:|---:|---:|
| Clean AUROC | 0.975284 | 0.983660 | +0.008377 |
| Clean balanced accuracy | 0.930500 | 0.945500 | +0.015000 |
| Robust mean AUROC | 0.964239 | 0.973722 | +0.009483 |
| Robust mean balanced accuracy | 0.892059 | 0.899676 | +0.007618 |
| Robust worst AUROC | 0.927802 | 0.953880 | +0.026079 |
| 固定阈值 | 0.798340 | 0.792480 | -0.005859 |

两轮最差条件均为 `resize__interpolation=bilinear_scale=0.25`。

### 外部与 hard-generator 测试

| 测试 | V1 AUROC / BA | V3 第一版 AUROC / BA | 备注 |
|---|---:|---:|---|
| External seen-family clean | 0.706446 / 0.656500 | — | V1 专用报告 |
| External exact-seen clean | 0.844743 / 0.770500 | 0.918969 / 0.814500 | V3 目录名为 `calibration_exact_seen_tuned` |
| Historical unseen clean | 0.846498 / 0.757000 | 0.874707 / 0.799000 | V1 对应报告还包含 21 条鲁棒性条件 |
| Historical unseen robust mean | 0.810741 / 0.725875 | — | V1，21 conditions |
| Historical unseen worst AUROC | 0.749797 | — | `strict_random_six` |
| Expanded V3 unseen clean（4k） | — | 0.878353 / 0.803000 | V3 第一版，21 conditions |
| Expanded V3 unseen robust mean（4k） | — | 0.849040 / 0.774463 | V3 第一版 |
| Expanded V3 unseen worst AUROC（4k） | — | 0.793141 | `strict_random_six` |
| Hard Hourglass | 0.353232 / 0.428000 | 0.613800 / 0.548000 | 单条件诊断 |
| Hard DFGAN | 0.376688 / 0.400000 | 0.636584 / 0.538000 | 单条件诊断 |
| Hard GALIP | 0.439840 / 0.418000 | 0.705568 / 0.590000 | 单条件诊断 |

外部测试的样本构成和协议并不完全相同，尤其是 V3 expanded unseen 4k，因此不要把所有行都当作严格的同集 A/B。每个目录中的 `run_card.json`、`summary.json`、`metrics_by_transform.csv` 和 `predictions.jsonl` 是最终审计依据。

## 产物入口

每个快照根目录包含：

- `best.pt` / `latest.pt`：最佳与最后训练状态；
- `resolved_config.yaml`：实际冻结训练配置；
- `DONE`：训练作业与最佳指标回执；
- `teacher_calibration*.json`：教师温度校准和门禁记录；
- 各评测子目录：汇总、逐条件指标、运行环境与逐样本预测；
- `mobile/`：ONNX、TorchScript、导出元数据；V3 第一版还包含 ONNX parity 结果。

关键部署产物：

| 文件 | Bytes | SHA-256 |
|---|---:|---|
| `v1/best.pt` | 50,812,968 | `ee78e66239bb5c5a8394f4c9fe3d165c5d985d016e343728231e0f7d43a0982c` |
| `v1/mobile/student_mnv3_fp32.onnx` | 16,808,474 | `b44979772ae2cfa4770c555b9c965c5224e523349c0f6d5280dcd13bc212804a` |
| `v1/mobile/student_mnv3_fp32.torchscript.pt` | 16,812,589 | `7033aeb5c232baf222a40c7a27bc373c29f4ef8766a2f8552d9cd5bca4b20eb8` |
| `v3_first/best.pt` | 50,812,264 | `362f618e3486d82761126b9b1d959f6c08e46e53d6d4aa8f66a73ac4f035be74` |
| `v3_first/mobile/student_mnv3_fp32.onnx` | 16,808,474 | `cdfcec7c56eed55218879ae2aa2eb00f0eac248fa0922b5687e72feb672f2306` |
| `v3_first/mobile/student_mnv3_fp32.torchscript.pt` | 16,812,452 | `ae5e1acaa18ec9d214a07db30d4dbd41033459606f5e6d8884f0287215654650` |

V3 第一版 ONNX parity 状态为 `passed`；测试用例最大绝对误差为 `2.2911e-6`。两轮完整快照分别约 163.4 MB 和 178.8 MB（十进制），合计约 342.2 MB。

## 来源

- V1 TC2 原始目录：`/home/msai/xjiang026/projects/repostguard-lite/outputs/community_forensics/student_mnv3_dual_teacher`
- V1 本次交付副本来源：`artifacts/tc2_run_32770/student_mnv3_dual_teacher`
- V3 第一版 TC2 原始目录：`/home/msai/xjiang026/projects/repostguard-lite/outputs/community_forensics_v3/student_mnv3_dual_teacher`
