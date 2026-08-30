# Community Forensics train-v2 / train-v3：Strict Unseen-generator 交集对比报告

> 更新时间：2026-08-30T22:48:10+08:00  
> 评测 manifest SHA256：`70434fc7b38ed2015cac67bc87897139fb202ffa13e736a5ce2f7c759833e42a`  
> 扰动矩阵 SHA256：`69531f3f7111651808c99f14f89723bf631345878b1cbd0cbe0eee8531dde83c`

## 结论摘要

- train-v3 的 Clean AUROC 最优模型为 **M3**（92.61%）。
- train-v3 的 Clean Accuracy 最优模型为 **M2**（85.55%，使用内部验证集冻结阈值）。
- train-v3 的 20 个非 Clean 条件平均 AUROC 最优模型为 **M3**（91.09%）。
- 五个模型中有 **2/5** 个在相同 unseen 交集上的 Clean AUROC 高于 train-v2；有 **5/5** 个在扰动平均 AUROC 上提高。
- 这是 v2/v3 评价集 2,000 张 sample-id 交集、同一 21 条件矩阵下的公平模型版本比较；阈值仍分别来自各训练版本的内部验证集，没有使用 external test 标签调参。
- 补充的 M3 门控消融使用完整 train-v3 4,000 张 unseen-generator 测试集：逐样本动态门控相对固定均值或跨样本打乱门控的整体收益接近于零，仅在 4-stage B 上观察到不足 0.1 AUROC 百分点的微小排序收益。该补充实验与前述 2,000 张 v2/v3 交集比较属于不同评测总体。

## 评测合同与数据范围

- 测试角色：外部 strict unseen-generator，仅用于最终评测，不参与 checkpoint、阈值或模型选择。
- 样本：2,000 张交集图片，1,000 Real / 1,000 AIGI；12 个训练未见精确生成器；COCO、FFHQ、LAION、RAISE 各 250 张真实图片。
- 条件：Clean + 17 个既有单/双阶段条件 + 两组 4-stage + 一组随机 6-stage，共 21 条件。
- 预测总量：2 个训练版本 × 5 个模型 × 2,000 张 × 21 条件 = 420,000 条。
- 正类为 AIGI（label=1），负类为 Real（label=0）。Accuracy、Precision 和 NPV 基于人为 50% AIGI 比例，不能直接外推到生产流量。

## 训练集变化

| 指标 | train-v2 | train-v3 | 变化 |
|---|---:|---:|---:|
| Total images | 20000 | 24000 | +4000 |
| Real images | 10000 | 12000 | +2000 |
| AIGI images | 10000 | 12000 | +2000 |
| Exact AIGI generators | 909 | 921 | +12 |
| GAN AIGI | 242 | 1242 | +1000 |
| Pixel-diffusion AIGI | 121 | 1121 | +1000 |

train-v3 在 train-v2 的 20,000 张基础上增加 1,000 张 GAN AIGI、1,000 张 pixel-diffusion AIGI 和 2,000 张均衡真实图片。多个因素同时变化，因此结果不能唯一归因于某一种新增生成器类别。

## Clean 指标对比

| 模型 | v2 Acc | v3 Acc | ΔAcc | v2 Recall | v3 Recall | v2 Spec | v3 Spec | v2 AUROC | v3 AUROC | ΔAUROC | ΔAUROC 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 74.45% | 74.65% | +0.20 pp | 70.80% | 77.20% | 78.10% | 72.10% | 0.8199 | 0.8199 | -0.0000 | -1.54–+1.62 pp |
| B1 | 74.80% | 74.85% | +0.05 pp | 74.20% | 79.00% | 75.40% | 70.70% | 0.8204 | 0.8152 | -0.0051 | -2.05–+0.87 pp |
| B2 | 69.10% | 69.55% | +0.45 pp | 55.60% | 60.20% | 82.60% | 78.90% | 0.7632 | 0.7641 | +0.0009 | -0.57–+0.69 pp |
| M2 | 84.00% | 85.55% | +1.55 pp | 86.90% | 87.80% | 81.10% | 83.30% | 0.9191 | 0.9252 | +0.0061 | -0.15–+1.40 pp |
| M3 | 85.25% | 84.95% | -0.30 pp | 91.00% | 86.50% | 79.50% | 83.40% | 0.9279 | 0.9261 | -0.0018 | -0.91–+0.49 pp |

AUROC/AP 衡量跨阈值排序能力；Accuracy、Recall、Specificity、F1 和 MCC 衡量各版本自身冻结阈值下的操作点。高 AUROC 不能自动修复不合适的冻结阈值。

## 20 个非 Clean 条件汇总

| 模型 | v2 mean AUROC | v3 mean AUROC | Δ | v2 worst AUROC | v3 worst AUROC | v2 mean Acc | v3 mean Acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.7480 | 0.7531 | +0.0050 | 0.5439 | 0.4893 | 68.12% | 68.88% |
| B1 | 0.7657 | 0.7848 | +0.0191 | 0.6393 | 0.6653 | 69.52% | 71.54% |
| B2 | 0.7635 | 0.7680 | +0.0044 | 0.6543 | 0.6620 | 58.53% | 59.66% |
| M2 | 0.8994 | 0.9107 | +0.0112 | 0.8276 | 0.8431 | 81.88% | 83.46% |
| M3 | 0.9082 | 0.9109 | +0.0027 | 0.8328 | 0.8409 | 82.72% | 83.52% |

## 多阶段共同扰动

| 模型 | 条件 | v2 AUROC | v3 AUROC | ΔAUROC | v2 Acc | v3 Acc |
|---|---|---:|---:|---:|---:|---:|
| B0 | 4-stage A platform repost | 0.7317 | 0.7379 | +0.0061 | 65.60% | 66.65% |
| B0 | 4-stage B edit repost | 0.7105 | 0.7159 | +0.0053 | 65.75% | 65.25% |
| B0 | 6-stage random composition | 0.6683 | 0.6507 | -0.0175 | 61.90% | 61.05% |
| B1 | 4-stage A platform repost | 0.7494 | 0.7842 | +0.0348 | 67.85% | 71.70% |
| B1 | 4-stage B edit repost | 0.6993 | 0.7809 | +0.0817 | 63.60% | 71.05% |
| B1 | 6-stage random composition | 0.6551 | 0.7342 | +0.0791 | 61.05% | 66.60% |
| B2 | 4-stage A platform repost | 0.7734 | 0.7718 | -0.0016 | 62.35% | 64.05% |
| B2 | 4-stage B edit repost | 0.6802 | 0.6813 | +0.0011 | 54.15% | 56.55% |
| B2 | 6-stage random composition | 0.6543 | 0.6620 | +0.0077 | 55.75% | 58.00% |
| M2 | 4-stage A platform repost | 0.8968 | 0.9094 | +0.0126 | 81.85% | 82.95% |
| M2 | 4-stage B edit repost | 0.8631 | 0.8821 | +0.0189 | 74.60% | 79.50% |
| M2 | 6-stage random composition | 0.8276 | 0.8431 | +0.0155 | 71.15% | 75.75% |
| M3 | 4-stage A platform repost | 0.9087 | 0.9081 | -0.0006 | 83.55% | 83.15% |
| M3 | 4-stage B edit repost | 0.8732 | 0.8841 | +0.0109 | 78.65% | 79.75% |
| M3 | 6-stage random composition | 0.8328 | 0.8409 | +0.0081 | 73.90% | 75.50% |

## M3 动态门控消融：逐样本加权整体无实质增益

本节是对 **train-v3 M3 checkpoint 内部融合机制**的补充诊断，不是新的 v2/v3 模型版本比较。评测使用完整的 train-v3 external strict unseen-generator 扩展集，共 4,000 张图片（2,000 Real / 2,000 AIGI），沿用同一 21 条件扰动矩阵和 M3 内部验证集冻结阈值 `0.9970703125`。作业 `32943` 在 `TC2N08` 正常完成，SLURM 终态为 `COMPLETED (0:0)`。

> M3 checkpoint SHA256：`c83f70641a9c8d7f6808e794cfc8c28c0e478feeca7506e489c772a512115b2f`  
> 完整 v3 unseen manifest SHA256：`59ca2e4ca966dac9fa4fb55281153f93e5becdd3e25da83bc2dff3fad36126cd`  
> 扰动矩阵 SHA256：`69531f3f7111651808c99f14f89723bf631345878b1cbd0cbe0eee8531dde83c`

四种推理方式共享完全相同的语义与取证分支特征，仅改变融合权重：

- **学习门控**：使用 checkpoint 学到的逐样本门控。
- **固定 0.5/0.5**：两个分支等权，用于同时检验全局非等权配比和逐样本变化的综合影响。
- **固定 Clean 均值**：对 4,000 张 Clean 测试图片的学习门控做无标签平均，然后对所有样本固定使用；得到语义分支 `58.83%`、取证分支 `41.17%`。该模式主要隔离逐样本门控变化的贡献。由于平均值使用了测试输入的无标签分布，它只用于机理诊断，不能视为从训练/验证阶段独立冻结的部署策略。
- **跨样本打乱门控**：保留每个条件内学习门控的边际分布，但随机打乱门控与样本的对应关系；该模式检验门控是否确实与对应样本匹配。

| 推理方式 | Clean AUROC | Clean Accuracy | 20 个非 Clean 平均 AUROC | 新增 3 组多阶段平均 AUROC | 6-stage AUROC | 最差非 Clean AUROC |
|---|---:|---:|---:|---:|---:|---:|
| 学习门控 | **0.930533** | 85.35% | 0.915381 | **0.883640** | 0.848941 | 0.848941 |
| 固定 Clean 均值 | 0.930264 | **85.43%** | **0.915464** | 0.883232 | 0.848951 | 0.848951 |
| 跨样本打乱门控 | 0.930452 | **85.43%** | 0.915289 | 0.883351 | **0.849029** | **0.849029** |
| 固定 0.5/0.5 | 0.929502 | 85.40% | 0.914846 | 0.881789 | 0.848065 | 0.848065 |

学习门控与固定 Clean 均值的 Clean AUROC 仅相差 `+0.000269`，与跨样本打乱门控仅相差 `+0.000081`。在 20 个非 Clean 条件的平均值上，固定 Clean 均值反而比学习门控高 `0.000083`；在 6-stage 条件上，固定均值和打乱门控也都略高于学习门控。因此，**逐样本门控变化以及门控与具体样本的匹配关系都没有带来稳定的实际收益**。

固定 0.5/0.5 在部分多阶段条件上相对较弱，说明 M3 学到的“语义分支约 58.8%、取证分支约 41.2%”这一全局非等权配比仍可能有价值；但这不能证明逐样本动态调整有效。换言之，当前证据更支持“整体分支配比有效”，而不是“每张图片都需要不同门控”。

### 配对 Bootstrap：4-stage B 存在微小例外

下表报告学习门控减去三种消融方式的差值。区间来自共享样本、Real/AIGI 分层的 1,000 次配对 bootstrap；Accuracy 差值以百分点（pp）表示。

| 条件 | 比较 | ΔAUROC | ΔAUROC 95% CI | ΔAccuracy | ΔAccuracy 95% CI |
|---|---|---:|---:|---:|---:|
| Clean | 学习 − 固定 0.5/0.5 | +0.001031 | -0.000463–+0.002529 | -0.05 pp | -0.53–+0.45 pp |
| Clean | 学习 − 固定 Clean 均值 | +0.000269 | -0.000156–+0.000797 | -0.08 pp | -0.20–+0.03 pp |
| Clean | 学习 − 打乱门控 | +0.000081 | -0.000486–+0.000707 | -0.08 pp | -0.18–0.00 pp |
| 4-stage A | 学习 − 固定 0.5/0.5 | +0.001869 | +0.000388–+0.003332 | +0.90 pp | +0.32–+1.45 pp |
| 4-stage A | 学习 − 固定 Clean 均值 | +0.000303 | -0.001037–+0.001675 | -0.10 pp | -0.48–+0.27 pp |
| 4-stage A | 学习 − 打乱门控 | +0.000244 | -0.001122–+0.001509 | +0.05 pp | -0.22–+0.32 pp |
| 4-stage B | 学习 − 固定 0.5/0.5 | +0.002809 | +0.001214–+0.004304 | +1.60 pp | +0.90–+2.22 pp |
| 4-stage B | 学习 − 固定 Clean 均值 | +0.000932 | +0.000265–+0.001673 | -0.10 pp | -0.28–+0.07 pp |
| 4-stage B | 学习 − 打乱门控 | +0.000713 | +0.000063–+0.001399 | -0.03 pp | -0.22–+0.15 pp |
| 6-stage | 学习 − 固定 0.5/0.5 | +0.000876 | -0.000872–+0.002627 | +0.65 pp | +0.10–+1.22 pp |
| 6-stage | 学习 − 固定 Clean 均值 | -0.000010 | -0.000894–+0.000919 | +0.03 pp | -0.22–+0.25 pp |
| 6-stage | 学习 − 打乱门控 | -0.000089 | -0.001243–+0.001011 | -0.08 pp | -0.30–+0.17 pp |

4-stage B 是唯一在“学习门控对固定均值”和“学习门控对打乱门控”两项比较中 AUROC 区间均未覆盖零的条件，但收益分别只有 `0.000932` 和 `0.000713`，均不足 0.1 AUROC 百分点，而且 Accuracy 没有同步提高。因此不能把结论写成“动态门控绝对没有贡献”，更准确的判断是：**逐样本动态门控在完整 train-v3 unseen 测试上没有稳定且具有实际意义的总体增益，只在 4-stage B 上保留了极小的排序收益。**

完整性核验方面，学习门控的 21 条件指标与既有正式 M3 评测逐项一致，最大绝对差为 `0.0`。本次消融共生成 84 组模式/条件指标和 84,000 条逐样本预测；checkpoint、manifest 与扰动矩阵均记录 SHA256，可从结构化产物追溯。该实验仍只使用一个训练种子，bootstrap 区间不覆盖训练随机性、checkpoint 选择不确定性或数据集构建偏差。

## 精确生成器与真实来源切片

完整 CSV/HTML 表分别给出 12 个精确生成器的 Clean Recall/TP/FN 与四个真实来源的 Clean Specificity/TN/FP。生成器切片只有正类，不能单独定义 Precision、Specificity 或 Accuracy；真实来源切片只有负类，不能单独定义 Recall。每个生成器比较复用同一真实面板时，其 ROC 估计彼此相关，本报告仅作诊断。

## 谱系与完整性检查

- train-v2/test 精确身份重叠：`{'path': 0, 'sample_id': 0, 'sha256': 0, 'source_locator': 0}`。
- train-v3/test 精确身份重叠：`{'path': 0, 'sample_id': 0, 'sha256': 0, 'source_locator': 0}`。
- 每个模型均核对 COMPLETE、run card、checkpoint SHA256、评价 manifest SHA256、matrix SHA256、交集 sample_id 一一对齐、标签一致和概率范围；v2 的 42,000 条交集预测还与既有 metrics_by_transform.csv 复算一致，v3 指标则从 84,000 条扩展集预测中筛选 42,000 条交集预测后重新计算。
- Clean 差值区间使用共享样本的 Real/AIGI 分层配对 bootstrap；区间不覆盖训练随机种子、checkpoint 选择或数据集构建偏差。

## 局限性与下一步

1. 当前每个训练版本只有一个随机种子；小幅差值不应被称为架构层面的统计优势。
2. 训练集同时改变生成器、真实来源与样本规模，无法从本实验唯一识别因果来源。
3. 部署前应在独立 calibration set 上定义可接受 FPR，并报告 TPR@0.1%、1%、5% FPR；不能利用本 strict-unseen test 重新选阈值。
4. 建议至少训练三个种子，并进行样本配对、生成器分层的 bootstrap/置换检验及多重比较校正。
5. 使用贴近部署 AIGI 流行率的回放流量重新估计 Precision、NPV 和成本加权指标。

## 结构化产物

- `reports/evaluations/train_v2_v3_unseen_intersection_comparison/community_forensics_train_v2_v3_unseen_intersection_clean.csv`
- `reports/evaluations/train_v2_v3_unseen_intersection_comparison/community_forensics_train_v2_v3_unseen_intersection_all_conditions.csv`
- `reports/evaluations/train_v2_v3_unseen_intersection_comparison/community_forensics_train_v2_v3_unseen_intersection_multistage.csv`
- `reports/evaluations/train_v2_v3_unseen_intersection_comparison/community_forensics_train_v2_v3_unseen_intersection_generator.csv`
- `reports/evaluations/train_v2_v3_unseen_intersection_comparison/community_forensics_train_v2_v3_unseen_intersection_real_source.csv`
- `reports/evaluations/train_v2_v3_unseen_intersection_comparison/community_forensics_train_v2_v3_unseen_intersection_artifact.json`
- `reports/evaluations/train_v2_v3_unseen_intersection_comparison/community_forensics_train_v2_v3_unseen_intersection_audit.json`
- `outputs/community_forensics_v3_m3_gate_ablation_v1/summary.json`
- `outputs/community_forensics_v3_m3_gate_ablation_v1/metrics_by_mode_and_transform.csv`
- `outputs/community_forensics_v3_m3_gate_ablation_v1/paired_bootstrap_key_conditions.csv`
- `outputs/community_forensics_v3_m3_gate_ablation_v1/predictions_by_mode.jsonl`
- `outputs/community_forensics_v3_m3_gate_ablation_v1/run_card.json`
