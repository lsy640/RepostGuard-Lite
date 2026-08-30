# Community Forensics train-v3 B0/B1/B2/M2/M3 独立评测报告

> 生成时间：2026-08-30T21:52:32+08:00  
> 评测作业：`32885`  
> 扰动矩阵 SHA256：`69531f3f7111651808c99f14f89723bf631345878b1cbd0cbe0eee8531dde83c`

## 结论摘要

- 完整4,000张 unseen 上，Clean AUROC最高为 **M2：0.9308**。
- 完整4,000张 unseen 上，冻结阈值 Clean Accuracy最高为 **M2：85.78%**。
- 三个困难生成器的 Clean 宏平均 AUROC最高为 **B2：0.7261**；其六阶段宏平均AUROC为 0.5659。
- 五切片 Clean 等权宏平均最高为 **B2：0.7521**；新增三组多阶段宏平均最高为 **B2：0.6825**。
- 本报告只描述train-v3，不使用v2结果，也未用external/hard测试标签重新选择checkpoint或阈值。

## 评测范围与数据角色

| 切片 | 角色 | 总数 | Real | AIGI | Manifest SHA256 |
|---|---|---|---|---|---|
| External exact-seen generator | exact-seen external test | 2000 | 1000 | 1000 | `fc4bbd7d05c1` |
| Hard Hourglass | hard-generator test | 500 | 250 | 250 | `6acf2986cfa9` |
| Hard DFGAN | hard-generator test | 500 | 250 | 250 | `b54846aec698` |
| Hard GALIP | hard-generator test | 500 | 250 | 250 | `9e20b9d04d5c` |
| Full unseen-generator (4,000) | strict unseen test | 4000 | 2000 | 2000 | `59ca2e4ca966` |

五个切片每模型共 7,500 张图像、21个条件；五模型共 787,500 条逐样本预测。 Full unseen包含 12 个精确生成器：`dalle2, dalle3, firefly-image2, firefly-image3, flux-dev, flux-schnell, ideogramv1, ideogramv2, imagen3, midjourneyv5-2, midjourneyv6-1, stable-cascade`。 真实来源为 COCO=500, FFHQ=500, LAION=500, RAISE=500。

## Full unseen-generator（4,000张）详细Clean指标

| 模型 | Accuracy | Precision | Recall | Specificity | F1 | MCC | BA | AUROC | AP | TPR@1%FPR | TPR@5%FPR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 | 74.10% | 72.40% | 77.90% | 70.30% | 75.05% | 0.4834 | 0.7410 | 0.8125 | 0.7882 | 10.20% | 29.45% |
| B1 | 74.33% | 72.18% | 79.15% | 69.50% | 75.51% | 0.4888 | 0.7432 | 0.8117 | 0.7860 | 10.00% | 29.95% |
| B2 | 69.80% | 74.18% | 60.75% | 78.85% | 66.79% | 0.4027 | 0.6980 | 0.7707 | 0.7700 | 9.55% | 30.40% |
| M2 | 85.78% | 84.09% | 88.25% | 83.30% | 86.12% | 0.7164 | 0.8578 | 0.9308 | 0.9136 | 0.00% | 45.40% |
| M3 | 85.35% | 83.96% | 87.40% | 83.30% | 85.64% | 0.7076 | 0.8535 | 0.9305 | 0.9125 | 0.00% | 45.30% |

Accuracy、Precision与NPV基于人为50% AIGI测试先验，部署先验变化时不能直接外推。AUROC/AP用于排序，固定阈值指标用于当前内部验证阈值的操作点，两类指标必须联合解释。

## Full unseen-generator 多阶段扰动

| 模型 | 4-stage A AUROC | 4-stage B AUROC | 6-stage AUROC | 6-stage Accuracy | 6-stage Recall | 6-stage Specificity | 6-stage F1 |
|---|---|---|---|---|---|---|---|
| B0 | 0.7447 | 0.7173 | 0.6597 | 61.38% | 52.05% | 70.70% | 57.40% |
| B1 | 0.7872 | 0.7828 | 0.7435 | 67.03% | 65.35% | 68.70% | 66.46% |
| B2 | 0.7743 | 0.6910 | 0.6743 | 58.73% | 26.85% | 90.60% | 39.41% |
| M2 | 0.9153 | 0.8877 | 0.8525 | 76.53% | 74.65% | 78.40% | 76.08% |
| M3 | 0.9132 | 0.8888 | 0.8489 | 76.15% | 74.15% | 78.15% | 75.66% |

## External exact-seen Clean指标

| 模型 | Accuracy | Recall | Specificity | F1 | AUROC | AP | 6-stage AUROC |
|---|---|---|---|---|---|---|---|
| B0 | 70.30% | 89.00% | 51.60% | 74.98% | 0.7933 | 0.7530 | 0.6898 |
| B1 | 69.90% | 91.70% | 48.10% | 75.29% | 0.7999 | 0.7648 | 0.7321 |
| B2 | 73.70% | 77.50% | 69.90% | 74.66% | 0.8114 | 0.8013 | 0.7517 |
| M2 | 75.20% | 86.20% | 64.20% | 77.66% | 0.8558 | 0.8327 | 0.8302 |
| M3 | 76.05% | 86.30% | 65.80% | 78.28% | 0.8578 | 0.8352 | 0.8254 |

## 三个困难生成器Clean指标

| 切片 | 模型 | Accuracy | Recall | Specificity | F1 | AUROC | 6-stage AUROC |
|---|---|---|---|---|---|---|---|
| Hard Hourglass | B0 | 41.80% | 23.20% | 60.40% | 28.50% | 0.3304 | 0.3194 |
| Hard Hourglass | B1 | 41.00% | 28.00% | 54.00% | 32.18% | 0.3056 | 0.2782 |
| Hard Hourglass | B2 | 58.20% | 30.40% | 86.00% | 42.11% | 0.7343 | 0.5655 |
| Hard Hourglass | M2 | 46.20% | 20.40% | 72.00% | 27.49% | 0.4588 | 0.3703 |
| Hard Hourglass | M3 | 46.20% | 18.80% | 73.60% | 25.90% | 0.4615 | 0.3670 |
| Hard DFGAN | B0 | 45.80% | 31.20% | 60.40% | 36.53% | 0.4584 | 0.4474 |
| Hard DFGAN | B1 | 45.40% | 36.80% | 54.00% | 40.26% | 0.4385 | 0.4698 |
| Hard DFGAN | B2 | 56.40% | 26.80% | 86.00% | 38.07% | 0.6856 | 0.5118 |
| Hard DFGAN | M2 | 42.40% | 12.80% | 72.00% | 18.18% | 0.4614 | 0.3929 |
| Hard DFGAN | M3 | 42.80% | 12.00% | 73.60% | 17.34% | 0.4595 | 0.3816 |
| Hard GALIP | B0 | 47.40% | 34.40% | 60.40% | 39.54% | 0.4559 | 0.4469 |
| Hard GALIP | B1 | 41.00% | 28.00% | 54.00% | 32.18% | 0.3691 | 0.4544 |
| Hard GALIP | B2 | 59.60% | 33.20% | 86.00% | 45.11% | 0.7586 | 0.6205 |
| Hard GALIP | M2 | 50.00% | 28.00% | 72.00% | 35.90% | 0.5460 | 0.4689 |
| Hard GALIP | M3 | 50.20% | 26.80% | 73.60% | 34.99% | 0.5526 | 0.4651 |

三个困难切片共享真实负类面板，因此这些AUROC是相关的切片诊断，不能当作三个统计独立总体。每个困难切片同时包含Real与AIGI，因而可报告完整二分类指标；如果后续只抽取单一生成器正类，则只能解释Recall/TP/FN。

## 五切片宏平均

| 模型 | Clean AUROC | 17扰动均值 | 4-stage A | 4-stage B | 6-stage | 新增3组均值 | 新增3组最坏 |
|---|---|---|---|---|---|---|---|
| B0 | 0.5701 | 0.5487 | 0.5294 | 0.5203 | 0.5126 | 0.5208 | 0.2705 |
| B1 | 0.5450 | 0.5503 | 0.5555 | 0.5538 | 0.5356 | 0.5483 | 0.2782 |
| B2 | 0.7521 | 0.7448 | 0.7203 | 0.7024 | 0.6247 | 0.6825 | 0.5118 |
| M2 | 0.6506 | 0.6455 | 0.6293 | 0.6227 | 0.5830 | 0.6117 | 0.3703 |
| M3 | 0.6524 | 0.6462 | 0.6285 | 0.6154 | 0.5776 | 0.6072 | 0.3670 |

## 最低AUROC条件

| 模型 | 切片 | 条件 | AUROC | Accuracy | Recall | Specificity |
|---|---|---|---|---|---|---|
| B0 | Hard Hourglass | 4-stage A platform repost | 0.2705 | 38.60% | 24.80% | 52.40% |
| B1 | Hard Hourglass | 6-stage random composition | 0.2782 | 41.20% | 25.20% | 57.20% |
| B0 | Hard Hourglass | 2-stage crop 0.8 + JPEG Q50 | 0.2797 | 40.20% | 27.60% | 52.80% |
| B0 | Hard Hourglass | Gaussian noise sigma=0.05 | 0.2848 | 38.20% | 24.00% | 52.40% |
| B0 | Hard Hourglass | 2-stage resize 0.5 + JPEG Q70 | 0.2881 | 37.00% | 28.40% | 45.60% |
| B0 | Hard Hourglass | Center crop ratio=0.8 | 0.2891 | 40.20% | 28.00% | 52.40% |
| B1 | Hard Hourglass | Color jitter 0.8/0.8/0.8 | 0.2906 | 42.60% | 20.40% | 64.80% |
| B0 | Hard Hourglass | Resize 0.5 bicubic | 0.2907 | 36.80% | 32.80% | 40.80% |
| B1 | Hard Hourglass | Center crop ratio=0.8 | 0.2925 | 41.40% | 29.60% | 53.20% |
| B0 | Hard Hourglass | Gaussian blur sigma=1.0 | 0.2979 | 37.80% | 30.00% | 45.60% |

## 方法与完整性

- 逐模型核对冻结best checkpoint、resolved config、训练完成标记和内部clean validation阈值。
- 逐切片核对COMPLETE、manifest SHA256、21条件matrix SHA256、每条件样本数与run card checkpoint SHA256。
- Clean、17个既有扰动、两组4-stage和一组确定性随机6-stage均来自同一冻结矩阵。
- 报告不重新训练、不重新推理、不改变阈值，也不使用test/hard标签做模型选择。

## 局限性与下一步

1. 当前只有单训练种子且没有模型差异的配对置信区间；小差异不应表述为统计显著。
2. Full unseen只覆盖12个精确生成器及四类真实来源，不能代表所有未来生成器和真实流量。
3. 固定阈值接近1的模型仍存在校准风险；部署前应在独立calibration set上按FPR约束重新确定操作点。
4. 建议针对最差困难生成器、最低真实来源Specificity以及六阶段条件执行样本配对和生成器分层bootstrap。
5. 使用多个预注册扰动种子和贴近部署流行率的流量回放，重新报告Precision、NPV与成本加权指标。

## 结构化产物

- `reports/evaluations/community_forensics_v3_evaluation/community_forensics_v3_evaluation_metrics.csv`
- `reports/evaluations/community_forensics_v3_evaluation/community_forensics_v3_evaluation_artifact.json`
- `reports/evaluations/community_forensics_v3_evaluation/community_forensics_v3_evaluation_audit.json`
