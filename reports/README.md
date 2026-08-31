# Reports index

本目录按报告用途组织。生成脚本、SLURM 作业和项目 README 均使用下列路径，后续报告不要再直接写入 `reports/` 根目录。

现有 `data_statistics/`、`evaluations/robustness_v2/`、`evaluations/unseen_generator/` 和根级 `atlases/` 产物属于 protocol-v1。External seen-family 被加入训练后，train-v2 报告将分别写入 `data_statistics/train_v2/`、`evaluations/robustness_v2_train_v2/`、`evaluations/unseen_generator_train_v2/` 和 `atlases/train_v2/`，禁止覆盖历史结果。

| 目录 | 内容 | 主要入口 |
|---|---|---|
| [`summaries/`](summaries/) | 当前项目的人工整理总结 | [模型、数据与阶段性验证总结](summaries/COMMUNITY_FORENSICS_PROJECT_SUMMARY.md) / [train-v1 与 train-v2 对比](summaries/COMMUNITY_FORENSICS_TRAIN_V1_V2_COMPARISON.md) / [可视化 HTML](summaries/COMMUNITY_FORENSICS_TRAIN_V1_V2_COMPARISON.html) |
| [`data_statistics/`](data_statistics/) | 数据统计 HTML、CSV、审计 JSON、TIFF 检查和交付回执 | [数据统计报告](data_statistics/COMMUNITY_FORENSICS_DATA_STATISTICS.html) |
| [`evaluations/external_split/`](evaluations/external_split/) | seen-family 与 strict unseen-generator 构成及 AUROC 诊断 | [外部 split 诊断](evaluations/external_split/community_forensics_external_split_diagnostic.html) |
| [`evaluations/robustness_v2/`](evaluations/robustness_v2/) | 六切片、21 条件的 B0/B1/B2/M2/M3 鲁棒性评测 | [Robustness-v2 报告](evaluations/robustness_v2/COMMUNITY_FORENSICS_B0_B1_B2_M2_M3_ROBUSTNESS_V2.html) |
| [`evaluations/unseen_generator/`](evaluations/unseen_generator/) | strict unseen-generator 的详细固定阈值指标、曲线与切片结果 | [Unseen-generator 准确率报告](evaluations/unseen_generator/COMMUNITY_FORENSICS_UNSEEN_GENERATOR_ACCURACY.html) |
| [`evaluations/train_v1_v2_comparison/`](evaluations/train_v1_v2_comparison/) | train-v1/v2 对齐指标、Strict unseen 105 条明细、审计与 HTML artifact | [对比 CSV](evaluations/train_v1_v2_comparison/community_forensics_train_v1_v2_split_comparison.csv) / [Strict unseen 全指标](evaluations/train_v1_v2_comparison/community_forensics_strict_unseen_all_metrics.csv) |
| [`evaluations/student_distillation_v1_v3/`](evaluations/student_distillation_v1_v3/) | V1 与 train-v3 第一版 Student 的完整蒸馏、测试、checkpoint、逐样本预测和移动端导出 | [结果说明与指标对比](evaluations/student_distillation_v1_v3/README.md) |
| [`atlases/`](atlases/) | 90 个精确生成器示意图、分图、索引和审计 | [总图](atlases/community_forensics_exact_generators_atlas.jpg) |
| [`historical/`](historical/) | CIFAKE、SID-Set 原型阶段结果 | [CIFAKE](historical/INITIAL_RESULTS.md) / [SID-Set](historical/SIDSET_B0_B1_B2_M2_M3_SUMMARY.md) |

## 文件约定

- 大写文件名用于面向阅读者的主报告；
- 小写 CSV/JSON 用于机器读取、审计和报告重建；
- `*_artifact.json` 是便携式 HTML 的规范输入；
- `*_notes.json` 记录报告契约、来源与限制；
- `*_delivery_receipt.log` 记录便携式报告交付验证；
- 历史报告不代表对应原始数据或 checkpoint 仍然存在。

## 重新生成

所有 Python 和报告构建操作必须在 SLURM Compute Node 上执行：

```bash
sbatch scripts/slurm/validate_community_forensics_tiff.sbatch
sbatch scripts/slurm/report_community_forensics_data_statistics.sbatch
sbatch scripts/slurm/diagnose_external_split_auroc.sbatch
sbatch scripts/slurm/build_external_split_report.sbatch
sbatch scripts/slurm/report_community_forensics_robustness_v2.sbatch
sbatch scripts/slurm/report_community_forensics_unseen_accuracy.sbatch
sbatch scripts/slurm/build_community_forensics_exact_generator_atlas.sbatch
```
