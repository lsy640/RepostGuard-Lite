# Reports 文件清单与审计索引

本文件是 RepostGuard-Lite 的完整报告目录。审计快照日期为 **2026-08-31**，范围为 [reports/](./) 下的全部 **73 个文件、14 个目录**，文件逻辑大小约 **19 MiB**。本次仅做静态文件、结构、链接和血缘核验，没有重新训练、推理或计算评测指标。

## 建议阅读顺序

| 目的 | 首选入口 | 协议状态 |
|---|---|---|
| 快速查看当前 train-v3 鲁棒性 | [Robustness Evaluation Summary](summaries/COMMUNITY_FORENSICS_V3_ROBUSTNESS_EVALUATION_SUMMARY.md) | 当前 |
| 查看代表性 FP/FN 与部署权衡 | [Error Analysis Note](summaries/COMMUNITY_FORENSICS_V3_ERROR_ANALYSIS_NOTE.md) | 当前 |
| 查看五模型、五切片、21 条件完整结果 | [train-v3 完整评测 Markdown](summaries/COMMUNITY_FORENSICS_V3_EVALUATION_REPORT.md) / [便携 HTML](summaries/COMMUNITY_FORENSICS_V3_EVALUATION_REPORT.html) | 当前 |
| 重建 train-v1/v2/v3 与评测数据 | [Train/Val Dataset Manifest Summary](summaries/COMMUNITY_FORENSICS_TRAIN_V3_DATASET_MANIFEST_SUMMARY.md) | 当前 |
| 比较 train-v2 与 train-v3 | [v2/v3 交集对比 Markdown](summaries/COMMUNITY_FORENSICS_TRAIN_V2_V3_UNSEEN_INTERSECTION_COMPARISON.md) | 当前版本对比 |
| 追溯 CIFAKE、SID-Set 和早期 Community Forensics | [historical/](historical/) 及下方历史清单 | 历史 |

## 审计结论

| 检查项 | 结果 |
|---|---|
| 文件覆盖 | 本索引逐项列出 73/73 个文件；不是只列主报告 |
| 类型分布 | 22 CSV、16 JSON、10 Markdown、10 JPEG、7 HTML、6 LOG、1 SVG、1 TSV |
| JSON | 16/16 可由 jq 解析 |
| HTML | 7/7 非空且含标题；其中 6 份有交付回执 |
| 交付回执 | 6/6 显示 validation/package passed；浏览器验证均为 structural_only，因为 browser_unavailable |
| 图片 | 3 张 atlas JPEG、7 张错误案例缩略图均可识别；SVG 通过 XML 结构检查 |
| 错误案例血缘 | 7/7 缩略图与 [TSV 血缘清单](assets/error_analysis/error_analysis_examples.tsv) 中 SHA-256 一致 |
| 文件卫生 | 无空文件、无小于 100 B 的异常文件、无符号链接、无字节级完全重复文件 |
| Markdown 链接 | 当前主报告与本索引的本地链接可解析；历史 SID-Set 报告仍有 27 个指向已清理产物的失效链接 |

需要特别注意：

- [COMMUNITY_FORENSICS_PROJECT_SUMMARY.md](summaries/COMMUNITY_FORENSICS_PROJECT_SUMMARY.md) 是 train-v2 阶段快照，已被 train-v3 文档取代，不能再作为“当前项目状态”。
- [v2/v3 对比 Markdown](summaries/COMMUNITY_FORENSICS_TRAIN_V2_V3_UNSEEN_INTERSECTION_COMPARISON.md) 后续加入了 M3 门控消融；同名 HTML 生成较早，未包含该新增段落，因此精确结论以 Markdown 为准。
- [atlases/](atlases/) 展示的是 protocol-v1 的 69 train + 9 seen-family + 12 unseen 生成器示意。原 seen-family 后来被提升进 train-v2，因此这些图不能解释为当前 train-v3 split。
- [SID-Set 历史总结](historical/SIDSET_B0_B1_B2_M2_M3_SUMMARY.md) 的正文仍可阅读，但原始数据、checkpoint 和多数结构化输出已按存储清理要求删除；27 个证据链接因此不可恢复。
- [CIFAKE 初始结果](historical/INITIAL_RESULTS.md) 末尾的旧 outputs 路径在目录整理后不再有效；当前只保留报告和聚合 CSV。
- 交付回执中的 structural_only 证明 artifact 载荷、打包结构和语义回退通过，不等同于真实 Chromium 中的交互、窄屏或跨浏览器验证。

## 状态标签

- **当前**：train-v3 冻结协议或直接支持当前结论。
- **版本对比**：用于比较不同训练集谱系，不应与单版本指标混合。
- **历史**：保留早期协议证据，不代表当前数据角色或 checkpoint 仍存在。
- **支持产物**：机器可读指标、artifact、audit、notes、receipt、图片或索引；不应脱离对应主报告单独解释。

## 完整文件清单

### reports 根目录

| 文件 | 用途 | 状态与审计 |
|---|---|---|
| [README_reports.md](README_reports.md) | 本目录完整索引 | 当前；覆盖 73 个文件 |

### assets/error_analysis

| 文件 | 用途 | 状态与审计 |
|---|---|---|
| [error_analysis_examples.tsv](assets/error_analysis/error_analysis_examples.tsv) | 7 个 FP/FN 缩略图的 sample_id、来源、原图/缩略图 SHA-256、尺寸和字节数 | 当前支持产物；8 行含表头，7/7 checksum 已核对 |
| [fp_l_laion.jpg](assets/error_analysis/fp_l_laion.jpg) | FP-L，LAION Real 商品图缩略图 | 当前 Error Analysis 资产 |
| [fp_c_coco.jpg](assets/error_analysis/fp_c_coco.jpg) | FP-C，COCO Real 玩偶照片缩略图 | 当前 Error Analysis 资产 |
| [fp_f_ffhq.jpg](assets/error_analysis/fp_f_ffhq.jpg) | FP-F，FFHQ Real 人像缩略图 | 当前 Error Analysis 资产 |
| [fn_f_firefly_image2.jpg](assets/error_analysis/fn_f_firefly_image2.jpg) | FN-F，Firefly Image 2 建筑图缩略图 | 当前 Error Analysis 资产 |
| [fn_d_dalle2.jpg](assets/error_analysis/fn_d_dalle2.jpg) | FN-D，DALL·E 2 风景图缩略图 | 当前 Error Analysis 资产 |
| [fn_s_stable_cascade.jpg](assets/error_analysis/fn_s_stable_cascade.jpg) | FN-S，Stable Cascade 海面云层图缩略图 | 当前 Error Analysis 资产 |
| [fn_i_imagen3.jpg](assets/error_analysis/fn_i_imagen3.jpg) | FN-I，Imagen 3 瀑布城市图缩略图 | 当前 Error Analysis 资产 |

### summaries

| 文件 | 用途 | 状态与审计 |
|---|---|---|
| [COMMUNITY_FORENSICS_V3_ROBUSTNESS_EVALUATION_SUMMARY.md](summaries/COMMUNITY_FORENSICS_V3_ROBUSTNESS_EVALUATION_SUMMARY.md) | 4,000 张 strict unseen 上 Clean 与 20 个 transformed 条件的紧凑总结 | **当前首要摘要** |
| [COMMUNITY_FORENSICS_V3_ERROR_ANALYSIS_NOTE.md](summaries/COMMUNITY_FORENSICS_V3_ERROR_ANALYSIS_NOTE.md) | M2 代表性 FP/FN、错误集中与方法权衡 | **当前错误分析**；已嵌入 7 张有血缘记录的缩略图 |
| [COMMUNITY_FORENSICS_V3_EVALUATION_REPORT.md](summaries/COMMUNITY_FORENSICS_V3_EVALUATION_REPORT.md) | train-v3 五模型、五切片、21 条件完整文本报告 | **当前完整评测** |
| [COMMUNITY_FORENSICS_V3_EVALUATION_REPORT.html](summaries/COMMUNITY_FORENSICS_V3_EVALUATION_REPORT.html) | 上述评测的便携交互版本 | 当前；配套 receipt 为 structural_only |
| [COMMUNITY_FORENSICS_TRAIN_V3_DATASET_MANIFEST_SUMMARY.md](summaries/COMMUNITY_FORENSICS_TRAIN_V3_DATASET_MANIFEST_SUMMARY.md) | train-v1/v2/v3、val/test 清单及从空目录重建流程 | **当前数据复现入口** |
| [COMMUNITY_FORENSICS_TRAIN_V2_V3_UNSEEN_INTERSECTION_COMPARISON.md](summaries/COMMUNITY_FORENSICS_TRAIN_V2_V3_UNSEEN_INTERSECTION_COMPARISON.md) | v2/v3 在共同 2,000 张 strict-unseen 上的对比，另含完整 4,000 张 M3 门控消融 | **当前版本对比，以此文件为准** |
| [COMMUNITY_FORENSICS_TRAIN_V2_V3_UNSEEN_INTERSECTION_COMPARISON.html](summaries/COMMUNITY_FORENSICS_TRAIN_V2_V3_UNSEEN_INTERSECTION_COMPARISON.html) | v2/v3 交集对比的便携可视化 | 版本对比；早于 MD 更新，缺少后来加入的门控消融 |
| [COMMUNITY_FORENSICS_TRAIN_V1_V2_COMPARISON.md](summaries/COMMUNITY_FORENSICS_TRAIN_V1_V2_COMPARISON.md) | train-v1 与 train-v2 的人工对比 | 历史版本对比 |
| [COMMUNITY_FORENSICS_TRAIN_V1_V2_COMPARISON.html](summaries/COMMUNITY_FORENSICS_TRAIN_V1_V2_COMPARISON.html) | train-v1/v2 对比和 strict-unseen 细节可视化 | 历史版本对比；本地引用均存在 |
| [COMMUNITY_FORENSICS_PROJECT_SUMMARY.md](summaries/COMMUNITY_FORENSICS_PROJECT_SUMMARY.md) | 2026-08-30 的模型、数据与阶段总结 | **已被 train-v3 文档取代**；内容完整但仅代表 train-v2 快照 |

### evaluations/community_forensics_v3_evaluation

| 文件 | 用途 | 状态与审计 |
|---|---|---|
| [community_forensics_v3_evaluation_metrics.csv](evaluations/community_forensics_v3_evaluation/community_forensics_v3_evaluation_metrics.csv) | 5 模型 × 5 split × 21 条件，共 525 条指标记录 | 当前机器可读主表 |
| [community_forensics_v3_evaluation_artifact.json](evaluations/community_forensics_v3_evaluation/community_forensics_v3_evaluation_artifact.json) | 便携 HTML 的规范 artifact | 当前支持产物；JSON 合法 |
| [community_forensics_v3_evaluation_audit.json](evaluations/community_forensics_v3_evaluation/community_forensics_v3_evaluation_audit.json) | 协议、数据完整性、模型谱系、矩阵与图表审计 | 当前审计主文件；协议 ID 已冻结 |
| [community_forensics_v3_evaluation_delivery_receipt.log](evaluations/community_forensics_v3_evaluation/community_forensics_v3_evaluation_delivery_receipt.log) | HTML 验证与打包回执 | ok；validation/package passed；structural_only |
| [robustness_clean_vs_transformed.svg](evaluations/community_forensics_v3_evaluation/robustness_clean_vs_transformed.svg) | Robustness Summary 的 Clean/平均 transformed/最坏条件可视化 | 当前支持图；XML 结构有效 |

### evaluations/train_v2_v3_unseen_intersection_comparison

| 文件 | 用途 | 状态与审计 |
|---|---|---|
| [community_forensics_train_v2_v3_unseen_intersection_all_conditions.csv](evaluations/train_v2_v3_unseen_intersection_comparison/community_forensics_train_v2_v3_unseen_intersection_all_conditions.csv) | 5 模型 × 21 条件的 v2/v3 交集对比，共 105 条 | 当前版本对比主表 |
| [community_forensics_train_v2_v3_unseen_intersection_clean.csv](evaluations/train_v2_v3_unseen_intersection_comparison/community_forensics_train_v2_v3_unseen_intersection_clean.csv) | Clean 条件五模型对比 | 版本对比支持产物 |
| [community_forensics_train_v2_v3_unseen_intersection_generator.csv](evaluations/train_v2_v3_unseen_intersection_comparison/community_forensics_train_v2_v3_unseen_intersection_generator.csv) | 12 个生成器 × 5 模型对比，共 60 条 | 版本对比支持产物 |
| [community_forensics_train_v2_v3_unseen_intersection_real_source.csv](evaluations/train_v2_v3_unseen_intersection_comparison/community_forensics_train_v2_v3_unseen_intersection_real_source.csv) | 4 个 Real 来源 × 5 模型对比，共 20 条 | 版本对比支持产物 |
| [community_forensics_train_v2_v3_unseen_intersection_multistage.csv](evaluations/train_v2_v3_unseen_intersection_comparison/community_forensics_train_v2_v3_unseen_intersection_multistage.csv) | 两组 4-stage 与一组 6-stage 的 15 条模型对比 | 版本对比支持产物 |
| [community_forensics_train_v2_v3_unseen_intersection_lineage.csv](evaluations/train_v2_v3_unseen_intersection_comparison/community_forensics_train_v2_v3_unseen_intersection_lineage.csv) | v2/v3 各五模型 checkpoint、阈值和哈希谱系 | 版本对比支持产物 |
| [community_forensics_train_v2_v3_unseen_intersection_artifact.json](evaluations/train_v2_v3_unseen_intersection_comparison/community_forensics_train_v2_v3_unseen_intersection_artifact.json) | 同名 HTML 的规范 artifact | 版本对比支持产物；JSON 合法 |
| [community_forensics_train_v2_v3_unseen_intersection_audit.json](evaluations/train_v2_v3_unseen_intersection_comparison/community_forensics_train_v2_v3_unseen_intersection_audit.json) | 2,000 张交集身份、协议、阈值与 bootstrap 审计 | 版本对比审计主文件 |
| [community_forensics_train_v2_v3_unseen_intersection_delivery_receipt.log](evaluations/train_v2_v3_unseen_intersection_comparison/community_forensics_train_v2_v3_unseen_intersection_delivery_receipt.log) | HTML 交付回执 | ok；validation/package passed；structural_only |

### evaluations/train_v1_v2_comparison

| 文件 | 用途 | 状态与审计 |
|---|---|---|
| [community_forensics_train_v1_v2_split_comparison.csv](evaluations/train_v1_v2_comparison/community_forensics_train_v1_v2_split_comparison.csv) | 5 模型 × 5 split 的 v1/v2 对比，共 25 条 | 历史版本对比主表 |
| [community_forensics_strict_unseen_all_metrics.csv](evaluations/train_v1_v2_comparison/community_forensics_strict_unseen_all_metrics.csv) | train-v2 strict-unseen 的 5 模型 × 21 条件，共 105 条 | 历史版本对比支持产物 |
| [community_forensics_strict_unseen_clean_metrics.csv](evaluations/train_v1_v2_comparison/community_forensics_strict_unseen_clean_metrics.csv) | train-v2 strict-unseen Clean 五模型指标 | 历史支持产物 |
| [community_forensics_strict_unseen_generator_metrics.csv](evaluations/train_v1_v2_comparison/community_forensics_strict_unseen_generator_metrics.csv) | 12 个生成器 × 5 模型，共 60 条 | 历史支持产物 |
| [community_forensics_strict_unseen_real_source_metrics.csv](evaluations/train_v1_v2_comparison/community_forensics_strict_unseen_real_source_metrics.csv) | 4 个 Real 来源 × 5 模型，共 20 条 | 历史支持产物 |
| [community_forensics_m3_unseen_generator_roc_metrics.csv](evaluations/train_v1_v2_comparison/community_forensics_m3_unseen_generator_roc_metrics.csv) | M3 的 12 生成器 × Clean/4-stage A/4-stage B ROC 明细，共 36 条 | 历史支持产物 |
| [community_forensics_train_v1_v2_artifact.json](evaluations/train_v1_v2_comparison/community_forensics_train_v1_v2_artifact.json) | train-v1/v2 HTML 的规范 artifact | 历史支持产物；JSON 合法 |
| [community_forensics_train_v1_v2_notes.json](evaluations/train_v1_v2_comparison/community_forensics_train_v1_v2_notes.json) | 报告契约、输入哈希、bootstrap 和数据集行数 | 历史审计支持产物 |
| [community_forensics_train_v1_v2_delivery_receipt.log](evaluations/train_v1_v2_comparison/community_forensics_train_v1_v2_delivery_receipt.log) | HTML 交付回执 | ok；validation/package passed；structural_only |

### evaluations/robustness_v2

这里的 robustness_v2 表示第二版扰动矩阵，不表示 train-v2。文件使用的是早期 Community Forensics train-v1 checkpoint。

| 文件 | 用途 | 状态与审计 |
|---|---|---|
| [COMMUNITY_FORENSICS_B0_B1_B2_M2_M3_ROBUSTNESS_V2.html](evaluations/robustness_v2/COMMUNITY_FORENSICS_B0_B1_B2_M2_M3_ROBUSTNESS_V2.html) | 5 模型 × 6 split × 21 条件的可视化报告 | 历史 protocol-v1 |
| [community_forensics_robustness_v2_metrics.csv](evaluations/robustness_v2/community_forensics_robustness_v2_metrics.csv) | 上述 630 个 model/split/condition 评测单元 | 历史机器可读主表 |
| [community_forensics_robustness_v2_report_artifact.json](evaluations/robustness_v2/community_forensics_robustness_v2_report_artifact.json) | 便携 HTML 的规范 artifact | 历史支持产物；JSON 合法 |
| [community_forensics_robustness_v2_report_notes.json](evaluations/robustness_v2/community_forensics_robustness_v2_report_notes.json) | 模型/checkpoint、扰动矩阵和报告契约 | 历史审计支持产物 |
| [community_forensics_robustness_v2_delivery_receipt.log](evaluations/robustness_v2/community_forensics_robustness_v2_delivery_receipt.log) | HTML 交付回执 | ok；validation/package passed；structural_only |

### evaluations/unseen_generator

| 文件 | 用途 | 状态与审计 |
|---|---|---|
| [COMMUNITY_FORENSICS_UNSEEN_GENERATOR_ACCURACY.html](evaluations/unseen_generator/COMMUNITY_FORENSICS_UNSEEN_GENERATOR_ACCURACY.html) | 早期 2,000 张 strict unseen 的详细分类、ROC/PR 和切片报告 | 历史 protocol-v1 |
| [community_forensics_unseen_generator_all_metrics.csv](evaluations/unseen_generator/community_forensics_unseen_generator_all_metrics.csv) | 5 模型 × 21 条件，共 105 条指标 | 历史机器可读主表 |
| [community_forensics_unseen_generator_clean_metrics.csv](evaluations/unseen_generator/community_forensics_unseen_generator_clean_metrics.csv) | Clean 五模型详细指标 | 历史支持产物 |
| [community_forensics_unseen_generator_slice_metrics.csv](evaluations/unseen_generator/community_forensics_unseen_generator_slice_metrics.csv) | 生成器 Recall 与 Real 来源 Specificity 切片 | 历史支持产物 |
| [community_forensics_unseen_generator_accuracy_artifact.json](evaluations/unseen_generator/community_forensics_unseen_generator_accuracy_artifact.json) | 便携 HTML 的规范 artifact | 历史支持产物；JSON 合法 |
| [community_forensics_unseen_generator_accuracy_notes.json](evaluations/unseen_generator/community_forensics_unseen_generator_accuracy_notes.json) | manifest、矩阵、输入哈希和 bootstrap 契约 | 历史审计支持产物 |
| [community_forensics_unseen_generator_accuracy_delivery_receipt.log](evaluations/unseen_generator/community_forensics_unseen_generator_accuracy_delivery_receipt.log) | HTML 交付回执 | ok；validation/package passed；structural_only |

### evaluations/external_split

| 文件 | 用途 | 状态与审计 |
|---|---|---|
| [community_forensics_external_split_diagnostic.html](evaluations/external_split/community_forensics_external_split_diagnostic.html) | 解释早期 seen-family 与 unseen-family AUROC 差异 | 历史 protocol-v1；无单独 delivery receipt，仅完成静态结构检查 |
| [community_forensics_external_split_diagnostic.json](evaluations/external_split/community_forensics_external_split_diagnostic.json) | 比较行、生成器行、构成、bootstrap 与来源清单 | 历史机器可读诊断 |
| [community_forensics_external_split_diagnostic_artifact.json](evaluations/external_split/community_forensics_external_split_diagnostic_artifact.json) | 上述 HTML 的规范 artifact | 历史支持产物；JSON 合法 |

### data_statistics

该目录是 train-v1 时八份冻结 manifest 的统计快照；原 External seen-family 后续已进入训练，不能将这里的 split 角色视为 train-v3 当前协议。

| 文件 | 用途 | 状态与审计 |
|---|---|---|
| [COMMUNITY_FORENSICS_DATA_STATISTICS.html](data_statistics/COMMUNITY_FORENSICS_DATA_STATISTICS.html) | 训练集与当时全部 val/test 的统计可视化 | 历史 protocol-v1 |
| [community_forensics_data_statistics.csv](data_statistics/community_forensics_data_statistics.csv) | 8 个 manifest 的 split 聚合 | 历史支持产物 |
| [community_forensics_distribution_statistics.csv](data_statistics/community_forensics_distribution_statistics.csv) | 类别、来源、架构、格式等 115 条分布记录 | 历史支持产物 |
| [community_forensics_generator_statistics.csv](data_statistics/community_forensics_generator_statistics.csv) | 1,025 条精确生成器统计 | 历史支持产物 |
| [community_forensics_tiff_integrity.json](data_statistics/community_forensics_tiff_integrity.json) | TIFF 解码与异常统计 | 历史审计支持产物；JSON 合法 |
| [community_forensics_data_statistics_artifact.json](data_statistics/community_forensics_data_statistics_artifact.json) | 便携 HTML 的规范 artifact | 历史支持产物；JSON 合法 |
| [community_forensics_data_statistics_notes.json](data_statistics/community_forensics_data_statistics_notes.json) | manifest 血缘、审计细节与报告契约 | 历史审计支持产物 |
| [community_forensics_data_statistics_delivery_receipt.log](data_statistics/community_forensics_data_statistics_delivery_receipt.log) | HTML 交付回执 | ok；validation/package passed；structural_only |

### atlases

| 文件 | 用途 | 状态与审计 |
|---|---|---|
| [community_forensics_exact_generators_atlas.jpg](atlases/community_forensics_exact_generators_atlas.jpg) | 90 个精确生成器总图 | 历史 protocol-v1；69 train + 9 seen-family + 12 unseen |
| [community_forensics_train_exact_generators_atlas.jpg](atlases/community_forensics_train_exact_generators_atlas.jpg) | 训练生成器分图 | 历史；69/900 个 train-v1 生成器代表 |
| [community_forensics_test_exact_generators_atlas.jpg](atlases/community_forensics_test_exact_generators_atlas.jpg) | 测试生成器分图 | 历史；9 seen-family + 12 unseen |
| [community_forensics_exact_generators_atlas_index.csv](atlases/community_forensics_exact_generators_atlas_index.csv) | 90 个 tile 的样本、生成器和位置索引 | 历史支持产物；tile/generator/sample_id 均唯一 |
| [community_forensics_exact_generators_atlas_audit.json](atlases/community_forensics_exact_generators_atlas_audit.json) | atlas 选择规则、计数、文件大小和 SHA-256 | 历史审计；当前四个输出与记录值一致 |

### historical

| 文件 | 用途 | 状态与审计 |
|---|---|---|
| [INITIAL_RESULTS.md](historical/INITIAL_RESULTS.md) | CIFAKE 32×32 的 B0/B1/B2/M2 pilot | 历史；正文完整，旧 outputs 路径不再有效 |
| [pilot_comparison.csv](historical/pilot_comparison.csv) | CIFAKE 四模型聚合指标 | 历史支持产物；4 条模型记录与 Markdown 表一致 |
| [SIDSET_B0_B1_B2_M2_M3_SUMMARY.md](historical/SIDSET_B0_B1_B2_M2_M3_SUMMARY.md) | SID-Set 五模型、格式去偏与 strict-six 原型总结 | 历史；正文可读，27 个底层证据链接因清理失效 |

## 文件约定

- 大写文件名通常是面向阅读者的主报告；小写 CSV/JSON/TSV 是机器可读支持产物。
- artifact.json 是便携 HTML 的规范输入；notes.json 记录报告契约与限制；audit.json 记录协议和完整性；delivery_receipt.log 记录交付验证。
- Markdown/HTML 是叙述层，精确数值优先回溯同目录 CSV、artifact、audit 和冻结 manifest。
- family-seen、exact-generator-seen 和 strict-unseen 是不同协议角色，不能因名称相近而合并。
- 历史报告不保证原始数据、逐图 predictions 或 checkpoint 仍保留。
- 新报告应放入现有语义目录或新增版本化子目录，不要再次堆放在 reports 根目录；完成后同步更新本索引。

## 重新生成入口

所有 Python、图像处理和报告构建必须在 SLURM Compute Node 上执行。提交前先查询当前 QoS；以下是已纳入项目的入口：

~~~bash
sbatch scripts/slurm/report_community_forensics_v3_evaluation.sbatch
sbatch scripts/slurm/report_community_forensics_train_v2_v3_unseen.sbatch
sbatch scripts/slurm/report_community_forensics_train_v1_v2_comparison.sbatch
sbatch scripts/slurm/report_community_forensics_robustness_v2.sbatch
sbatch scripts/slurm/report_community_forensics_unseen_accuracy.sbatch
sbatch scripts/slurm/report_community_forensics_data_statistics.sbatch
sbatch scripts/slurm/build_external_split_report.sbatch
sbatch scripts/slurm/build_community_forensics_exact_generator_atlas.sbatch
sbatch scripts/slurm/export_error_analysis_examples.sbatch
~~~

重新生成并不保证成功：历史报告引用的 checkpoint、逐图 predictions 或原始数据可能已经清理。应先核对对应 notes/audit 中的 manifest、checkpoint、矩阵和 SHA-256，再决定是否重建。
