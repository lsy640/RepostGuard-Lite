# Community Forensics train-v3 Error Analysis Note

## 结论

当前 train-v3 的首选模型 M2 在完整 4,000 张 strict unseen-generator 测试集上取得 0.9308 Clean AUROC 和 85.78% Accuracy，但错误并非均匀分布：Clean false positives 主要集中于 LAION real，false negatives 主要集中于 Stable Cascade 与 DALL·E 2。六阶段共同扰动会同时增加两类错误，FP 从 334 增至 432，FN 从 235 增至 507。

因此，M2 适合作为当前综合基线，但不能把其 `pred` 当作已经校准的真实概率，也不适合直接用于严格低误报场景。部署前需要使用与目标平台同分布、且与本测试集隔离的 calibration set，根据可接受的 FP/FN 成本重新冻结阈值。

## 分析范围

| 项目 | 冻结定义 |
|---|---|
| 主要分析模型 | M2；train-v3 当前综合推荐模型 |
| 对照模型 | M3、B2，仅用于说明架构与操作点权衡 |
| 测试集 | `unseen_generator_expanded`，4,000 张，Real/AIGI 各 2,000 张 |
| 测试角色 | External strict unseen-generator test；不参与 checkpoint、阈值或模型选择 |
| 生成器覆盖 | 12 个训练未见精确生成器 |
| 真实来源 | COCO、FFHQ、LAION、RAISE，各 500 张 |
| 条件 | Clean + 17 个原始扰动 + 4-stage A + 4-stage B + 6-stage，共 21 个 |
| M2 冻结阈值 | `0.99658203125`，来自内部 Clean validation |
| 判定规则 | `pred >= threshold` 判为 AIGI，否则判为 Real |
| 正式评测作业 | `32885` |
| 数据快照 | 2026-08-30 完成的冻结评测产物 |

测试 manifest SHA256：`59ca2e4ca966dac9fa4fb55281153f93e5becdd3e25da83bc2dff3fad36126cd`

扰动矩阵 SHA256：`69531f3f7111651808c99f14f89723bf631345878b1cbd0cbe0eee8531dde83c`

以下样本级结论均来自冻结的逐图预测，没有重新推理，也没有利用测试标签调整阈值。

## 方法与数据完整性

本报告以 `sample_id` 将 M2 的冻结 `predictions.jsonl` 与测试 manifest 静态连接，在每个条件内使用同一个 validation 阈值重建 FP/FN 判定，再分别按 Real 来源和 AIGI 精确生成器统计错误率。代表性个案从高 AIGI-score FP、低 AIGI-score FN 以及跨条件持续/新增错误中挑选，人工观察仅针对 Clean 原图。

静态完整性核验结果：

- 预测共 84,000 条，恰好等于 21 个条件 × 4,000 个样本；
- 84,000 个 `(condition, sample_id)` 组合全部唯一，每个条件均为 4,000 条；
- manifest 含 4,000 个唯一 `sample_id`，预测与 manifest 双向连接覆盖率均为 100%；
- 标签只包含 0/1，`pred` 均在 `[0,1]`，每个条件均为 2,000 Real + 2,000 AIGI；
- FP/FN 分层统计的分母始终是当前条件内对应 source/generator 的样本数，而不是把 21 次重复测量当作独立图片。

本 Error Analysis Note 没有另画聚合指标图：Clean 与 transformed 的总体可视化已由配套 Robustness Summary 提供；本报告的主要任务是保留逐样本分数、判定、来源和路径的精确可审计关系，使用表格比重复总体图更合适。原始数据不进入 Git；下文仅嵌入 7 张由冻结 Clean 原图生成的压缩缩略图，并通过 [`error_analysis_examples.tsv`](../assets/error_analysis/error_analysis_examples.tsv) 记录原图 SHA-256、缩略图 SHA-256 和样本血缘。

## 总体错误变化

| 条件 | AUROC | Accuracy/BA | FP / 2,000 Real | FPR | FN / 2,000 AIGI | FNR | Recall | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Clean | 0.9308 | 85.78% | 334 | 16.70% | 235 | 11.75% | 88.25% | 83.30% |
| 4-stage A platform repost | 0.9153 | 83.38% | 408 | 20.40% | 257 | 12.85% | 87.15% | 79.60% |
| 4-stage B edit repost | 0.8877 | 79.98% | 327 | 16.35% | 474 | 23.70% | 76.30% | 83.65% |
| 6-stage random composition | 0.8525 | 76.53% | 432 | 21.60% | 507 | 25.35% | 74.65% | 78.40% |

两组四阶段处理呈现不同的错误权衡：4-stage A 主要增加真实图片误报；4-stage B 的 Specificity 略高于 Clean，但 Recall 下降 11.95 个百分点。六阶段则同时损伤两侧类别。

Clean 与六阶段的样本配对结果进一步说明，新增错误不只是原有错误的重复：

| 错误类型 | Clean 与六阶段都错 | 仅 Clean 错 | 六阶段新增错误 | 两者都正确 |
|---|---:|---:|---:|---:|
| False positive（Real） | 240 | 94 | 192 | 1,474 |
| False negative（AIGI） | 185 | 50 | 322 | 1,443 |

### 不同扰动会把错误推向不同类别

原 17 种扰动也不是统一地降低分数。以下条件展示了最明显的操作点方向差异：

| 条件 | FP | FPR | FN | FNR | 相对 Clean 的主要变化 |
|---|---:|---:|---:|---:|---|
| Clean | 334 | 16.70% | 235 | 11.75% | 参考操作点 |
| Gaussian blur σ=2.0 | 536 | 26.80% | 194 | 9.70% | 分数整体更偏向 AIGI，增加 FP、减少 FN |
| Crop 0.8 + JPEG Q50 | 483 | 24.15% | 167 | 8.35% | 同样主要增加 FP |
| Gaussian noise σ=0.02 | 275 | 13.75% | 339 | 16.95% | 分数更偏向 Real，减少 FP、增加 FN |
| Color jitter 1.2/1.2/1.2 | 291 | 14.55% | 335 | 16.75% | 主要增加 FN |
| 4-stage B edit repost | 327 | 16.35% | 474 | 23.70% | FP 接近 Clean，但 FN 大幅增加 |
| 6-stage random composition | 432 | 21.60% | 507 | 25.35% | 两类错误同时增加 |

因此，“鲁棒性下降”不能只用一个平均 Accuracy 概括。不同处理链会沿相反方向移动分数分布，固定阈值下的误报成本与漏报成本可能完全不同。

## 代表性 false positives

下表中的内容描述来自对 Clean 原图的人工查看，只用于帮助定位个案，不构成视觉属性导致错误的因果证据。分数均为 AIGI sigmoid score；括号内为冻结阈值下的判定。

| 代号 | Real 来源与样本 | 人工观察 | Clean | 4-stage A | 4-stage B | 6-stage |
|---|---|---|---:|---:|---:|---:|
| FP-L | LAION · `cf_external_unseen_v3_real_laion_26ee8390ffc10966` | 白底单物体商品图，背景和表面纹理平滑 | 1.0000（FP） | 1.0000（FP） | 1.0000（FP） | 1.0000（FP） |
| FP-C | COCO · `cf_test_external_unseen_generator_real_696ae99363416c3a` | 浅景深室内玩偶照片，主体边缘与背景较柔和 | 1.0000（FP） | 0.9995（FP） | 1.0000（FP） | 1.0000（FP） |
| FP-F | FFHQ · `cf_test_external_unseen_generator_real_7d9a62f010d3fef7` | 近距离人像、太阳镜、高曝光皮肤与高对比发丝 | 0.9995（FP） | 0.9971（FP） | 0.9897（正确） | 0.9922（正确） |

以下均为 **Clean 原图缩略图**；点击图片可查看报告资产中的较大版本。它们用于把表中代号与具体个案对应起来，不代表各来源中最常见的错误形态。

<table>
  <tr>
    <td align="center"><a href="../assets/error_analysis/fp_l_laion.jpg"><img src="../assets/error_analysis/fp_l_laion.jpg" alt="FP-L：LAION 真实商品图被 M2 高置信误报为 AIGI" width="280"></a></td>
    <td align="center"><a href="../assets/error_analysis/fp_c_coco.jpg"><img src="../assets/error_analysis/fp_c_coco.jpg" alt="FP-C：COCO 真实玩偶照片被 M2 高置信误报为 AIGI" width="280"></a></td>
    <td align="center"><a href="../assets/error_analysis/fp_f_ffhq.jpg"><img src="../assets/error_analysis/fp_f_ffhq.jpg" alt="FP-F：FFHQ 真实人像被 M2 误报为 AIGI" width="280"></a></td>
  </tr>
  <tr>
    <td align="center"><strong>FP-L · LAION</strong><br>Clean score 1.0000</td>
    <td align="center"><strong>FP-C · COCO</strong><br>Clean score 1.0000</td>
    <td align="center"><strong>FP-F · FFHQ</strong><br>Clean score 0.9995</td>
  </tr>
</table>

FP-L 与 FP-C 在四种条件下持续被高置信误判，说明这些个案不是只由某一个新增处理链触发。FP-F 则在 4-stage B 与六阶段回到阈值下方，说明后处理并不总是单向增加 AIGI score。

完整 manifest 相对路径：

```text
test_external_unseen_v3_additions/real/cf_external_unseen_v3_real_laion_26ee8390ffc10966_00002615.jpeg.jpg
test_external_unseen_generator/real/cf_test_external_unseen_generator_real_696ae99363416c3a_000000093076.jpg.jpg
test_external_unseen_generator/real/cf_test_external_unseen_generator_real_7d9a62f010d3fef7_55660.png.png
```

## 代表性 false negatives

| 代号 | AIGI 生成器与样本 | 人工观察 | Clean | 4-stage A | 4-stage B | 6-stage |
|---|---|---|---:|---:|---:|---:|
| FN-F | Firefly Image 2 · `cf_test_external_unseen_generator_aigi_e443b5a1c729d1e4` | 高细节历史建筑立面，整体构图接近建筑摄影 | 0.7700（FN） | 0.7759（FN） | 0.8838（FN） | 0.9624（FN） |
| FN-D | DALL·E 2 · `cf_test_external_unseen_generator_aigi_6b5b17c0a5ce4791` | 蓝天、云层、草坡与树林组成的自然风景 | 0.7798（FN） | 0.8657（FN） | 0.9263（FN） | 0.9644（FN） |
| FN-S | Stable Cascade · `cf_external_unseen_v3_aigi_other_689deeb7f92c7a13` | 海面和云层场景，主体少且包含大面积平滑渐变 | 0.8711（FN） | 0.9570（FN） | 0.9194（FN） | 0.9385（FN） |
| FN-I | Imagen 3 · `cf_external_unseen_v3_aigi_commercial_9b00d2d4e402ab3b` | 瀑布与城市天际线的长曝光风景 | 0.9966（正确） | 0.9980（正确） | 0.9932（FN） | 0.7012（FN） |

前三张是 Clean 与三组多阶段条件下持续漏报的案例；FN-I 则用于展示由处理链诱发的漏报。图片仍为 Clean 原图缩略图，不能据此目测 transformed 条件中的像素变化。

<table>
  <tr>
    <td align="center"><a href="../assets/error_analysis/fn_f_firefly_image2.jpg"><img src="../assets/error_analysis/fn_f_firefly_image2.jpg" alt="FN-F：Firefly Image 2 建筑图被 M2 漏报为 Real" width="300"></a></td>
    <td align="center"><a href="../assets/error_analysis/fn_d_dalle2.jpg"><img src="../assets/error_analysis/fn_d_dalle2.jpg" alt="FN-D：DALL-E 2 自然风景被 M2 漏报为 Real" width="300"></a></td>
  </tr>
  <tr>
    <td align="center"><strong>FN-F · Firefly Image 2</strong><br>Clean score 0.7700</td>
    <td align="center"><strong>FN-D · DALL·E 2</strong><br>Clean score 0.7798</td>
  </tr>
  <tr>
    <td align="center"><a href="../assets/error_analysis/fn_s_stable_cascade.jpg"><img src="../assets/error_analysis/fn_s_stable_cascade.jpg" alt="FN-S：Stable Cascade 海面云层图被 M2 漏报为 Real" width="300"></a></td>
    <td align="center"><a href="../assets/error_analysis/fn_i_imagen3.jpg"><img src="../assets/error_analysis/fn_i_imagen3.jpg" alt="FN-I：Imagen 3 瀑布城市图在多阶段处理后被 M2 漏报为 Real" width="300"></a></td>
  </tr>
  <tr>
    <td align="center"><strong>FN-S · Stable Cascade</strong><br>Clean score 0.8711</td>
    <td align="center"><strong>FN-I · Imagen 3</strong><br>Clean 0.9966；6-stage 0.7012</td>
  </tr>
</table>

前三个案例在 Clean 和三组多阶段处理下都保持 FN，代表模型对高度摄影化的建筑或自然风景存在持续盲点。FN-I 在 Clean 时恰好达到阈值，但六阶段后下降至 0.7012，是处理链诱发漏报的代表案例。

完整 manifest 相对路径：

```text
test_external_unseen_generator/aigi/cf_test_external_unseen_generator_aigi_e443b5a1c729d1e4_00000068_1.jpg.jpg
test_external_unseen_generator/aigi/cf_test_external_unseen_generator_aigi_6b5b17c0a5ce4791_00000550.png.png
test_external_unseen_v3_additions/aigi/cf_external_unseen_v3_aigi_other_689deeb7f92c7a13_000312.png.png
test_external_unseen_v3_additions/aigi/cf_external_unseen_v3_aigi_commercial_9b00d2d4e402ab3b_00000184_0.png.png
```

这些观察不能证明“风景”“平滑区域”或“摄影化构图”是错误原因。内容类别、图像格式、原始分辨率、生成器架构、数据来源和历史处理链均是竞争解释，需要受控对照数据才能区分。

## 错误集中切片

### Real 来源的 false-positive rate

| Real 来源 | Clean | 4-stage A | 4-stage B | 6-stage |
|---|---:|---:|---:|---:|
| LAION | **52.40%** | **57.60%** | **45.80%** | **57.80%** |
| COCO | 8.40% | 14.40% | 11.00% | 16.00% |
| FFHQ | 3.80% | 2.00% | 0.20% | 0.40% |
| RAISE | 2.20% | 7.60% | 8.40% | 12.20% |

LAION 在四种条件下都是主要 FP 来源。该结果只能表述为来源级错误集中；LAION 的内容分布、网页图像处理、分辨率与格式构成可能同时影响结果，不能仅由该表归因于某一种风格。

### AIGI 生成器的 false-negative rate

下表列出最困难或最具对照意义的生成器；每个生成器包含 166–168 张图片。

| 精确生成器 | Clean | 4-stage A | 4-stage B | 6-stage |
|---|---:|---:|---:|---:|
| Stable Cascade | **37.95%** | **47.59%** | **54.22%** | **54.82%** |
| DALL·E 2 | 27.11% | 25.90% | 35.54% | 27.71% |
| Ideogram V2 | 15.06% | 14.46% | 30.72% | 32.53% |
| Imagen 3 | 10.18% | 7.19% | 23.35% | 29.34% |
| Midjourney V5.2 | 8.93% | 7.74% | 23.81% | 27.38% |
| DALL·E 3 | 0.60% | 2.40% | 6.59% | 10.18% |

Stable Cascade 是四种条件下最稳定的 FN 集中点；DALL·E 3 则是相对容易的对照生成器。新增组合处理对不同生成器的影响并不一致，因此单一总体 Recall 会隐藏生成器间差异。

## 方法权衡

### 1. 广泛 strict-unseen 泛化与困难生成器盲区

M2 在完整 strict unseen 上明显领先 B2，但在 Hourglass、DFGAN、GALIP 三个困难切片上，B2 的排序更好：

| 目标 | B2 | M2 | 解释 |
|---|---:|---:|---|
| Full strict-unseen Clean AUROC | 0.7707 | **0.9308** | M2 的广泛未见生成器泛化更强 |
| Hard Hourglass Clean AUROC | **0.7343** | 0.4588 | B2 对该困难生成器排序更稳定 |
| Hard DFGAN Clean AUROC | **0.6856** | 0.4614 | M2 Recall 仅 12.80% |
| Hard GALIP Clean AUROC | **0.7586** | 0.5460 | 两种方法的冻结阈值 Recall 都偏低 |

这些困难生成器在 train-v3 谱系下属于 exact-seen intervention slice，并共享真实负类面板；它们是相关的困难诊断，不是三个独立 strict-unseen 总体。当前不存在一个模型同时统治所有测试角色。

### 2. 排序性能与冻结阈值性能

B2 的 Clean AUROC 为 0.7707，20 个 transformed 条件均值反而为 0.7722；但 Accuracy/BA 从 69.80% 降至 59.83%，六阶段 Recall 仅 26.85%、Specificity 为 90.60%。这说明扰动后排序可以保留，而分数分布相对冻结阈值发生偏移，系统用大量新增 FN 换取更少 FP。

因此，AUROC/AP 与冻结阈值下的 Recall、Specificity、F1 必须联合报告。不能因为 AUROC 不降就宣称部署性能未受影响。

### 3. M2 与 M3 的融合策略

在 full strict-unseen Clean 上，M2/M3 AUROC 为 0.9308/0.9305，Recall 为 88.25%/87.40%，Specificity 同为 83.30%；六阶段 AUROC 为 0.8525/0.8489。当前 train-v3 下 M2 略优。

但 M3 并非普遍无效：它在历史 SID-Set 和 train-v2 设置中相对 M2 有一致优势，且在 train-v3 external exact-seen 上 AUROC 为 0.8578，略高于 M2 的 0.8558。现有门控消融只表明逐样本动态门控在当前 train-v3 checkpoint 上没有稳定总体增益，更合理的结论是其收益依赖训练数据规模和生成器多样性。

### 4. 低 FPR 与分数校准

M2/M3 的 Clean AUROC 都约为 0.931，但 TPR@1%FPR 均为 0%；到 5% FPR 才分别达到 45.4%/45.3%。这不满足严格低误报部署目标。

M2 Clean 的 Brier score 为 0.3999、ECE-15 为 0.4395；2,000 张 Real 中有 37 张得到精确的 `pred=1.0`，其中包括高置信 FP。冻结阈值又高达 0.9966。因此 `pred` 应解释为排序或置信分数，而不是“图片由 AIGC 生成的已校准概率”。

### 5. 测试先验与业务成本

当前测试集人为保持 50% AIGI。Accuracy、Precision、NPV 和最佳阈值都会随真实平台先验及 FP/FN 成本变化。项目尚未给出可接受 FPR、人工复核容量或漏检成本，因此本报告不能据此提供最终生产阈值。

## 建议的改进顺序

1. 保持当前 strict-unseen 与困难切片冻结，从新的、互斥的目标平台数据建立 calibration/development set。
2. 按目标 FPR 报告 TPR，并在多个预注册扰动种子下给出配对置信区间；不要使用本测试集重新选阈值。
3. 在新的训练或 development 数据中增加 LAION-like Real hard negatives，以及 Stable Cascade、DALL·E 2、Hourglass、DFGAN、GALIP 类困难正例；通过内容、格式、分辨率匹配的对照抽样区分真正成因。
4. 对 M2/M3 使用共享初始化和多个训练 seed，在受控的数据规模与生成器多样性阶梯上重做融合消融。
5. 在部署接口中明确 `pred` 未校准，并考虑设置“不确定/人工复核”区间，而不是强制每张图像立即二分类。

## 尚待回答的问题

1. 目标平台可接受的 FPR、漏检成本和人工复核容量分别是多少？这些约束将决定阈值与模型选择，而不是当前 50/50 测试集上的 Accuracy。
2. LAION FP 集中在控制内容类别、分辨率、格式和历史压缩后是否仍存在？
3. Stable Cascade 与 DALL·E 2 的 FN 是由特定内容构成、生成器特征弱化，还是训练覆盖不足造成？
4. 多个预注册六阶段随机种子是否保持相同的 source/generator 错误排序？
5. 在共享初始化、多 seed 与受控数据多样性阶梯下，M3 门控何时能够稳定优于 M2？

## 证据边界

- 当前模型各只有一个训练 seed；M2/M3 的小差异不构成统计显著性证明。
- 代表性图片是人工挑选的极端或持续错误案例，不是随机样本，也不估计错误类型的流行率。
- source/generator 分层结果是描述性统计；同一来源内仍混有内容、格式、分辨率和处理链差异。
- 六阶段只使用一个固定种子；不能视为所有平台处理链的概率分布。
- 测试标签没有用于模型、checkpoint 或阈值选择，也不应在后续训练中直接把这些测试图片提升为 hard examples。
- 原始数据不提交到 Git；样本路径只有在按 manifest 重新下载数据后才能本地解析。

## 可追溯证据

```text
data/manifests/community_forensics_test_external_unseen_generator_v3_expanded.csv
outputs/community_forensics_v3_robustness_v2/m2/unseen_generator_expanded/
├── COMPLETE
├── metrics_by_transform.csv
├── predictions.jsonl
├── run_card.json
└── summary.json

outputs/community_forensics_v3_robustness_v2/m3/unseen_generator_expanded/
reports/summaries/COMMUNITY_FORENSICS_V3_EVALUATION_REPORT.md
reports/summaries/COMMUNITY_FORENSICS_V3_ROBUSTNESS_EVALUATION_SUMMARY.md
reports/summaries/COMMUNITY_FORENSICS_TRAIN_V2_V3_UNSEEN_INTERSECTION_COMPARISON.md
```

完整的 Clean 与 transformed 汇总见 [`COMMUNITY_FORENSICS_V3_ROBUSTNESS_EVALUATION_SUMMARY.md`](COMMUNITY_FORENSICS_V3_ROBUSTNESS_EVALUATION_SUMMARY.md)。
