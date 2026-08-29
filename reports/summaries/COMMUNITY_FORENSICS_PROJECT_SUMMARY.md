# RepostGuard-Lite 模型、数据与阶段性验证总结

> 状态日期：2026-08-29
>
> 当前主实验：Community Forensics 24k + validation-v2 + robustness-v2
>
> 历史原型：CIFAKE、SID-Set（原始数据和 checkpoint 已清理，仅保留报告）

## 1. 项目结论概览

RepostGuard-Lite 面向“真实图像 / 全合成图像”二分类，重点检验模型在平台重编码、缩放、模糊、噪声、颜色变化、裁剪以及多阶段复合扰动下，能否继续识别未见过的精确生成器。

当前项目已经形成三层互补评测：

1. **内部 checkpoint 选择**：CommunityForensics-Small 的 `val_unseen_generator`，只用于选 checkpoint 和冻结阈值；
2. **外部分布诊断**：精确生成器已见、同大类但精确生成器未见、跨大类未见，以及 Hourglass、DFGAN、GALIP 三个困难生成器切片；
3. **鲁棒性评测**：clean、17 个原有扰动和 3 个新增多阶段复合扰动，共 21 个条件，外部 6 个切片 × 5 个模型共 630 个模型—切片—条件单元。

最重要的结果是：

- **M3 是当前 strict unseen-generator 的最佳固定阈值检测器**：clean Accuracy 79.30%、F1 80.12%、AIGI Recall 83.40%、AUROC 0.8631、AP 0.8206；20 个扰动条件平均 Accuracy 76.90%，最差为 71.05%。
- **M2 更偏保守**：strict unseen clean Precision 77.93%、Real Specificity 77.20%，均略高于 M3；M3 用更高召回换取了约 2 个百分点的特异率。
- **B2 的跨生成器排序泛化最好**：在 6 个外部切片上的 clean 宏平均 AUROC 为 0.7315，且在 Hourglass、DFGAN、GALIP 上分别达到 0.7271、0.6864、0.7488，显著高于 M2/M3。但 B2 的内部冻结阈值在 strict unseen 上 Recall 只有 48.30%，六阶段扰动下固定阈值召回进一步下降，因此“AUROC 最强”不等于“当前部署阈值最好”。
- **M2/M3 在内部验证和 strict unseen 上强，但困难生成器存在明显盲区**：这说明融合取证分支提高了目标域性能，却没有自动保证对所有未知生成机制都泛化。
- **B1 的增强策略确实改善了内部扰动鲁棒性**，但没有在所有外部分布上稳定优于 B0；不能把一次内部改进解释成普适泛化提升。

## 2. 阶段演进

| 阶段 | 数据与目的 | 主要改进 | 结果与局限 |
|---|---|---|---|
| CIFAKE 初始原型 | 10k train + 2k validation，32×32，单一 SD1.4 合成来源 | 建立 B0/B1/B2/M2 训练、评测、checkpoint 和 SLURM 流程 | 流程可运行；数据分辨率低、合成来源单一，只能作为 pipeline pilot |
| SID-Set 高分辨率原型 | 20k train + 4k validation，仅真实与全合成类别 | 加入 M3；发现并修复 JPEG/MPO 与 PNG 的标签—格式捷径；所有去偏在读取时完成 | 单一验证集上指标很高，但缺少独立外部生成器测试；原始数据和 checkpoint 已清理 |
| Community Forensics 24k | 18k train + 2k internal val + 两个 2k external test | 900 个训练精确生成器；等生成器采样；冻结 exact/family exposure；完整 SHA-256/pHash 审计 | 当前模型训练的主数据与主 checkpoint 谱系 |
| validation-v2 | 2k exact-seen + Hourglass/DFGAN/GALIP 各 500 | 首次把 exact-seen、family-seen/exact-unseen 和 family-unseen 明确分开 | 揭示 M2/M3 在困难精确生成器上的盲区 |
| robustness-v2 | 外部 6 切片 × 21 条件 × 5 模型 | 在原 17 个扰动外增加两组四阶段和一组六阶段复合扰动 | 给出当前最完整的外部鲁棒性结论 |
| strict unseen 详细指标 | strict unseen-generator 的 clean + 20 扰动 | 增加 Accuracy、Precision、Recall、Specificity、F1、MCC、AP、置信区间、切片指标；修正 ROC/PR 公共绘图网格 | 支持固定阈值部署判断，而不只比较 AUROC |

## 3. 当前模型

五个模型都从公开视觉预训练底座独立训练二分类器，并没有从 CIFAKE、SID-Set 检测 checkpoint 继续微调；作业恢复仅用于恢复同一 Community Forensics 训练的完整状态。

| 模型 | 主干与分类头 | 可训练参数 | 训练策略 | 设计目标 |
|---|---|---:|---|---|
| B0 | ImageNet 预训练 EfficientNet-B0；Dropout(0.2) + Linear(1280,1) | 4,008,829 | BCE；无额外鲁棒增强 | 端到端视觉基线 |
| B1 | 与 B0 相同 | 4,008,829 | BCE；类别对称的单/双扰动增强 | 检验数据增强的作用 |
| B2 | 冻结 OpenCLIP ViT-B/32（LAION-2B）+ Linear(512,1) | 513 / 总计 87,849,729 | BCE；只训练线性头 | 检验通用语义表征的迁移能力 |
| M2 | 冻结 CLIP 语义分支 + DCT/SRM-inspired/NPR ResNet-18 取证分支 + 融合头 | 11,574,226 / 总计 99,423,442 | clean/degraded 双视图 BCE + KL 一致性 + 特征一致性 | 融合语义与低层取证线索 |
| M3 | M2 + 六维无标签质量特征门控 | 11,574,528 / 总计 99,423,744 | 与 M2 相同 | 根据图像质量动态调节语义/取证权重 |

### 3.1 B0 与 B1

B0 对 ImageNet 预训练 EfficientNet-B0 全量微调。B1 不改变模型容量，只改变训练输入分布，因此 B1/B0 是当前最直接的数据增强消融。

B1 的训练增强按 `clean 0.25 / 单扰动 0.50 / 双扰动 0.25` 采样，并且对真实与 AIGI 使用完全相同的扰动规则。扰动包括 JPEG、Gaussian blur、缩放、Gaussian noise、颜色变化和中心裁剪，避免增强本身泄露标签。

### 3.2 B2

B2 冻结 OpenCLIP ViT-B/32 图像塔，只训练 512→1 的线性分类头。它的优势是用大规模图文预训练获得更通用的语义排序能力；代价是分类头容量很小，而且内部验证集确定的固定阈值可能不能直接适配新的生成器分布。

### 3.3 M2

M2 包含两条路径：

- **语义分支**：冻结 CLIP 512 维特征，经 LayerNorm、Linear 和 GELU 投影到 256 维；
- **取证分支**：把 224×224 图像划分为 4×4 个 56×56 patch，按 DCT 能量选择两个低频和两个高频 patch；为每个 patch 构造 RGB、30 通道确定性 SRM-inspired 5×5 高通残差和 3 通道 NPR 特征，再经通道压缩与 ResNet-18 编码、patch embedding 和注意力池化得到 256 维表示。

两条 256 维表示拼接后经 LayerNorm、Linear、GELU、Dropout 和二分类头输出 logit。

M2/M3 的 paired loss 为：

```text
clean BCE
+ degraded BCE
+ 0.50 × symmetric Bernoulli KL
+ 0.25 × cosine feature inconsistency
```

### 3.4 M3

M3 在 M2 上增加一个 302 参数的质量门控。门控输入均为无标签、确定性的质量统计：梯度强度、Laplacian、JPEG blockiness、高频噪声、有效分辨率 proxy 和亮度动态范围。六维特征经小型 MLP 输出语义/取证两个权重。

门控最后一层零初始化，并把 softmax 权重乘 2，因此训练起点与 M2 的等权融合一致。质量特征只决定两个分支权重，不直接进入分类器，降低其成为新标签捷径的风险。

## 4. 当前训练方法与可复现配置

### 4.1 通用设置

- 随机种子：`20260828`，确定性运行；
- 输入：RGB，224×224；
- 训练轮数：3；
- 优化器：AdamW，cosine learning-rate schedule，5% warm-up，最低学习率比例 0.05；
- AMP：启用；梯度裁剪：1.0；
- worker：6；
- checkpoint：每 250 step 保存，支持完整 optimizer/scheduler/scaler/RNG 状态恢复和原子写入；
- checkpoint 选择：内部 `val_unseen_generator` clean AUROC；
- 决策阈值：在内部 clean validation 上最大化 balanced accuracy，此后对外部 clean 和所有扰动保持冻结。

### 4.2 采样与格式去偏

训练使用逆组频率平衡采样。AIGI 按 `(label, source_dataset, exact generator)` 分组，真实图像按 `(label, source_dataset, real)` 分组，从而避免大生成器单独支配训练。

所有当前模型都使用在线格式去偏，不产生第二份图像数据：

1. 解码为 RGB 并丢弃容器元数据；
2. bicubic 统一到 224×224，破坏原有 JPEG block grid；
3. 所有类别执行相同 JPEG roundtrip；
4. 训练质量从 70/80/90/95 随机采样，评测固定为 90；
5. 再叠加 B1/M2/M3 的模型特定增强。

该策略降低格式捷径，但不能证明完全抹除历史压缩痕迹。

### 4.3 各模型训练配置与谱系

| 模型 | Batch / 累积 | 有效 Batch | LR | Weight decay | 训练 step | 训练 job | 基础评测 job | checkpoint SHA-256 前 12 位 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| B0 | 128 / 1 | 128 | 3e-4 | 1e-4 | 420 | 32446 | 32453 | `cf077cbfe1fa` |
| B1 | 128 / 1 | 128 | 3e-4 | 1e-4 | 420 | 32455 | 32456 | `b02a7fe98221` |
| B2 | 96 / 1 | 96 | 1e-3 | 1e-5 | 560 | 32457 | 32459 | `ff89953d5c0c` |
| M2 | 24 / 2 | 48 | 2e-4 | 1e-4 | 1,125 | 32460 | 32463 | `75be44fc704b` |
| M3 | 24 / 2 | 48 | 2e-4 | 1e-4 | 1,125 | 32464 | 32465 | `32f81545af4a` |

训练与评测在 TC2 Compute Node 上运行，记录环境为 Python 3.11.16、PyTorch 2.5.1、torchvision 0.20.1、CUDA 12.1 和单张 NVIDIA A40。

## 5. 当前数据集

### 5.1 来源与冻结版本

- `OwensLab/CommunityForensics-Small`：训练与内部验证；冻结 revision `6c539...`；
- `OwensLab/CommunityForensics-Eval`：外部 seen-family、strict unseen 和困难切片；冻结 revision `7d4a...`；
- `TheKernel01/AIGIBench`：exact-seen 的 Stable Diffusion 1.4 AIGI；冻结 revision `f125eabc5ac34a4729d74adc1aa1214540f91947`；
- exact-seen 真实类来自 ImageNet；
- Community Forensics 数据许可记录为 CC-BY-NC-SA-4.0。

### 5.2 八个冻结 manifest

| Manifest / 用途 | Real | AIGI | 精确生成器 | 生成器大类 | 与训练关系 |
|---|---:|---:|---:|---:|---|
| train | 9,000 | 9,000 | 900 | 3 | 训练 |
| internal `val_unseen_generator` | 1,000 | 1,000 | 100 | 1 | 精确生成器与 train 不重合；checkpoint 选择 |
| external seen-family | 1,000 | 1,000 | 9 | 3 | 大类见过，精确生成器完全未见 |
| external strict unseen-generator | 1,000 | 1,000 | 12 | 2 | 精确生成器与大类均未见 |
| external exact-seen | 1,000 | 1,000 | 1 | 1 | canonical SD1.4 精确身份在训练中出现 |
| hard Hourglass | 250 | 250 | 1 | 1 | 大类见过，困难精确生成器未见 |
| hard DFGAN | 250 | 250 | 1 | 1 | 大类见过，困难精确生成器未见 |
| hard GALIP | 250 | 250 | 1 | 1 | 大类见过，困难精确生成器未见 |

三个 hard manifest 共用同一组 250 张真实参考图，以便生成器间 AUROC 可直接比较；因此统计时必须区分 manifest 引用和唯一物理图像。

### 5.3 总量与完整性

- 基础 24k：18,000 train + 2,000 internal validation + 4,000 external test；
- validation-v2：新增 3,000 张唯一图像；
- 8 个 manifest 合计 27,500 条引用、27,000 张唯一图像；
- 唯一类别：13,250 real + 13,750 AIGI；
- 1,021 个唯一精确生成器、5 个生成器大类、6 个真实来源、4 种存储格式；
- 物理数据约 24.51 GiB；按 manifest 重复引用计算约 28.43 GiB；
- 由于 hard 三切片复用 250 张 real，共产生 500 条预期的重复引用。

冻结审计结果：路径缺失 0、大小不一致 0、split 内 SHA-256 重复 0、意外跨 split path/SHA 重叠 0、跨标签 SHA 重叠 0、基础 24k 的 pHash Hamming≤4 冲突 0。

TIFF 审计需精确表述：当前数据共有 594 张 TIFF，既有全解码检查只覆盖其中 500 张，500/500 有效、失败 0；validation-v2 后新增的 94 张 TIFF 尚未纳入同一次全解码审计。COCO val2017 / DALL-E Advanced 的 reserved-hash manifest 未提供，因此官方 split/source 规则已执行，但不能声称完成 reserved-image 哈希排除。

## 6. 评测协议

### 6.1 三种“见过”关系

- **Exact-seen**：训练和评测共享 canonical 精确生成器身份；
- **Seen-family / exact-unseen**：生成器技术大类在训练出现，但精确生成器身份完全不同；
- **Strict unseen-generator**：精确生成器身份和技术大类都没有在训练出现。

因此，seen-family 不能再被解释成 exact-seen。

### 6.2 21 个鲁棒性条件

每个外部切片评测 1 个 clean + 20 个扰动条件：

- JPEG：Q90、Q70、Q50、Q30；
- Gaussian blur：σ=0.5、1.0、2.0；
- resize：0.5× bicubic、0.25× bilinear；
- Gaussian noise：σ=0.02、0.05、0.10；
- color：0.8、1.2；
- center crop：0.8；
- 原有二阶段：resize 0.5 + JPEG70；crop 0.8 + JPEG50；
- 四阶段 A：crop 0.85 → resize 0.5 bicubic → blur 1.0 → JPEG50；
- 四阶段 B：color → resize 0.5 bilinear → noise 0.05 → JPEG50；
- 六阶段：crop → resize → color → blur → noise → JPEG；每个样本以固定 seed 从完整训练强度区间确定性采样。

所有外部指标使用内部验证冻结的概率阈值，禁止在测试切片上重新调阈值。

## 7. 当前验证与测试结果

### 7.1 内部 Small validation

| 模型 | Clean AUROC | Clean BA | 17 扰动平均 AUROC | 17 扰动平均 BA | 最差 AUROC | 最差条件 | 相对 clean AUROC 降幅 |
|---|---:|---:|---:|---:|---:|---|---:|
| B0 | 0.8991 | 0.8155 | 0.8571 | 0.7739 | 0.5926 | noise 0.10 | 0.0420 |
| B1 | 0.9021 | 0.8240 | 0.8751 | 0.7925 | 0.7556 | noise 0.10 | 0.0270 |
| B2 | 0.7224 | 0.6670 | 0.7052 | 0.5707 | 0.5999 | resize 0.5 + JPEG70 | 0.0172 |
| M2 | 0.9489 | **0.8905** | 0.9402 | 0.8743 | 0.9224 | resize 0.25 | **0.0087** |
| M3 | **0.9510** | 0.8865 | **0.9420** | **0.8749** | **0.9240** | resize 0.25 | 0.0090 |

这些指标参与 checkpoint 选择或内部诊断，不是独立外部测试结果。B1 相对 B0 的 17 扰动平均 AUROC 提升约 0.018，最差 AUROC 提升约 0.163；M2/M3 在该域明显领先。

### 7.2 六个外部切片的 clean AUROC

| 外部切片 | B0 | B1 | B2 | M2 | M3 |
|---|---:|---:|---:|---:|---:|
| Exact-seen SD1.4 | 0.6447 | 0.6420 | **0.7832** | 0.7250 | 0.7299 |
| Hard Hourglass | 0.2394 | 0.2552 | **0.7271** | 0.3633 | 0.3706 |
| Hard DFGAN | 0.4465 | 0.4255 | **0.6864** | 0.4053 | 0.3930 |
| Hard GALIP | 0.4180 | 0.4060 | **0.7488** | 0.4000 | 0.4083 |
| Seen-family / exact-unseen | 0.6160 | 0.5951 | **0.7209** | 0.6754 | 0.6840 |
| Strict unseen-generator | 0.7910 | 0.7595 | 0.7228 | 0.8564 | **0.8631** |
| **六切片宏平均** | 0.5260 | 0.5139 | **0.7315** | 0.5709 | 0.5748 |

B2 在广泛跨切片排序上最好；M2/M3 则在 strict unseen 主目标切片最好。Hourglass、DFGAN、GALIP 结果表明，当前取证融合分支的提升具有明显分布依赖性。

### 7.3 外部鲁棒性宏平均 AUROC

| 模型 | Clean | 原有 17 扰动平均 | 新增 3 个复合扰动平均 | 新增复合扰动最差 | 六阶段 |
|---|---:|---:|---:|---:|---:|
| B0 | 0.5260 | 0.5211 | 0.5225 | 0.2452 | 0.5226 |
| B1 | 0.5139 | 0.5132 | 0.5101 | 0.2833 | 0.5040 |
| B2 | **0.7315** | **0.7368** | **0.6783** | **0.5410** | **0.6228** |
| M2 | 0.5709 | 0.5732 | 0.5576 | 0.2975 | 0.5497 |
| M3 | 0.5748 | 0.5717 | 0.5526 | 0.3030 | 0.5431 |

这里的宏平均给予六个外部切片相同权重。B2 的全局最差情况和复合扰动排序最稳；M2/M3 的低宏平均主要由三个困难生成器切片拖累，而不是 strict unseen 上失效。

### 7.4 Strict unseen-generator clean 固定阈值指标

测试集为 1,000 real + 1,000 AIGI。AIGI 是正类。

| 模型 | Accuracy | Precision | Recall | Specificity | F1 | MCC | AUROC | AP | TN / FP / FN / TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| B0 | 0.7120 | 0.7345 | 0.6640 | 0.7600 | 0.6975 | 0.4260 | 0.7910 | 0.7478 | 760 / 240 / 336 / 664 |
| B1 | 0.7040 | 0.6835 | 0.7600 | 0.6480 | 0.7197 | 0.4106 | 0.7595 | 0.7073 | 648 / 352 / 240 / 760 |
| B2 | 0.6520 | 0.7296 | 0.4830 | **0.8210** | 0.5812 | 0.3230 | 0.7228 | 0.7122 | 821 / 179 / 517 / 483 |
| M2 | 0.7885 | **0.7793** | 0.8050 | 0.7720 | 0.7919 | 0.5773 | 0.8564 | 0.8164 | 772 / 228 / 195 / 805 |
| M3 | **0.7930** | 0.7708 | **0.8340** | 0.7520 | **0.8012** | **0.5880** | **0.8631** | **0.8206** | 752 / 248 / 166 / 834 |

95% stratified bootstrap 区间：

- B0：Accuracy 69.10%–73.15%，AUROC 0.7710–0.8100；
- B1：Accuracy 68.40%–72.45%，AUROC 0.7387–0.7810；
- B2：Accuracy 63.25%–67.05%，AUROC 0.7012–0.7438；
- M2：Accuracy 77.05%–80.60%，AUROC 0.8414–0.8727；
- M3：Accuracy 77.50%–81.05%，AUROC 0.8475–0.8783。

M3 相对 M2 的 clean AUROC 增加 0.0066、Accuracy 增加 0.45 个百分点、Recall 增加 2.9 个百分点，但 Specificity 降低 2.0 个百分点。两者区间高度重叠且尚未执行 paired significance test，因此这应解释为当前一次冻结实验中的小幅改进，而不是已证明的统计显著优势。

### 7.5 Strict unseen-generator 的 20 扰动汇总

| 模型 | 平均 Accuracy | 最差 Accuracy | 平均 F1 | 平均 Recall | 平均 Specificity | 平均 AUROC | 最差 AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.6541 | 0.5100 | 0.6296 | 0.5884 | 0.7197 | 0.7248 | 0.5025 |
| B1 | 0.6471 | 0.5305 | 0.6641 | 0.6959 | 0.5984 | 0.6929 | 0.5446 |
| B2 | 0.5651 | 0.5000 | 0.2573 | 0.1883 | 0.9420 | 0.7280 | 0.6217 |
| M2 | 0.7640 | 0.6995 | 0.7593 | 0.7491 | 0.7790 | 0.8374 | 0.7848 |
| M3 | **0.7690** | **0.7105** | **0.7721** | **0.7863** | 0.7517 | **0.8421** | **0.7902** |

B2 的 AUROC 仍有竞争力，却因冻结阈值产生“高特异率、低召回”的操作点；M3 在当前部署阈值下保持最好的整体平衡。

### 7.6 Seen-family 高于/低于 strict unseen 的解释

| 模型 | Seen-family AUROC | Strict unseen AUROC | Unseen − Seen | 95% bootstrap CI |
|---|---:|---:|---:|---|
| B0 | 0.6160 | 0.7910 | +0.1750 | [0.1456, 0.2053] |
| B1 | 0.5951 | 0.7595 | +0.1644 | [0.1285, 0.1965] |
| B2 | 0.7209 | 0.7228 | +0.0018 | [-0.0292, 0.0316] |
| M2 | 0.6754 | 0.8564 | +0.1810 | [0.1533, 0.2079] |
| M3 | 0.6840 | 0.8631 | +0.1790 | [0.1524, 0.2049] |

这不是“未知生成器天然更容易”的证据。两个切片的 AIGI 构成不同：seen-family 为 222 GAN、667 Latent Diffusion、111 Pixel Diffusion；strict unseen 为 917 Commercial、83 Other。真实类来源配比相同，但正类生成器机制、图像内容和数据来源仍是联合混杂因素。上述差异只能作描述性比较。

### 7.7 历史原型结果（仅作阶段对照）

以下结果解释项目为何继续扩展数据与评测协议，不应与当前 Community Forensics 外部测试直接横向排名。

**CIFAKE pilot：**

| 模型 | Clean AUROC | Clean BAcc | 扰动平均 AUROC | 扰动平均 BAcc | 最差 AUROC |
|---|---:|---:|---:|---:|---:|
| B0 | 0.9945 | 0.9705 | 0.8638 | 0.7448 | 0.5442 |
| B1 | 0.9929 | 0.9570 | **0.9730** | **0.9043** | **0.9120** |
| B2 | 0.9738 | 0.9220 | 0.9202 | 0.7693 | 0.7688 |
| M2 | 0.9904 | 0.9545 | 0.9641 | 0.8986 | 0.8971 |

CIFAKE 上 B1/M2 的扰动收益促成后续增强与双分支设计，但该数据只有单一合成来源且分辨率为 32×32。

**SID-Set 在线格式去偏后：**

| 模型 | Clean AUROC | 17 扰动平均 AUROC | 17 扰动最差 AUROC | Strict six AUROC | Strict six BAcc |
|---|---:|---:|---:|---:|---:|
| B0 | 0.999950 | 0.953356 | 0.469849 | 0.718696 | 0.510750 |
| B1 | 0.999933 | **0.999559** | 0.997098 | **0.991335** | **0.937750** |
| B2 | 0.998484 | 0.996342 | 0.984164 | 0.936252 | 0.618250 |
| M2 | 0.998046 | 0.996126 | 0.988386 | 0.961302 | 0.684500 |
| M3 | 0.999744 | 0.999402 | **0.998147** | 0.975617 | 0.739000 |

SID-Set 暴露了真实 JPEG/MPO、全合成 PNG 的严重格式混杂，推动了统一在线 JPEG roundtrip。即使去偏后结果很高，该阶段仍只在同一来源 validation 上评测，不能证明未知精确生成器泛化；这正是切换到 Community Forensics 并构建外部 exposure 切片的原因。

## 8. 各阶段改进的证据边界

### B0 → B1：增强改进了域内鲁棒性，但外部收益不稳定

- 内部 17 扰动平均 AUROC：0.8571 → 0.8751；
- 内部最差 AUROC：0.5926 → 0.7556；
- strict unseen clean AUROC：0.7910 → 0.7595，反而下降；
- strict unseen 最差扰动 AUROC有所提高，但平均 AUROC下降。

结论：增强显著缓解了训练定义中的严重噪声退化，但不保证跨数据来源泛化。

### B1 → B2：冻结语义表征提高了广域生成器排序

B2 的内部 Small AUROC不高，但在 exact-seen、三个 hard 和 seen-family 上都取得最高 clean AUROC，六切片 clean 宏平均达到 0.7315。这是当前最强的跨切片排序证据。其主要问题不是排序完全失败，而是内部阈值迁移：strict unseen clean Recall 只有 48.3%。

### B2 → M2：融合取证分支提高目标域效果，但出现困难生成器盲区

- 内部 clean AUROC：0.7224 → 0.9489；
- strict unseen clean AUROC：0.7228 → 0.8564；
- strict unseen Accuracy：65.20% → 78.85%；
- 但 Hourglass/DFGAN/GALIP clean AUROC 从 B2 的 0.686–0.749 降至 M2 的 0.363–0.405。

结论：M2 对当前 strict unseen 目标分布非常有效，但不能替代广域生成器诊断。

### M2 → M3：质量门控带来小幅、偏召回的改进

- 内部 clean AUROC：+0.0021；
- strict unseen clean AUROC：+0.0066；Accuracy：+0.45 个百分点；
- strict unseen 20 扰动平均 Accuracy：+0.50 个百分点；最差 Accuracy：+1.10 个百分点；
- Recall 提升、Specificity 略降；hard 三生成器没有一致提升。

结论：M3 是当前固定阈值主模型，但门控收益仍需多 seed、paired bootstrap/DeLong 或 generator-level hierarchical bootstrap 进一步确认。

### 数据与协议改进：比单一模型增益更关键

从 CIFAKE/SID-Set 转到 Community Forensics 的最大改进是评测语义被严格冻结：

- exact identity 与 family exposure 分开；
- checkpoint-selection validation 与外部 test 分开；
- 测试阈值冻结；
- SHA-256/pHash、revision、source locator 和替换 lineage 可追溯；
- 既报告 AUROC 排序，也报告固定阈值的 Accuracy/Precision/Recall/Specificity；
- 增加多阶段复合扰动和精确生成器困难切片。

## 9. 当前模型使用建议

| 使用场景 | 建议模型 | 原因 |
|---|---|---|
| 当前 strict unseen 主任务、固定阈值直接检测 | M3 | Accuracy、F1、Recall、MCC、AUROC、AP 和扰动均值整体最佳 |
| 更看重减少 real 误报 | M2 | Precision/Specificity 略高于 M3，可再做成本敏感阈值校准 |
| 未知来源广泛、优先排序或二阶段筛查 | B2 | 外部六切片宏平均 AUROC及困难切片最强，但必须重新解决阈值校准 |
| 轻量端到端部署 | B1 或 B0 | 约 4M 参数；B1 域内扰动更稳，但外部泛化需单独验证 |

不建议只用单一 aggregate AUROC 选择部署模型。实际使用至少应同时检查：目标切片 AUROC、冻结阈值 Recall/Specificity、困难生成器最差表现和复合扰动结果。

## 10. 局限与下一步

1. 目前每个架构主要是单一 seed，无法分离随机波动；
2. M2/M3 在 Hourglass、DFGAN、GALIP 上接近或低于随机排序，是必须修复的泛化缺口；
3. external seen/unseen 的生成器大类构成不同，跨切片差异存在内容与来源混杂；
4. B2 的排序能力与固定阈值性能不一致，需要不接触测试标签的域外校准方案；
5. 594 张 TIFF 中仍有 94 张未进入既有全解码专项审计；
6. reserved COCO/DALL-E Advanced 哈希清单缺失，保留排除结论尚不完整；
7. 格式去偏减少但不能证明消除所有历史编解码痕迹；
8. 下一轮应优先做多 seed、精确生成器分层 bootstrap、worst-generator 训练目标、门控消融和外部无标签校准。

## 11. 证据与产物索引

### 当前配置与代码

- 模型配置：[`configs/community_forensics/`](../../configs/community_forensics/)
- 鲁棒性矩阵：[`configs/community_forensics_robustness_v2.yaml`](../../configs/community_forensics_robustness_v2.yaml)
- 模型定义：[`src/repostguard/models/`](../../src/repostguard/models/)
- 数据读取与格式去偏：[`src/repostguard/data/dataset.py`](../../src/repostguard/data/dataset.py)、[`src/repostguard/data/transforms.py`](../../src/repostguard/data/transforms.py)
- 训练入口：[`src/repostguard/train.py`](../../src/repostguard/train.py)
- 评测入口：[`src/repostguard/evaluate.py`](../../src/repostguard/evaluate.py)

### 当前报告

- [Community Forensics 数据统计报告](../data_statistics/COMMUNITY_FORENSICS_DATA_STATISTICS.html)
- [外部 split 构成与 AUROC 诊断](../evaluations/external_split/community_forensics_external_split_diagnostic.html)
- [B0/B1/B2/M2/M3 robustness-v2 报告](../evaluations/robustness_v2/COMMUNITY_FORENSICS_B0_B1_B2_M2_M3_ROBUSTNESS_V2.html)
- [Strict unseen-generator 详细准确率报告](../evaluations/unseen_generator/COMMUNITY_FORENSICS_UNSEEN_GENERATOR_ACCURACY.html)
- [Strict unseen clean 指标 CSV](../evaluations/unseen_generator/community_forensics_unseen_generator_clean_metrics.csv)
- [Strict unseen 全条件指标 CSV](../evaluations/unseen_generator/community_forensics_unseen_generator_all_metrics.csv)
- [完整 robustness-v2 单元格 CSV](../evaluations/robustness_v2/community_forensics_robustness_v2_metrics.csv)

### 历史报告

- [CIFAKE 初始结果](../historical/INITIAL_RESULTS.md)
- [SID-Set B0/B1/B2/M2/M3 总结](../historical/SIDSET_B0_B1_B2_M2_M3_SUMMARY.md)

历史报告用于说明项目演进，不代表当前仍保留对应原始数据或 checkpoint。
