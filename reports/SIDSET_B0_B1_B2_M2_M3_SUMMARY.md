# SID-Set B0、B1、B2、M2、M3 实验总结

> 更新日期：2026-08-28  
> 实验状态：五组模型均已完成训练、18 条件基础验证和严格六重随机组合扰动验证  
> 结果性质：SID-Set validation 上的初步实验结果，不等同于独立最终测试集结果

## 1. 核心结论

- **B1 是当前综合性能和效率最好的基线**：17 种扰动的平均 AUROC 最高，为 0.999559；总参数仅 4.01M。
- **M3 的最坏条件表现最好**：最坏 AUROC 为 0.998147，略高于 B1，但模型明显更大，且质量特征存在间接携带标签信息的风险。
- **M2 改善了固定阈值下的鲁棒性，但没有超过 B2 的平均 AUROC**。
- **B0 的 clean 指标很高，但对强高斯噪声几乎完全失效**，说明不能只用 clean 指标判断模型鲁棒性。
- 在六种扰动全部随机叠加的 `strict_random_six` 条件下，**B1 仍以 0.991335 AUROC 和 0.937750 balanced accuracy 排名第一**；M3 排名第二，但固定阈值下的 AIGC recall 只有 0.478。
- 当前所有结果来自同一 SID-Set validation split 的 clean/扰动版本，尚未验证未见生成器或外部数据集泛化能力。

## 2. 数据集与共同输入流程

### 2.1 SID-Set 子集

- Hugging Face 数据集：`saberzl/SID_Set`
- 固定 revision：`dc03ead57929879319ce30a82bfcfb8d317b10bd`
- License：CC-BY-4.0
- 标签映射：
  - `0`：真实图片
  - `1`：全合成图片
  - `2`：局部篡改图片，当前实验排除
- 训练集：20,000 张
  - 真实图片：10,000 张
  - 全合成图片：10,000 张
- 验证集：4,000 张
  - 真实图片：2,000 张
  - 全合成图片：2,000 张
- 训练集仅来自 SID-Set 官方 train split。
- 验证集仅来自 SID-Set 官方 validation split。
- reserved sets 未使用。
- 训练—验证精确图像重叠为 0。

数据审计记录：[sidset_subset_audit.json](../data/manifests/sidset_subset_audit.json)。

### 2.2 原始格式偏差

原始训练数据存在明显的类别—文件格式相关性：

| 类别 | JPEG | MPO | PNG |
|---|---:|---:|---:|
| 真实图片 | 9,998 | 2 | 0 |
| 全合成图片 | 0 | 0 | 10,000 |

如果不处理，模型可能主要学习 PNG/JPEG 编码差异，而不是生成图像本身的特征。

### 2.3 在线格式去偏

五个模型使用完全相同的在线格式去偏流程：

1. 将源图片解码为 RGB；
2. 使用 bicubic resize 到 `224×224`；
3. 对所有类别执行相同的 JPEG round-trip；
4. 训练时 JPEG quality 从 `70/80/90/95` 中随机选择；
5. 验证时固定 JPEG quality 为 `90`；
6. JPEG subsampling 固定为 `2`；
7. 全部操作在数据读取过程中执行，不重新保存第二份数据。

配置和实现：

- [SID-Set 基础配置](../configs/sidset/base.yaml)
- [dataset.py](../src/repostguard/data/dataset.py)
- [transforms.py](../src/repostguard/data/transforms.py)

该策略可以削弱直接的 PNG/JPEG 格式捷径，但不能证明源文件中已有的历史压缩或生成器痕迹已经被彻底清除。

## 3. 模型架构

### 3.1 架构总览

| 模型 | 核心架构 | 总参数 | 可训练参数 | 冻结参数 | 权重初始化 |
|---|---|---:|---:|---:|---|
| B0 | EfficientNet-B0 + Linear | 4,008,829 | 4,008,829 | 0 | ImageNet |
| B1 | 与 B0 完全相同 | 4,008,829 | 4,008,829 | 0 | ImageNet |
| B2 | 冻结 CLIP ViT-B/32 + Linear | 87,849,729 | 513 | 87,849,216 | LAION2B CLIP |
| M2 | 冻结 CLIP 语义分支 + 法证分支 + 融合头 | 99,423,442 | 11,574,226 | 87,849,216 | CLIP 预训练，法证分支随机初始化 |
| M3 | M2 + 质量感知双分支门控 | 99,423,744 | 11,574,528 | 87,849,216 | 与 M2 相同，门控随机初始化 |

模型实现：[detectors.py](../src/repostguard/models/detectors.py)。

### 3.2 B0：clean CNN 基线

```text
224×224 RGB
  → ImageNet Normalize
  → EfficientNet-B0 features
  → Global Average Pooling
  → 1280 维特征
  → Dropout(0.2)
  → Linear(1280, 1)
```

- 使用 torchvision EfficientNet-B0 默认 ImageNet 预训练权重。
- 整个网络端到端微调。
- 训练输入经过共同格式去偏，但不加入额外鲁棒增强。
- 作用：衡量普通 CNN 的 clean 性能和自然鲁棒性。

### 3.3 B1：鲁棒增强 CNN 基线

B1 与 B0 的架构、参数量和初始化完全相同，唯一核心差异是训练时启用类别对称的随机鲁棒增强：

- 25%：不增加额外扰动；
- 50%：随机一种扰动；
- 25%：随机两种不同扰动。

增强参数范围：

| 增强 | 训练采样范围 |
|---|---|
| JPEG | quality 30–95 |
| Gaussian blur | σ 0.1–2.5 |
| resize | 比例 0.25–0.75；bilinear/bicubic |
| Gaussian noise | σ 0.005–0.10 |
| brightness/contrast/saturation | 各 0.8–1.2 |
| center crop | 比例 0.75–0.95 |

增强分布不依赖类别，真实和合成图片使用同一套策略。

### 3.4 B2：冻结 CLIP 语义基线

当前实际配置为：

- OpenCLIP `ViT-B-32`；
- 预训练权重 `laion2b_s34b_b79k`；
- 图像特征维度为 512；
- CLIP visual encoder 完全冻结并保持 eval 模式；
- 仅训练 `Linear(512,1)` 分类头。

B2 不使用额外鲁棒增强，只使用共同格式去偏。它主要测试预训练语义表征本身区分真实和全合成图片的能力。

### 3.5 M2：语义—法证双分支

#### 语义分支

```text
冻结 CLIP ViT-B/32
  → 512 维特征
  → LayerNorm
  → Linear(512,256)
  → GELU
```

#### 法证分支

1. 将 `224×224` 图像划分为 `56×56` 非重叠 patch，共 16 个；
2. 对 patch 计算 `16×16 DCT` 高频能量比例；
3. 选择 2 个最低频比例 patch 和 2 个最高频比例 patch，共 4 个；
4. 为每个 patch 构造 36 通道输入：
   - RGB：3 通道；
   - SRM-inspired 残差：30 通道；
   - NPR 最近邻缩放残差：3 通道；
5. 使用 `1×1 Conv: 36→16→3` 适配通道；
6. 输入随机初始化的 ResNet18；
7. 将 512 维输出投影到 256 维；
8. 加入高频/低频 patch 类型 embedding；
9. 通过 attention pooling 获得 256 维法证特征。

法证分支实现：[forensic.py](../src/repostguard/models/forensic.py)。

#### 特征融合

```text
语义特征 256 + 法证特征 256
  → concat 512
  → LayerNorm
  → Linear(512,256)
  → GELU
  → Dropout(0.2)
  → Linear(256,1)
```

### 3.6 M3：M2 + 质量感知门控

M3 保留完整 M2 架构，在两个分支融合前增加一个 302 参数的质量门控。

门控提取六个确定性质量统计量：

- 梯度能量；
- Laplacian 能量；
- JPEG 8×8 blockiness；
- 高频噪声；
- 有效分辨率 proxy；
- 亮度动态范围。

```text
6 维质量特征
  → LayerNorm
  → Linear(6,32)
  → GELU
  → Linear(32,2)
  → Softmax
  → 语义/法证分支权重
```

门控最后一层初始化为零，因此初始 softmax 权重为 `0.5/0.5`。模型将其乘以 2，使两个分支的初始缩放均为 1，与 M2 的初始融合形式一致。

质量特征不会直接进入最终分类器，只通过调整两个分支的权重间接影响预测。实现见 [quality_gate.py](../src/repostguard/models/quality_gate.py)。

## 4. 训练方法

### 4.1 权重来源

SID-Set 五组模型**不是从 CIFAKE 检测器 checkpoint 继续训练**：

- B0/B1 从 ImageNet EfficientNet 权重开始；
- B2/M2/M3 从 LAION2B CLIP 权重开始，且 CLIP 保持冻结；
- M2/M3 的法证、融合和门控模块随机初始化；
- M3 不是从 M2 的 `best.pt` 初始化；
- 如果作业中断，只会恢复各自 SID-Set 输出目录中的训练 checkpoint，不会加载 CIFAKE 检测器权重。

### 4.2 共同训练配置

- 随机种子：`20260827`
- deterministic：开启
- epoch：3
- optimizer：AdamW
- scheduler：5% warmup + cosine decay
- 最低学习率比例：0.05
- AMP：开启
- gradient clipping：1.0
- checkpoint 间隔：每 250 optimizer steps
- sampler：按照 `(label, source_dataset, generator_id)` 分组加权采样
- 每个 epoch 后在 clean validation 上计算 AUROC
- `best.pt` 按 clean validation AUROC 选择

基础配置：[base.yaml](../configs/base.yaml)。

### 4.3 各模型超参数和完成状态

| 模型 | Batch / 有效 Batch | LR | Weight decay | 成功更新步数 | 完成 Job ID |
|---|---:|---:|---:|---:|---:|
| B0 | 128 | 3e-4 | 1e-4 | 468 | 32269 |
| B1 | 128 | 3e-4 | 1e-4 | 468 | 32268 |
| B2 | 96 | 1e-3 | 1e-5 | 624 | 32275 |
| M2 | 24 × 累积 2 = 48 | 2e-4 | 1e-4 | 1,251 | 32284 |
| M3 | 24 × 累积 2 = 48 | 2e-4 | 1e-4 | 1,249 | 32338 |

M3 训练期间出现两次 AMP gradient overflow。训练器正确跳过对应 optimizer step，因此最终比计划少两个有效更新步；训练和后续评测正常完成。

各模型配置：

- [B0](../configs/sidset/b0.yaml)
- [B1](../configs/sidset/b1.yaml)
- [B2](../configs/sidset/b2.yaml)
- [M2](../configs/sidset/m2.yaml)
- [M3](../configs/sidset/m3.yaml)

### 4.4 损失函数

B0、B1、B2 使用 binary cross-entropy with logits。

M2、M3 使用 clean/增强成对输入：

$$
L = L_{\mathrm{BCE}}^{\mathrm{clean}}
  + L_{\mathrm{BCE}}^{\mathrm{aug}}
  + 0.5 L_{\mathrm{symmetric\ KL}}
  + 0.25 L_{\mathrm{cosine}}.
$$

- 两项 BCE 保证 clean 和扰动图片都能正确分类；
- symmetric Bernoulli KL 约束 clean/扰动预测概率一致；
- cosine loss 约束 clean/扰动融合特征一致。

实现见 [losses.py](../src/repostguard/losses.py)。

## 5. 测试方法

### 5.1 测试输入和 checkpoint

- 每个模型使用各自的 `best.pt`；
- 在相同的 4,000 张 SID-Set validation 图片上评测；
- clean 和全部扰动条件使用相同的图片与标签顺序；
- 所有评测结果均可追溯到 checkpoint SHA256 和 resolved config。

测试中的 `clean` 并非原始文件直接输入，而是：

```text
RGB → 224×224 bicubic → JPEG quality 90
```

其余扰动继续施加在该统一格式输入上。

### 5.2 18 个评测条件

共测试 1 个 clean 条件和 17 个扰动条件：

| 类别 | 条件 |
|---|---|
| clean | 无额外扰动 |
| JPEG | quality 90/70/50/30 |
| Gaussian blur | σ 0.5/1.0/2.0 |
| resize | 0.5 bicubic、0.25 bilinear |
| Gaussian noise | σ 0.02/0.05/0.10 |
| color jitter | brightness/contrast/saturation 全部为 0.8 或 1.2 |
| center crop | ratio 0.8 后恢复到原尺寸 |
| 组合扰动 | resize 0.5 + JPEG 70 |
| 组合扰动 | crop 0.8 + JPEG 50 |

完整评测矩阵：[transforms.yaml](../configs/transforms.yaml)。

### 5.3 阈值和指标

每个模型先在 clean validation 上选择使 balanced accuracy 最大的阈值，然后将该阈值固定用于全部 17 个扰动条件。

- AUROC：阈值无关的排序能力；
- balanced accuracy：固定 clean 阈值在分布偏移下的实际分类稳定性；
- robust mean：17 个非 clean 条件的算术平均；
- worst AUROC：17 个非 clean 条件中的最低 AUROC；
- delta AUROC：clean AUROC 减去 robust mean AUROC，越低越稳定。

不同模型的阈值数值不能直接比较，因为它们的输出概率校准不同。

## 6. 总体测试结果

| 模型 | Clean AUROC | Clean BAcc | 17 扰动平均 AUROC | 17 扰动平均 BAcc | 最坏 AUROC | Clean→扰动下降 | Clean 阈值 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | **0.999950** | **0.997000** | 0.953356 | 0.904941 | 0.469849 | 0.046594 | 0.4021 |
| B1 | 0.999933 | **0.997000** | **0.999559** | **0.992221** | 0.997098 | 0.000374 | 0.4407 |
| B2 | 0.998484 | 0.982250 | 0.996342 | 0.938912 | 0.984164 | 0.002142 | 0.4851 |
| M2 | 0.998046 | 0.979750 | 0.996126 | 0.964882 | 0.988386 | 0.001920 | 0.7261 |
| M3 | 0.999744 | 0.993750 | 0.999402 | 0.987544 | **0.998147** | **0.000342** | 0.8574 |

所有模型的最坏 AUROC 都出现在 `Gaussian noise σ=0.1`。

结果文件：

- [B0 summary](../outputs/sidset/b0/summary.json)
- [B1 summary](../outputs/sidset/b1/summary.json)
- [B2 summary](../outputs/sidset/b2/summary.json)
- [M2 summary](../outputs/sidset/m2/summary.json)
- [M3 summary](../outputs/sidset/m3/summary.json)

## 7. 各扰动条件 AUROC 对比

| 条件 | B0 | B1 | B2 | M2 | M3 |
|---|---:|---:|---:|---:|---:|
| Clean | **0.999950** | 0.999933 | 0.998484 | 0.998046 | 0.999744 |
| JPEG 90 | **0.999953** | 0.999938 | 0.998524 | 0.998038 | 0.999750 |
| JPEG 70 | 0.999907 | **0.999912** | 0.998493 | 0.998236 | 0.999752 |
| JPEG 50 | 0.999783 | **0.999841** | 0.998011 | 0.997904 | 0.999720 |
| JPEG 30 | 0.999338 | **0.999761** | 0.996557 | 0.997038 | 0.999626 |
| Blur 0.5 | **0.999946** | 0.999931 | 0.998505 | 0.998124 | 0.999833 |
| Blur 1.0 | 0.999703 | **0.999886** | 0.998112 | 0.998105 | 0.999783 |
| Blur 2.0 | 0.997051 | **0.999464** | 0.995336 | 0.994818 | 0.999289 |
| Resize 0.5 | 0.999884 | **0.999904** | 0.997595 | 0.997209 | 0.999624 |
| Resize 0.25 | 0.992370 | **0.999077** | 0.994935 | 0.993394 | 0.998944 |
| Noise 0.02 | 0.983222 | **0.999773** | 0.996211 | 0.996109 | 0.999727 |
| Noise 0.05 | 0.768842 | **0.999265** | 0.992492 | 0.992783 | 0.999210 |
| Noise 0.10 | 0.469849 | 0.997098 | 0.984164 | 0.988386 | **0.998147** |
| Color 0.8 | 0.999570 | **0.999844** | 0.998738 | 0.997939 | 0.999732 |
| Color 1.2 | 0.999378 | **0.999763** | 0.997586 | 0.996451 | 0.999184 |
| Crop 0.8 | 0.999770 | **0.999785** | 0.997884 | 0.996525 | 0.999035 |
| Resize 0.5 + JPEG 70 | 0.999610 | **0.999872** | 0.997691 | 0.997088 | 0.999695 |
| Crop 0.8 + JPEG 50 | 0.998878 | **0.999393** | 0.996979 | 0.995992 | 0.998781 |

B1 在 18 个条件中的 14 个取得最高 AUROC；B0 在 clean、JPEG 90 和轻度 blur 上略高；M3 在最强高斯噪声下取得最高 AUROC。

各模型完整逐条件指标：

- [B0 metrics](../outputs/sidset/b0/metrics_by_transform.csv)
- [B1 metrics](../outputs/sidset/b1/metrics_by_transform.csv)
- [B2 metrics](../outputs/sidset/b2/metrics_by_transform.csv)
- [M2 metrics](../outputs/sidset/m2/metrics_by_transform.csv)
- [M3 metrics](../outputs/sidset/m3/metrics_by_transform.csv)

## 8. 严格六重随机组合扰动测试

### 8.1 测试定义

在原有 18 条件测试之外，增加一个 `strict_random_six` 条件。每张验证图片都会依次经历全部六类扰动：

```text
中心裁剪
  → 降采样并恢复原尺寸
  → brightness/contrast/saturation 颜色变化
  → Gaussian blur
  → Gaussian noise
  → JPEG 压缩
```

每个扰动的强度独立随机采样：

| 扰动 | 随机范围 |
|---|---|
| center crop | ratio 0.75–0.95 |
| resize | scale 0.25–0.75；bilinear/bicubic 随机 |
| brightness | 0.8–1.2 |
| contrast | 0.8–1.2 |
| saturation | 0.8–1.2 |
| Gaussian blur | σ 0.1–2.5 |
| Gaussian noise | σ 0.005–0.10 |
| JPEG | quality 30–95；subsampling 2 |

随机强度使用 `seed 20260828 + validation 样本索引` 生成。因此：

- 不同图片得到不同的随机扰动强度；
- B0、B1、B2、M2、M3 对同一图片使用完全相同的六重扰动；
- 结果不依赖 DataLoader worker 调度，可用相同 seed 精确复现；
- 严格性来自每张图片都必须同时经历六种扰动，而不是只抽取一种或两种。

测试继续使用 4,000 张 SID-Set validation 图片。先重新计算同一 checkpoint 的 clean 输出并选择 clean balanced accuracy 最优阈值，再将该阈值固定应用到严格扰动条件。由于 GPU 浮点计算的微小差异，本次 clean 阈值与原 18 条件 run card 可能略有不同；以下 strict 指标均使用本次 run card 记录的阈值。

实现和配置：

- [strict6 评测矩阵](../configs/sidset_strict6.yaml)
- [六重扰动实现](../src/repostguard/data/transforms.py)
- [独立评测输出支持](../src/repostguard/evaluate.py)
- [SLURM 评测脚本](../scripts/slurm/evaluate_sidset_strict6.sbatch)

评测 Job `32356` 于 2026-08-28 在单张 NVIDIA A40 上完成，状态为 `COMPLETED`、退出码为 0，用时 7 分 18 秒。B0→B1→B2→M2→M3 在同一作业中顺序评测，没有重新训练或修改任何 checkpoint。

### 8.2 严格六重扰动结果

| 模型 | AUROC | AP | BAcc | Macro F1 | AIGC recall | Real specificity | FPR | TPR@1%FPR | TPR@5%FPR | Clean→Strict AUROC 下降 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.718696 | 0.718154 | 0.510750 | 0.359252 | 0.024500 | 0.997000 | 0.003000 | 0.083500 | 0.216000 | 0.281254 |
| B1 | **0.991335** | **0.991919** | **0.937750** | **0.937617** | **0.891500** | 0.984000 | 0.016000 | **0.861000** | **0.959000** | **0.008597** |
| B2 | 0.936252 | 0.939652 | 0.618250 | 0.553524 | 0.237500 | 0.999000 | 0.001000 | 0.488000 | 0.707000 | 0.062232 |
| M2 | 0.961302 | 0.965733 | 0.684500 | 0.649746 | 0.369500 | 0.999500 | 0.000500 | 0.657000 | 0.826000 | 0.036745 |
| M3 | 0.975617 | 0.978577 | 0.739000 | 0.719921 | 0.478000 | **1.000000** | **0.000000** | 0.748000 | 0.886000 | 0.024128 |

严格条件的 confusion matrix：

| 模型 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| B0 | 1,994 | 6 | 1,951 | 49 |
| B1 | 1,968 | 32 | 217 | 1,783 |
| B2 | 1,998 | 2 | 1,525 | 475 |
| M2 | 1,999 | 1 | 1,261 | 739 |
| M3 | 2,000 | 0 | 1,044 | 956 |

完整结果：

- [B0 strict6 summary](../outputs/sidset_strict6/b0/summary.json) / [metrics](../outputs/sidset_strict6/b0/metrics_by_transform.csv) / [run card](../outputs/sidset_strict6/b0/run_card.json)
- [B1 strict6 summary](../outputs/sidset_strict6/b1/summary.json) / [metrics](../outputs/sidset_strict6/b1/metrics_by_transform.csv) / [run card](../outputs/sidset_strict6/b1/run_card.json)
- [B2 strict6 summary](../outputs/sidset_strict6/b2/summary.json) / [metrics](../outputs/sidset_strict6/b2/metrics_by_transform.csv) / [run card](../outputs/sidset_strict6/b2/run_card.json)
- [M2 strict6 summary](../outputs/sidset_strict6/m2/summary.json) / [metrics](../outputs/sidset_strict6/m2/metrics_by_transform.csv) / [run card](../outputs/sidset_strict6/m2/run_card.json)
- [M3 strict6 summary](../outputs/sidset_strict6/m3/summary.json) / [metrics](../outputs/sidset_strict6/m3/metrics_by_transform.csv) / [run card](../outputs/sidset_strict6/m3/run_card.json)

### 8.3 结果排序和解释

严格条件下，AUROC 和固定阈值 balanced accuracy 的排序完全一致：

```text
B1 > M3 > M2 > B2 > B0
```

主要观察：

1. **B1 仍是明显最强模型。** 相比 B0，B1 的 strict AUROC 高 27.264 个百分点、balanced accuracy 高 42.700 个百分点。这进一步确认 B1 的收益来自六类类别对称鲁棒增强，而不是网络规模。
2. **M3 的组合扰动表现优于 M2。** AUROC 高 1.431 个百分点，balanced accuracy 高 5.450 个百分点，但仍明显低于 B1：AUROC 低 1.572 个百分点，balanced accuracy 低 19.875 个百分点。
3. **M2 优于 B2。** AUROC 高 2.505 个百分点，balanced accuracy 高 6.625 个百分点，说明法证分支和成对一致性训练在多扰动叠加时比单项扰动测试呈现出更明显的收益。
4. **B0 基本失去可用 operating point。** 虽然 AUROC 仍有 0.718696，但 clean 阈值下 AIGC recall 只有 2.45%，balanced accuracy 接近随机水平。
5. **主要失败模式是合成图片被判为真实。** B2/M2/M3 的 real specificity 接近或达到 1，但 AIGC recall 分别只有 23.75%、36.95% 和 47.80%。六重扰动整体使合成图片分数向真实类别方向偏移，而不是造成大量真实图片误报。
6. **B1 的阈值稳定性显著更好。** 在仍保持 98.4% real specificity 的同时，AIGC recall 达到 89.15%，远高于其他模型。

### 8.4 M3 门控在严格条件下的变化

| 条件 | 语义分支平均权重 | 法证分支平均权重 |
|---|---:|---:|
| clean | 0.621542 | 0.378458 |
| strict random six | 0.614122 | 0.385878 |
| 变化 | -0.007420 | +0.007420 |

M3 在六重扰动下只小幅提高法证分支权重。考虑到输入已经同时经历强缩放、模糊、噪声和 JPEG，门控响应幅度较小，尚不能证明质量门控充分适配了严重组合退化。

### 8.5 严格测试的局限

- 每张图片的强度不同，但当前只使用一个全局 seed profile；需要更换多个 seed 重复评测，报告均值、标准差和置信区间。
- 六种扰动采用固定顺序；不同平台的真实处理链顺序可能不同。
- 随机范围沿用训练增强范围，测试的是“六种全部叠加”，并没有把单项强度扩大到训练范围之外。
- 阈值仍来自同一 validation split 的 clean 条件，不是独立校准集阈值。
- 所有严格样本仍由原 4,000 张 validation 图片生成，不能替代外部数据和未见生成器测试。

## 9. 关键对比与解释

### 9.1 B1 对 B0：鲁棒增强带来最大收益

B1 与 B0 架构完全相同，因此两者差异主要来自训练增强：

- clean AUROC 仅下降 0.000017，可视为基本持平；
- robust mean AUROC 提升 4.620 个百分点；
- robust mean balanced accuracy 提升 8.728 个百分点；
- worst AUROC 提升 52.725 个百分点；
- noise 0.1 下，B0 AUROC 只有 0.469849，B1 为 0.997098。

这说明 B0 的高 clean AUROC 只能代表同分布识别能力，不能代表真实转发和后处理条件下的可靠性。

### 9.2 M2 对 B2：提高固定阈值稳定性

M2 相比 B2：

- clean AUROC 下降 0.044 个百分点；
- robust mean AUROC 下降 0.022 个百分点；
- robust mean balanced accuracy 提升 2.597 个百分点；
- worst AUROC 提升 0.422 个百分点。

noise 0.1 条件下：

| 模型 | AUROC | Balanced accuracy |
|---|---:|---:|
| B2 | 0.984164 | 0.726000 |
| M2 | 0.988386 | 0.931250 |

法证分支和一致性训练明显改善了固定 clean 阈值的抗偏移能力，但当前没有提高平均 AUROC。由于尚无仅法证分支、仅一致性损失等独立消融，不能把收益明确归因于某个组件。

### 9.3 M3 对 M2：整体提高，但归因尚不充分

M3 相比 M2：

- clean AUROC 提升 0.170 个百分点；
- clean balanced accuracy 提升 1.400 个百分点；
- robust mean AUROC 提升 0.328 个百分点；
- robust mean balanced accuracy 提升 2.266 个百分点；
- worst AUROC 提升 0.976 个百分点。

但是 M2 和 M3 是两次独立随机训练，M3 并非从 M2 checkpoint 加入 gate 后继续训练。因此这些提升不能全部直接归因于新增的 302 个门控参数。需要共享初始化的固定 `0.5/0.5 gate` 对照和多随机种子实验。

### 9.4 M3 对 B1：最坏情况略优，平均性能和效率不占优

M3 相比 B1：

- robust mean AUROC 低 0.0157 个百分点；
- robust mean balanced accuracy 低 0.468 个百分点；
- worst AUROC 高 0.105 个百分点；
- 可训练参数约为 B1 的 2.89 倍；
- 总参数约为 B1 的 24.8 倍。

因此，B1 更适合作为当前主基线；M3 的主要价值是更好的 worst-case AUROC 和进一步研究潜力。

## 10. M3 质量门控审计

M3 的六个输入统计量在设计上是无标签的图像质量特征，但当前 validation 审计显示，部分特征本身具有明显标签可分性：

| 质量特征 | 标签可分 AUROC |
|---|---:|
| 高频噪声 | 0.6686 |
| Laplacian 能量 | 0.6681 |
| 梯度能量 | 0.6564 |
| 有效分辨率 proxy | 0.6245 |
| JPEG blockiness | 0.5359 |
| 亮度动态范围 | 0.5240 |

如果质量特征完全不携带标签信息，标签可分 AUROC 应接近 0.5。虽然这些特征不会直接进入分类头，模型仍可能通过门控权重间接利用真实/合成图片的低层统计差异。

门控平均法证分支权重：

| 条件 | 法证分支权重 |
|---|---:|
| clean | 0.3785 |
| blur 2.0 | 0.4171 |
| resize 0.25 | 0.4221 |
| noise 0.1 | 0.3912 |

严重模糊和降采样下，M3 反而提高了法证分支权重。这不一定是错误行为，但与“低层证据受损时更多依赖语义分支”的原始直觉不完全一致，需要通过消融实验进一步解释。

完整审计：[quality_gate_audit.json](../outputs/sidset/m3/quality_gate_audit.json)。

## 11. 当前模型定位

1. **B0**：clean-only 对照，用于证明普通 CNN 的扰动脆弱性，不适合作为当前最终模型。
2. **B1**：当前综合最优基线，架构简单、参数少、平均鲁棒性最高，应作为后续模型必须超过的主要对照。
3. **B2**：强冻结语义基线，训练参数极少，但固定阈值下的稳定性弱于 B1/M2/M3。
4. **M2**：证明语义—法证融合可能改善部分阈值稳定性和最坏情况，但尚未超过 B2 的平均 AUROC。
5. **M3**：worst-case AUROC 最好，性能接近 B1，但质量门控存在潜在标签捷径，且模型复杂度明显更高。

## 12. 结果局限与下一步

### 12.1 当前局限

- 只进行了一次固定随机种子实验；
- best checkpoint 使用相同 validation split 的 clean AUROC 选择；
- clean threshold 也在该 validation split 上选择；
- 17 个扰动条件均由同一批 validation 图片生成；
- 尚无独立 blind test；
- 尚无未见生成器、外部数据集或跨平台真实转发测试；
- 格式去偏不能保证源数据历史痕迹完全消失；
- M2 缺少法证分支和一致性损失的独立消融；
- M3 缺少相同初始化的固定门控对照。
- strict random six 当前只测试了一个随机 seed profile 和一种固定扰动顺序。

因此，当前结果适合用于初步可行性判断和架构筛选，不应直接解释为真实世界泛化能力已经得到确认。

### 12.2 建议的后续实验

1. 对 B1、M2、M3 至少运行 3–5 个随机种子，报告均值和标准差；
2. 增加 M2 消融：semantic-only、forensic-only、无 KL、无 feature consistency；
3. 增加 M3 固定 `0.5/0.5` gate 对照，并与可学习 gate 共享初始化；
4. 对门控质量特征进行类别条件去偏或加入防捷径约束；
5. 使用未参与模型选择的 SID reserved split；
6. 增加未见生成器和外部 AIGC 数据集测试；
7. 增加真实社交平台重编码、截图、裁剪、缩放和组合转发链路验证；
8. 在独立校准集上选择阈值，并在完全独立测试集上报告最终结果。
9. 对 strict random six 使用多个随机 seed 和多种处理顺序，报告结果分布而不是单次点估计。
