# Community Forensics train-v1 / train-v2 训练与测试对比

> 状态日期：2026-08-30
>
> train-v2 完整鲁棒性评测：SLURM job `32745`，`COMPLETED (0:0)`，节点 `TC2N08`
>
> 比较对象：B0、B1、B2、M2、M3；单 seed（`20260828`）

## 1. 结论摘要

train-v2 把原 External seen-family 的 1,000 张 Real 和 1,000 张 AIGI 加入训练，在不改变模型架构、主要训练超参数、在线格式去偏和内部验证集的前提下，将训练集从 18,000 张扩展到 20,000 张。原 seen-family 从此仅属于训练谱系，不再作为验证或测试集。

与 train-v1 checkpoint 在相同测试图片和相同 21 条件扰动矩阵上的结果相比：

- 五个模型的五切片 clean 宏平均 AUROC 全部提高；M2、M3 的增幅最大，分别为 `+0.0711` 和 `+0.0707`。
- exact-seen 提升最显著：B0/M2/M3 分别提高 `+0.1595/+0.1008/+0.0955`，符合新增精确生成器训练样本的预期。
- strict unseen-generator 没有因加入 seen-family 而退化，五个模型均提高；M3 从 `0.8631` 提升到 `0.9279`，并取得最高 clean Balanced Accuracy `0.8525`。
- B2 仍是跨五切片排序最稳的模型：train-v2 clean 宏平均 AUROC `0.7456`、20 个扰动宏平均 AUROC `0.7357`，在 Hourglass、DFGAN、GALIP 三个 hard 切片上均为第一。
- M3 是 strict unseen-generator 的最佳模型：clean AUROC `0.9279`，两组四阶段扰动分别为 `0.9087/0.8732`，随机六阶段扰动为 `0.8328`。
- 将 Hourglass、DFGAN、GALIP 的相同精确生成器加入训练后，M2/M3 的 hard 结果有所改善，但 Hourglass/DFGAN 仍低于随机排序附近；这说明问题不只是“生成器身份未见”，还包含数据来源、内容或处理链分布偏移。

因此，train-v2 是明确的总体改进，但没有产生单一全场景最优模型：strict unseen 的固定阈值检测优先 M3，跨困难生成器排序优先 B2。

## 2. 训练集变化

| 项目 | train-v1 | train-v2 | 变化 |
|---|---:|---:|---:|
| 总图片 | 18,000 | 20,000 | +2,000（+11.1%） |
| Real / AIGI | 9,000 / 9,000 | 10,000 / 10,000 | 各 +1,000，继续平衡 |
| 数据来源 | Small 18,000 | Small 18,000 + Eval 2,000 | 新增外部来源 |
| AIGI 精确生成器 | 900 | 909 | +9 |
| AIGI GAN | 20 | 242 | +222 |
| AIGI Latent Diffusion | 8,970 | 9,637 | +667 |
| AIGI Pixel Diffusion | 10 | 121 | +111 |
| 格式 | JPEG 3,993；PNG 14,007 | JPEG 4,475；PNG 15,270；TIFF 250；WEBP 5 | 格式更多样 |

新增的 9 个精确生成器为：`decidiffusionv2`、`dfgan`、`galip`、`hourglass`、`kandinsky-2-2`、`kvikontent-midjourney-v6`、`lcm-lora-sdv15`、`lcm-lora-sdxl`、`lcm-lora-ssd1b`。

train-v2 审计确认：训练集内部以及 train-v2 与 strict unseen 之间的 path、sample ID、SHA-256 和 source locator 精确重叠均为 0；三个 hard-v2 切片与 train-v2 也没有图片级重叠。新增图片没有复制为第二份数据，而是保留原物理路径和来源谱系。

## 3. 公平比较条件

两轮训练保持以下条件不变：

- 五种模型架构及公开预训练底座；
- seed `20260828`、3 epochs、optimizer、学习率、batch size 和损失定义；
- clean / single / double 在线增强概率 `0.25 / 0.50 / 0.25`；
- 在线 JPEG 格式去偏：训练质量随机取 `70/80/90/95`，评测质量固定为 `90`；
- 内部 checkpoint 选择集 `community_forensics_val_unseen_generator.csv`；
- 外部评测阈值来自内部 clean validation，测试标签不用于调阈值；
- robustness-v2 共 21 个条件：clean、17 个原有扰动、两组四阶段复合扰动和一组随机六阶段复合扰动。

直接对比使用五个保留切片：

| 切片 | Real | AIGI | train-v2 exposure | 用途 |
|---|---:|---:|---|---|
| exact-seen generator | 1,000 | 1,000 | 精确生成器已见，图片不重合 | 常规 exact-seen |
| Hard Hourglass | 250 | 250 | 精确生成器已见，图片不重合 | 困难切片 |
| Hard DFGAN | 250 | 250 | 精确生成器已见，图片不重合 | 困难切片 |
| Hard GALIP | 250 | 250 | 精确生成器已见，图片不重合 | 困难切片 |
| strict unseen-generator | 1,000 | 1,000 | 精确生成器和生成器大类均未见 | 主外部测试 |

三个 hard-v2 manifest 与 v1 hard manifest 的 sample ID 和 SHA-256 完全一致，仅 exposure 标签随训练协议变化。因此 hard 指标可以逐项比较。原 seen-family 的 2,000 张图片已进入 train-v2，故不再纳入 v2 测试或宏平均；v1 seen-family 结果只保留为历史记录。

每个模型实际评测 `5,500 × 21 = 115,500` 个图像—条件样本，五个模型共 `577,500` 个推理样本。

## 4. 内部验证结果

括号内为 train-v2 相对 train-v1 的绝对变化。

| 模型 | train-v2 clean AUROC | train-v2 17 扰动均值 AUROC | 解释 |
|---|---:|---:|---|
| B0 | 0.9169（+0.0178） | 0.8751（+0.0180） | clean 与扰动均改善 |
| B1 | 0.9069（+0.0049） | 0.8648（-0.0103） | clean 小幅改善，但内部扰动均值下降 |
| B2 | 0.7524（+0.0299） | 0.7350（+0.0298） | 线性 CLIP 基线稳定改善 |
| M2 | 0.9645（+0.0157） | 0.9538（+0.0136） | 强内部验证表现 |
| M3 | **0.9645（+0.0136）** | **0.9540（+0.0120）** | 与 M2 基本持平 |

内部结果只用于 checkpoint 选择和阈值冻结，不替代外部测试结论。

## 5. 五个保留外部切片的 clean AUROC

单元格为 `train-v2 AUROC（相对 train-v1 变化）`。

| 模型 | Exact-seen | Hourglass | DFGAN | GALIP | Strict unseen | 五切片宏平均 |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.8042（+0.1595） | 0.3014（+0.0620） | 0.4387（-0.0078） | 0.4405（+0.0225） | 0.8199（+0.0289） | 0.5610（+0.0530） |
| B1 | 0.7049（+0.0629） | 0.2749（+0.0197） | 0.4468（+0.0213） | 0.4376（+0.0316） | 0.8204（+0.0609） | 0.5369（+0.0393） |
| B2 | 0.7947（+0.0115） | **0.7315（+0.0045）** | **0.6873（+0.0009）** | **0.7511（+0.0023）** | 0.7632（+0.0404） | **0.7456（+0.0119）** |
| M2 | **0.8258（+0.1008）** | 0.4415（+0.0782） | 0.4350（+0.0296） | 0.4843（+0.0842） | 0.9191（+0.0627） | 0.6211（+0.0711） |
| M3 | 0.8254（+0.0955） | 0.4328（+0.0623） | 0.4316（+0.0386） | 0.5004（+0.0921） | **0.9279（+0.0648）** | 0.6236（+0.0707） |

关键观察：

1. M2/M3 对 exact-seen 和 strict unseen 的提升最大，但 hard 排序仍不稳。
2. B2 在三个 hard 切片上明显领先，并保持最高五切片宏平均；它仍是更好的广域排序/二阶段筛查候选。
3. M3 的 strict unseen AUROC 比 M2 高 `0.0087`，但 exact-seen 略低 `0.0004`，差异很小；单 seed 下不能宣称门控稳定显著优于 M2。

## 6. 20 个扰动条件平均 AUROC

本表不包含 clean。单元格为 `train-v2 扰动均值（相对 train-v1 变化）`。

| 模型 | Exact-seen | Hourglass | DFGAN | GALIP | Strict unseen | 五切片宏平均 |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.7583（+0.1308） | 0.3294（+0.0713） | 0.4651（-0.0103） | 0.4542（+0.0149） | 0.7480（+0.0232） | 0.5510（+0.0460） |
| B1 | 0.6926（+0.0644） | 0.3057（+0.0189） | 0.4950（+0.0347） | 0.4607（+0.0308） | 0.7657（+0.0727） | 0.5439（+0.0443） |
| B2 | 0.7830（+0.0110） | **0.7103（+0.0015）** | **0.6847（-0.0018）** | **0.7371（-0.0036）** | 0.7635（+0.0355） | **0.7357（+0.0085）** |
| M2 | 0.8192（+0.0904） | 0.4242（+0.0747） | 0.4377（+0.0038） | 0.4802（+0.0671） | 0.8994（+0.0620） | 0.6122（+0.0596） |
| M3 | **0.8222（+0.0951）** | 0.4122（+0.0609） | 0.4262（+0.0169） | 0.4891（+0.0706） | **0.9082（+0.0661）** | 0.6116（+0.0619） |

五切片 clean Balanced Accuracy 宏平均也全部提高：B0 `0.5566（+0.0411）`、B1 `0.5271（+0.0173）`、B2 `0.6181（+0.0070）`、M2 `0.5846（+0.0419）`、M3 `0.5761（+0.0374）`。其中 B2 的 AUROC 与 Balanced Accuracy 宏平均均最高，但在 strict unseen 单切片上 M3 的 clean Balanced Accuracy 最高，为 `0.8525（+0.0595）`。

## 7. Strict unseen 多阶段复合扰动

| 模型 | 四阶段 platform repost AUROC | 四阶段 edit repost AUROC | 随机六阶段 AUROC | 随机六阶段 BAcc |
|---|---:|---:|---:|---:|
| B0 | 0.7318（+0.0361） | 0.7105（+0.0108） | 0.6683（+0.0168） | 0.6190（+0.0200） |
| B1 | 0.7494（+0.0719） | 0.6993（+0.0883） | 0.6551（+0.0967） | 0.6105（+0.0600） |
| B2 | 0.7734（+0.0328） | 0.6802（+0.0381） | 0.6543（+0.0325） | 0.5575（+0.0175） |
| M2 | 0.8968（+0.0622） | 0.8631（+0.0548） | 0.8276（+0.0428） | 0.7115（+0.0120） |
| M3 | **0.9087（+0.0710）** | **0.8732（+0.0615）** | **0.8328（+0.0426）** | **0.7390（+0.0285）** |

所有模型的 strict unseen 三组复合扰动 AUROC 均高于 train-v1。M3 在三组条件上均为最高；随机六阶段仍是 M2/M3 的最差或接近最差条件，说明连续编辑链仍是主要压力场景。

## 8. 模型选择与下一步

| 目标 | 当前建议 | 依据 |
|---|---|---|
| strict unseen 固定阈值检测 | M3 | clean AUROC/BAcc 与三组多阶段扰动均最佳 |
| 跨 hard 生成器排序或二阶段召回 | B2 | 三个 hard 和五切片宏平均均最佳 |
| exact-seen 检测 | M2/M3 | clean 与扰动均值最高，二者非常接近 |
| 轻量部署 | B0/B1 | 参数量约 4M，但 hard 泛化明显不足 |

下一轮不应继续只增加同生成器样本。优先级应是：

1. 对 Hourglass、DFGAN、GALIP 做 generator-level error audit，区分内容、分辨率、来源与编解码混杂；
2. 进行至少 3 个 seed，并对 generator 分层 bootstrap，验证 M2/M3 的增益是否稳定；
3. 研究 B2 与 M3 的 score/feature ensemble，在保持 strict unseen 性能的同时补足 hard 排序；
4. 使用独立校准集进行阈值校准，不接触 strict unseen 或 hard 测试标签；
5. 保持原 seen-family 永久退出评测，避免训练—测试泄漏。

## 9. 证据与可追溯性

- train-v2 manifest 与审计：[`community_forensics_train_v2.csv`](../../data/manifests/community_forensics_train_v2.csv)、[`community_forensics_train_v2_audit.json`](../../data/manifests/community_forensics_train_v2_audit.json)
- train-v2 checkpoint/内部评测：[`outputs/community_forensics_v2/`](../../outputs/community_forensics_v2/)
- train-v2 25 个外部鲁棒性结果：[`outputs/community_forensics_v2_robustness_v2/`](../../outputs/community_forensics_v2_robustness_v2/)
- train-v1 历史鲁棒性结果：[`outputs/community_forensics_robustness_v2/`](../../outputs/community_forensics_robustness_v2/)
- 扰动矩阵：[`community_forensics_robustness_v2.yaml`](../../configs/community_forensics_robustness_v2.yaml)
- 完整评测日志：[`rg_cf_v2_robust_32745.out`](../../logs/rg_cf_v2_robust_32745.out)、[`rg_cf_v2_robust_32745.err`](../../logs/rg_cf_v2_robust_32745.err)

train-v2 训练/基础评测 jobs `32730/32731/32732/32733/32735/32737/32738/32740/32742/32743` 及完整鲁棒性 job `32745` 均为 `COMPLETED (0:0)`；列出的正式训练与评测作业均运行在 `TC2N08`。完整鲁棒性评测产生 5 个模型完成标记、25 个切片完成标记、25 份 `summary.json` 和 25 份 `metrics_by_transform.csv`。

## 10. 解释限制

- 当前主要结论来自单 seed，未给出显著性检验；小差异只能描述，不能宣称稳定优势。
- train-v2 同时改变了精确生成器覆盖、来源、架构占比和格式分布，无法把提升唯一归因于某一因素。
- hard 切片在 v1 是 exact-unseen，在 v2 是 exact-seen；图片相同但 exposure 语义改变，比较回答的是“加入相同生成器训练样本后是否改善”，不是两个相同开放集协议的重复实验。
- AUROC 衡量排序，Balanced Accuracy 衡量冻结阈值决策；两者必须共同报告。
