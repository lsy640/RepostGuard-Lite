# Community Forensics train-v1/v2/v3 数据清单与重建说明

> 项目：RepostGuard-Lite<br>
> 配置入口：[`configs/community_forensics_v3/base.yaml`](../../configs/community_forensics_v3/base.yaml)<br>
> 协议：`community_forensics_train_v3_small_gan_pixdiff_expansion`<br>
> 文档目的：汇总 train-v1/v2/v3 的数据血缘、当前 train-v3 使用的训练/验证/测试清单，以及各阶段依据冻结清单重新获取原始数据的方法。

## 1. 结论摘要

train-v3 的正式训练清单包含 **24,000 张图像**，类别完全平衡：12,000 张真实图像和 12,000 张 AIGI 图像。它不是独立构建的新数据集，而是由 train-v2 的 20,000 张图像与 v3 新增的 4,000 张图像合并得到：

```text
train-v1：CommunityForensics-Small 18,000 张
    + train-v2：原 External seen-family 2,000 张并入训练
    + train-v3：Small 新增 GAN 1,000 + PixDiff 1,000
                + 两组对应真实图像各 1,000
    = train-v3：24,000 张（Real 12,000 / AIGI 12,000）
```

当前配置声明的评测数据包括：

- 内部 checkpoint-selection 验证集：2,000 张；
- external exact-seen 诊断集：2,000 张；
- Hourglass、DFGAN、GALIP 三个困难切片：各 500 张；
- expanded strict unseen-generator 测试集：4,000 张。

上述评测清单合计声明 9,500 行，但三个困难切片共享同一组 250 张真实图像，因此不能把 9,500 直接理解为互不重复的图像总数。

清单已经记录 Hugging Face 数据集 ID、固定 revision、Parquet 文件、row group、row index、SHA-256 和 pHash，足以精确定位并校验样本。需要注意：**现有代码是按构建阶段恢复数据，并非一个“输入任意最终 CSV 即一键下载全部图像”的通用工具**。从空目录重建时仍需按本文第 10 节的依赖顺序执行构建脚本。

## 2. 当前配置的数据角色

| 数据角色 | 当前清单 | 行数 | Real / AIGI | 用途与边界 |
|---|---|---:|---:|---|
| 正式训练集 | [`community_forensics_train_v3.csv`](../../data/manifests/community_forensics_train_v3.csv) | 24,000 | 12,000 / 12,000 | B0、B1、B2、M2、M3 的 train-v3 训练输入 |
| 内部验证集 | [`community_forensics_val_unseen_generator.csv`](../../data/manifests/community_forensics_val_unseen_generator.csv) | 2,000 | 1,000 / 1,000 | Small 内部留出集；用于 checkpoint 选择 |
| exact-seen 诊断集 | [`community_forensics_val_external_exact_seen_generator.csv`](../../data/manifests/community_forensics_val_external_exact_seen_generator.csv) | 2,000 | 1,000 / 1,000 | 精确生成器在训练中见过，但评测图像与来源数据集不同 |
| Hourglass 困难切片 | [`community_forensics_val_hard_hourglass_v2_exact_seen.csv`](../../data/manifests/community_forensics_val_hard_hourglass_v2_exact_seen.csv) | 500 | 250 / 250 | train-v2 已将 Hourglass 并入训练，因此当前定义为 exact-seen |
| DFGAN 困难切片 | [`community_forensics_val_hard_dfgan_v2_exact_seen.csv`](../../data/manifests/community_forensics_val_hard_dfgan_v2_exact_seen.csv) | 500 | 250 / 250 | train-v2 已将 DFGAN 并入训练，因此当前定义为 exact-seen |
| GALIP 困难切片 | [`community_forensics_val_hard_galip_v2_exact_seen.csv`](../../data/manifests/community_forensics_val_hard_galip_v2_exact_seen.csv) | 500 | 250 / 250 | train-v2 已将 GALIP 并入训练，因此当前定义为 exact-seen |
| strict unseen-generator 测试集 | [`community_forensics_test_external_unseen_generator_v3_expanded.csv`](../../data/manifests/community_forensics_test_external_unseen_generator_v3_expanded.csv) | 4,000 | 2,000 / 2,000 | 架构大类与精确生成器均未用于训练的正式外部测试 |

协议边界：原 `External seen-family` 的 2,000 张图像已经全部并入 train-v2，冻结旧清单只用于数据血缘追踪，**不得再作为 train-v2/train-v3 后续模型的测试数据**。

## 3. train-v1：24k 基础池的构建与重新下载

train-v1 是后续 v2/v3 的物理数据基础。它固定扫描 `CommunityForensics-Small` 和 `CommunityForensics-Eval` 两个 Hugging Face revision，生成 24,000 行选择计划并只物化这些被选样本，而不是把上游约 466 GB 的全部 Parquet 数据下载到本地。

### 3.1 train-v1 冻结数据组成

| train-v1 数据角色 | 来源 | Real / AIGI | 合计 | 后续状态 |
|---|---|---:|---:|---|
| `train` | Small | 9,000 / 9,000 | 18,000 | train-v2/v3 继续继承 |
| `val_unseen_generator` | Small | 1,000 / 1,000 | 2,000 | 继续作为内部 checkpoint-selection 验证集 |
| `test_external_seen_family` | Eval | 1,000 / 1,000 | 2,000 | v1 外部测试；从 v2 起全部提升为训练，禁止继续评测 |
| `test_external_unseen_generator` | Eval | 1,000 / 1,000 | 2,000 | 冻结 strict unseen 基础测试集 |
| **总计** | Small + Eval | **12,000 / 12,000** | **24,000** | 物理基础池 |

AIGI 的确定性抽样规则如下：

- Small 中满足每类至少 10 张的 1,000 个精确生成器进入候选池；
- 900 个精确生成器用于 train，每类 10 张，共 9,000 张 AIGI；
- 另外 100 个精确生成器用于内部 validation，每类 10 张，共 1,000 张 AIGI；
- train 与 validation 的精确生成器集合不相交；
- Eval 的 family-seen AIGI 来自 GAN、LatDiff、PixDiff 三个训练已见大类中的 9 个未见精确生成器；
- Eval 的 strict unseen AIGI 来自 Commercial、Other 两个训练未见大类中的 12 个未见精确生成器；
- Eval 的真实图像来自 RAISE、COCO、FFHQ、LAION，各选 500 张，并在两个 external 测试集间各分配 250 张。

Small 固定 revision 的真实图像均记录为 `real_source=N/A`，因此 10,000 张 Small Real 只能进行全局确定性抽样，再分为 train 9,000 张和 validation 1,000 张，不能声称实现了 FFHQ/VISION/COCO/Landscapes HQ 的来源均衡。

### 3.2 train-v1 冻结制品

| 制品 | 行数 | SHA-256 |
|---|---:|---|
| [`community_forensics_selection_plan.csv`](../../data/manifests/community_forensics_selection_plan.csv) | 24,000 | `b7818aa4c27d9c7d35ae2910dea54b1d8eb111c5c5e00b8578eebae3c80788b4` |
| [`community_forensics_train.csv`](../../data/manifests/community_forensics_train.csv) | 18,000 | `fdfddade1b8b400183ec03aaa912f203f5fb1edc28bad6e92709372a4340dc55` |
| [`community_forensics_val_unseen_generator.csv`](../../data/manifests/community_forensics_val_unseen_generator.csv) | 2,000 | `11bfa4b6d7c538ce0a3d774c3f2902ac11ffc7dbe513de354f87fbaad1d6b6ba` |
| [`community_forensics_test_external_seen_family.csv`](../../data/manifests/community_forensics_test_external_seen_family.csv) | 2,000 | `c55d0652d4d95e17bf2c10c6a8dde2bdde5a4e5ea18a05baa22def0a669d1b42` |
| [`community_forensics_test_external_unseen_generator.csv`](../../data/manifests/community_forensics_test_external_unseen_generator.csv) | 2,000 | `70434fc7b38ed2015cac67bc87897139fb202ffa13e736a5ce2f7c759833e42a` |
| [`community_forensics_audit.json`](../../data/manifests/community_forensics_audit.json) | — | `6d51fd0b2ce5bf097ac6cc14ed44a643a030d76dc8080a1e0cdb5d5e66c6b10e` |

本次冻结物化 24,000 张图像，共 23,419,473,481 bytes。审计记录显示 split 间无 SHA-256 精确重复，并且 pHash 汉明距离阈值 4 下无近重复。当前缺少 COCO val2017/DALL-E Advanced 保留集的 hash 清单，因此只能确认官方 split/source 约束，**不能确认与该外部保留集不存在 hash 重叠**。

### 3.3 train-v1 构建器的阶段与恢复状态

入口脚本是 [`scripts/build_community_forensics.py`](../../scripts/build_community_forensics.py)，实际实现在 [`src/repostguard/data/community_forensics.py`](../../src/repostguard/data/community_forensics.py)。构建顺序为：

1. 解析两个固定 Hugging Face revision，远程扫描 Parquet 元数据列；
2. 把来源文件、row group、row index 和规范化生成器信息写入 `data/state/community_forensics.sqlite3`；
3. 使用种子 `20260828` 冻结 24,000 行选择计划；
4. 只读取被选择样本所在的 row group，并原子写入 `data/raw/community_forensics/`；
5. 解码验证图像，记录尺寸、格式、byte size、SHA-256 与 pHash；
6. 对精确重复和跨 split pHash 近重复执行同 split/同生成器或同真实来源的确定性替换；
7. 把被替换文件移入 `data/quarantine/` 并记录 repair JSON；
8. 写出四个冻结清单、审计 JSON 和 `data/raw/community_forensics/COMPLETE`。

中断恢复依赖 **SQLite 状态与已经物化的原图同时存在且一致**。作业收到超时信号后在安全边界退出，SLURM 包装器校验状态并最多自动续提 8 次。已下载文件会按 SHA-256 复用，不必从头开始。

### 3.4 train-v1 首次构建或完整重新下载

所有 Python、下载、测试和图像处理必须通过 Compute Node 的 `sbatch` 执行。提交前先查看当前资源限制：

```bash
cd /home/msai/lius0131/AGI/repostguard-lite
mytcinfo
```

随后先提交 Compute Node 预检/测试作业，再让准备作业依赖其成功状态：

```bash
CHECK_JOB_ID=$(sbatch --parsable scripts/slurm/validate_community_forensics.sbatch)
sbatch --dependency=afterok:${CHECK_JOB_ID} \
  --export=NONE,CF_PREP_RESTART_COUNT=0 \
  scripts/slurm/prepare_community_forensics.sbatch
```

当前准备脚本申请 `MGPU-TC2 / normal`、8 CPU、30 GB memory、5:50:00，不申请 GPU；这些值在每次提交前仍需以 `mytcinfo` 和当前 QoS 为准。脚本使用 `data/cache/huggingface` 作为 Hugging Face 缓存，日志写入 `logs/rg_cf_data_<jobid>.{out,err}`。

完整重新下载前必须区分两种状态：

- **下载中断但原图与 SQLite 均保留：** 直接重提相同准备作业，让状态机断点续传；
- **原图已经删除但旧 SQLite 仍存在：** 不得把旧数据库当成可恢复状态。旧记录可能仍将缺失图像标为 `complete`，必须先把旧 SQLite、`COMPLETE` 标记和残留原图作为同一组移到备份位置，或改用全新的 data root/state database，再开始完整构建。

不要删除 Git 中的冻结 CSV、audit 和 repair JSON；它们是重建后核对身份的基准。重建完成后必须检查：

```bash
sacct -j <JOB_ID> --format=JobID,State,ExitCode,Elapsed
sha256sum data/manifests/community_forensics_*.csv
cat data/raw/community_forensics/COMPLETE
```

注意：现有 [`prepare_community_forensics.sbatch`](../../scripts/slurm/prepare_community_forensics.sbatch) 在数据成功后会自动提交 train-v1 的模型训练链。如果只想恢复数据而不训练模型，应先增加 data-only 开关或使用不包含末尾训练提交逻辑的专用 SLURM 包装器，不能把自动提交行为忽略掉。

## 4. train-v2：角色提升构建与重新下载

train-v2 的核心操作是 **manifest promotion，而不是图像复制或重新下载**：

```text
train-v1 train：                  18,000（Real 9,000 / AIGI 9,000）
原 External seen-family 提升：    2,000（Real 1,000 / AIGI 1,000）
train-v2：                        20,000（Real 10,000 / AIGI 10,000）
```

提升后的 2,000 行保留原 `sample_id` 和物理 `path`，只在派生清单中更新 `split`、`project_split` 和 exposure。train-v2 不产生训练图像副本，因而相对于 v1 基础池几乎不增加图像存储。

### 4.1 train-v2 生成器协议变化

- train-v2 覆盖 909 个精确生成器和 GAN、LatDiff、PixDiff 三个训练架构大类；
- 新提升的 9 个精确生成器为 `decidiffusionv2`、`dfgan`、`galip`、`hourglass`、`kandinsky-2-2`、`kvikontent-midjourney-v6`、`lcm-lora-sdv15`、`lcm-lora-sdxl`、`lcm-lora-ssd1b`；
- 原 `test_external_seen_family` 从此只保留作血缘证据，禁止用于 train-v2/v3 测试；
- 冻结 strict unseen 的 12 个精确生成器及 Commercial/Other 架构仍与 train-v2 不相交；
- Hourglass、DFGAN、GALIP 三个困难切片的图像不变，但 exposure 被重新标为 `exact_seen_hard`。

### 4.2 train-v2 制品与校验

| 制品 | 行数 | SHA-256 |
|---|---:|---|
| [`community_forensics_train_v2.csv`](../../data/manifests/community_forensics_train_v2.csv) | 20,000 | `493749f51f552aa0a5235253f8c072b018a0f8250f02c94082aecdd98005ecd7` |
| [`community_forensics_train_v2_audit.json`](../../data/manifests/community_forensics_train_v2_audit.json) | — | `6f529ab2cbdc0f6084057192f1e18edf59bcda7835c506b506f9cfcde25d5fa3` |
| [`community_forensics_val_hard_hourglass_v2_exact_seen.csv`](../../data/manifests/community_forensics_val_hard_hourglass_v2_exact_seen.csv) | 500 | `6acf2986cfa9f1d9a8c7aedb52b82b406958bc552ddefc29d85b4481395fa4a6` |
| [`community_forensics_val_hard_dfgan_v2_exact_seen.csv`](../../data/manifests/community_forensics_val_hard_dfgan_v2_exact_seen.csv) | 500 | `b54846aec698e7386d6c26fdd70a50e39a132d20138891552870b6b2569d290b` |
| [`community_forensics_val_hard_galip_v2_exact_seen.csv`](../../data/manifests/community_forensics_val_hard_galip_v2_exact_seen.csv) | 500 | `9e20b9d04d5cdba9e4b53158f1834123f5864dcb23acc7e66d4a6b53c6eeca73` |

train-v2 审计要求 20,000 行全部具有唯一的 `sample_id`、`path`、SHA-256 和源定位符；训练集与 strict unseen 在图像身份、精确生成器和架构大类三个层面均不得重叠。成功后写入 `data/raw/community_forensics/TRAIN_V2_COMPLETE`。

### 4.3 train-v2 的下载依赖：validation-v2

train-v2 promotion 本身不下载图像，但当前构建器还会生成三个重新标注后的困难切片，因此要求 validation-v2 已完成。validation-v2 额外物化 3,000 张诊断图像，共 2,898,615,469 bytes：

| validation-v2 组成 | 数量 | 来源与作用 |
|---|---:|---|
| SD14 AIGI | 1,000 | AIGIBench；external exact-seen |
| external exact-seen Real | 1,000 | AIGIBench |
| Hourglass AIGI | 250 | CommunityForensics-Eval 困难切片 |
| DFGAN AIGI | 250 | CommunityForensics-Eval 困难切片 |
| GALIP AIGI | 250 | CommunityForensics-Eval 困难切片 |
| 三切片共享 Real panel | 250 | CommunityForensics-Eval；三个困难清单共同引用 |

validation-v2 通过独立 SQLite 状态 `data/state/community_forensics_validation_v2.sqlite3` 断点续传，并依赖 v1 的 `data/state/community_forensics.sqlite3` 取得 Eval 候选元数据。构建命令为：

```bash
mytcinfo
sbatch --export=NONE,CF_VAL2_RESTART_COUNT=0 \
  scripts/slurm/prepare_community_forensics_validation_v2.sbatch
```

当前包装器申请 8 CPU、30 GB memory、5:50:00，不申请 GPU；完成标记为 `data/raw/community_forensics/VALIDATION_V2_COMPLETE`。

### 4.4 train-v2 构建与重新下载流程

在 v1 基础池和 validation-v2 均通过校验后，提交：

```bash
sbatch scripts/slurm/prepare_community_forensics_train_v2.sbatch
```

该 CPU 作业当前申请 2 CPU、8 GB memory、20 分钟，执行 [`scripts/build_community_forensics_train_v2.py`](../../scripts/build_community_forensics_train_v2.py)，不访问 Hugging Face，也不复制图像。它会：

1. 校验 v1 `COMPLETE`、validation-v2 `VALIDATION_V2_COMPLETE` 和四个冻结基础制品；
2. 合并 v1 train 18,000 行与 retired seen-family 2,000 行；
3. 生成 20,000 行 train-v2 派生清单；
4. 把三个困难清单的 AIGI exposure 改为 `exact_seen_hard`；
5. 检查类别平衡、唯一性、图像路径/大小、strict-unseen 不相交和 hard-slice 不重叠；
6. 原子写入 train-v2 audit 与 `TRAIN_V2_COMPLETE`。

从完全空的数据目录恢复 train-v2 时，正确顺序不是直接运行 train-v2 builder，而是：

```text
重建 train-v1 24k 基础池
    -> 重建 validation-v2 3k 诊断池
    -> 运行 train-v2 promotion（零新增训练图像下载）
    -> 对照冻结 SHA-256 与完成标记
```

如果只有 `community_forensics_train_v2.csv` 丢失、原始图像与 v1/validation-v2 清单仍完整，则只需重跑 train-v2 promotion；如果物理图像已删除，则必须先按 train-v1 和 validation-v2 的固定 revision 重新下载，train-v2 CSV 本身不会恢复缺失图像。

## 5. train-v3 训练集组成

### 5.1 完整训练集

完整清单：[`community_forensics_train_v3.csv`](../../data/manifests/community_forensics_train_v3.csv)

| 项目 | 数值 |
|---|---:|
| 总图像数 | 24,000 |
| Real | 12,000 |
| AIGI | 12,000 |
| `CommunityForensics-Small` 来源 | 22,000 |
| `CommunityForensics-Eval` 来源 | 2,000 |
| 清单 SHA-256 | `fc0a7ab732faeb604ed1e77281fada715d7cffb353974a4985820548d871d9d6` |

train-v3 继承的 train-v2 部分为 20,000 张，包括：

- train-v1 的 `CommunityForensics-Small` 18,000 张；
- 原 External seen-family 的 `CommunityForensics-Eval` 2,000 张；
- train-v2 清单 SHA-256：`493749f51f552aa0a5235253f8c072b018a0f8250f02c94082aecdd98005ecd7`；
- train-v2 共覆盖 909 个精确生成器；其中 Hourglass、DFGAN、GALIP 等 9 个原 seen-family 精确生成器已被正式提升到训练集。

### 5.2 v3 新增的 4,000 张

增量清单：[`community_forensics_train_v3_additions.csv`](../../data/manifests/community_forensics_train_v3_additions.csv)<br>
选择计划：[`community_forensics_train_v3_selection_plan.csv`](../../data/manifests/community_forensics_train_v3_selection_plan.csv)<br>
审计记录：[`community_forensics_train_v3_audit.json`](../../data/manifests/community_forensics_train_v3_audit.json)

| 新增组 | 数量 | 选取原则 |
|---|---:|---|
| GAN AIGI | 1,000 | 覆盖 Small 中全部 12 个可用 GAN 精确生成器，每类贡献 83–84 张 |
| PixDiff AIGI | 1,000 | 覆盖 Small 中全部 3 个可用 PixDiff 精确生成器，每类贡献 333–334 张 |
| GAN-matched Real | 1,000 | 确定性真实图像配额，来自独立 Parquet row group |
| PixDiff-matched Real | 1,000 | 确定性真实图像配额，来自另一独立 Parquet row group |

GAN 精确生成器共 12 个：

`biggan`、`cips`、`gansformer`、`gigagan`、`progan`、`projectedgan`、`stylegan2`、`stylegan2-ada`、`stylegan3`、`styleganxl`、`stylesanxl`、`styleswin`。

PixDiff 精确生成器共 3 个：

`deepfloyd`、`glide`、`guideddiffusion`。

补充说明：固定版本的 `CommunityForensics-Small` 把真实图像统一标注为 `architecture=Real`、`subset=Real`、`real_source=N/A`，无法事实性地恢复更细的真实来源类型。因此当前“两组对应真实图像”是两个确定性、相互独立的配额，而不是虚构的真实来源标签。

| 增量制品 | 行数 | SHA-256 |
|---|---:|---|
| `community_forensics_train_v3_additions.csv` | 4,000 | `fec5097fd1a3a5488851caddf729bf3470916024d0dcf5533efddb20280ae2bd` |
| `community_forensics_train_v3_selection_plan.csv` | 4,000 | `401c0157d079a1b0310853b5ac8d0567cb6a0709233931db487073c2249ce709` |

增量图像已物化约 1.70 GB（1,703,324,237 bytes）。审计记录显示所有路径均已验证，增量与冻结清单在 `path`、`sample_id`、`sha256`、源定位符四个维度的精确重叠均为 0，pHash 汉明距离阈值 4 下的近重复数为 0。

## 6. expanded strict unseen-generator 测试集

完整清单：[`community_forensics_test_external_unseen_generator_v3_expanded.csv`](../../data/manifests/community_forensics_test_external_unseen_generator_v3_expanded.csv)<br>
新增清单：[`community_forensics_external_unseen_v3_additions.csv`](../../data/manifests/community_forensics_external_unseen_v3_additions.csv)<br>
选择计划：[`community_forensics_external_unseen_v3_selection_plan.csv`](../../data/manifests/community_forensics_external_unseen_v3_selection_plan.csv)<br>
审计记录：[`community_forensics_external_unseen_v3_audit.json`](../../data/manifests/community_forensics_external_unseen_v3_audit.json)

```text
冻结 strict unseen 基础集：2,000（Real 1,000 / AIGI 1,000）
v3 新增：                 2,000（Real 1,000 / AIGI 1,000）
最终 expanded 测试集：   4,000（Real 2,000 / AIGI 2,000）
```

新增 1,000 张真实图像按来源均衡抽取：COCO、FFHQ、LAION、RAISE 各 250 张。扩充后的 2,000 张真实图像在四个来源上各 500 张。

新增 AIGI 仅使用 `Commercial` 和 `Other` 两个训练未见架构大类，并覆盖固定 Eval revision 中可用的全部 12 个精确生成器：

`dalle2`、`dalle3`、`firefly-image2`、`firefly-image3`、`flux-dev`、`flux-schnell`、`ideogramv1`、`ideogramv2`、`imagen3`、`midjourneyv5-2`、`midjourneyv6-1`、`stable-cascade`。

新增部分每个精确生成器贡献 83–84 张。基础集与新增集合并后，每个精确生成器合计约 166–167 张；AIGI 架构分布为 Commercial 1,834 张、Other 166 张。

| 测试制品 | 行数 | SHA-256 |
|---|---:|---|
| 冻结 2k 基础清单 `community_forensics_test_external_unseen_generator.csv` | 2,000 | `70434fc7b38ed2015cac67bc87897139fb202ffa13e736a5ce2f7c759833e42a` |
| v3 新增清单 `community_forensics_external_unseen_v3_additions.csv` | 2,000 | `d5b662c1d67443c017b62bd133b61c78a615c68933d351fda01a10fee86c0e90` |
| v3 新增选择计划 | 2,000 | `9773ebcbf5f99840e085a8d79a93debf1b8b87ad7181641f4d8dbc661bd689dc` |
| expanded 4k 最终清单 | 4,000 | `59ca2e4ca966dac9fa4fb55281153f93e5becdd3e25da83bc2dff3fad36126cd` |

新增测试图像已物化约 7.03 GB（7,031,865,364 bytes）。审计记录显示所有路径均已验证，与所有冻结清单的四类精确标识重叠均为 0，pHash 汉明距离阈值 4 下的近重复数为 0。

## 7. 验证与困难切片

### 7.1 内部 checkpoint-selection 验证集

[`community_forensics_val_unseen_generator.csv`](../../data/manifests/community_forensics_val_unseen_generator.csv) 来自 `CommunityForensics-Small`，共 2,000 张，Real/AIGI 各 1,000 张。它是训练期间用于选择 checkpoint 的内部留出集，SHA-256 为：

`11bfa4b6d7c538ce0a3d774c3f2902ac11ffc7dbe513de354f87fbaad1d6b6ba`

该清单名称沿用早期命名；在报告中应把它称为“内部 checkpoint-selection 验证集”，避免与正式 external strict unseen-generator 测试集混淆。

### 7.2 external exact-seen

[`community_forensics_val_external_exact_seen_generator.csv`](../../data/manifests/community_forensics_val_external_exact_seen_generator.csv) 来自 `TheKernel01/AIGIBench`，共 2,000 张：

- 1,000 张 SD14 AIGI；
- 1,000 张真实图像；
- SD14 被规范化为 `compvis/stable-diffusion-v1-4`；
- 与训练集属于同一精确生成器身份，但样本与来源数据集不同；
- SHA-256：`fc4bbd7d05c10f6b40b3009edd4213acd6e6cf4de5085930642353a7fc950ac3`。

### 7.3 三个困难生成器

每个困难切片由 250 张对应生成器 AIGI 与同一组 250 张共享真实图像组成。train-v2 已将这些精确生成器的另一批图像加入训练，因此它们在 train-v3 协议中属于 **exact-seen 困难切片**，不是 unseen-generator。

| 切片 | 行数 | 当前配置清单 SHA-256 |
|---|---:|---|
| Hourglass | 500 | `6acf2986cfa9f1d9a8c7aedb52b82b406958bc552ddefc29d85b4481395fa4a6` |
| DFGAN | 500 | `b54846aec698e7386d6c26fdd70a50e39a132d20138891552870b6b2569d290b` |
| GALIP | 500 | `9e20b9d04d5cdba9e4b53158f1834123f5864dcb23acc7e66d4a6b53c6eeca73` |

验证数据构建审计见 [`community_forensics_validation_v2_audit.json`](../../data/manifests/community_forensics_validation_v2_audit.json)。

## 8. Hugging Face 来源与固定版本

| 来源数据集 | 固定 revision | 许可证 | 当前用途 |
|---|---|---|---|
| `OwensLab/CommunityForensics-Small` | `6c539a534c07917307c381f5af4053c6091b5278` | `CC-BY-NC-SA-4.0` | train-v1、train-v3 新增样本、内部验证集 |
| `OwensLab/CommunityForensics-Eval` | `7d4a74a88d2cac93b513c0853bf92c260eaceea0` | `CC-BY-NC-SA-4.0` | train-v2 promoted 数据、strict unseen 基础集与 v3 扩充集、困难切片 |
| `TheKernel01/AIGIBench` | `f125eabc5ac34a4729d74adc1aa1214540f91947` | 以源仓库声明为准 | external exact-seen SD14 诊断集 |

使用固定 revision 而不是默认分支，可以避免上游更新导致样本漂移。数据使用和再分发仍必须遵守各上游数据集的许可证；尤其 `CommunityForensics` 当前记录为非商业、署名、相同方式共享许可。

## 9. 清单字段与样本定位方式

正式清单共有 24 列：

```text
sample_id,path,label,split,source_dataset,generator_id,official_split,
project_split,real_source,model_name_raw,canonical_generator_id,
architecture,generator_exposure,sha256,phash,selection_seed,
source_revision,source_file,source_row_group,source_row_index,
width,height,format,byte_size
```

关键字段含义：

| 字段 | 作用 |
|---|---|
| `path` | 相对 [`data/raw/community_forensics`](../../data/raw/community_forensics) 的本地图像路径 |
| `label` | `0` 表示 Real，`1` 表示 AIGI |
| `source_dataset` | 规范化后的上游数据集标识 |
| `source_revision` | 上游 Hugging Face 固定 commit/revision |
| `source_file` | 上游仓库中的 Parquet 文件路径 |
| `source_row_group` / `source_row_index` | 在 Parquet 中精确定位原始行 |
| `canonical_generator_id` | 规范化后的精确生成器身份 |
| `architecture` | 生成器架构大类或真实图像来源类别 |
| `generator_exposure` | 该生成器相对训练协议的 seen/unseen 状态 |
| `sha256` | 检查文件内容是否与冻结清单一致 |
| `phash` | 检查跨集合视觉近重复 |
| `selection_seed` | 复现确定性抽样过程 |
| `width` / `height` / `format` / `byte_size` | 检查物化图像的基本属性 |

对 CommunityForensics 样本，远端定位逻辑等价于：

```text
datasets/<dataset_id>@<source_revision>/<source_file>
    -> source_row_group
    -> source_row_index
    -> image_data
```

下载后应同时验证文件大小与 SHA-256；仅凭本地 `path` 存在不能证明内容正确。

## 10. 从空目录重建的依赖顺序

若原始图像和 `data/state/*.sqlite3` 均已删除，建议按下列顺序重建：

1. 使用固定 revision 扫描 `CommunityForensics-Small` 和 `CommunityForensics-Eval`，重建基础元数据数据库、train-v1、内部验证集和冻结 external 清单；
2. 构建 validation-v2：从 AIGIBench 物化 external exact-seen，并生成 Hourglass、DFGAN、GALIP 困难切片；
3. 构建 train-v2：将原 External seen-family 2,000 张加入训练，同时保留旧清单作为血缘记录；
4. 构建 train-v3：依据选择计划从 Small 下载并物化新增 4,000 张，再与 train-v2 合并；
5. 构建 expanded external unseen v3：在冻结 2k 基础测试集上新增 Eval 的 2,000 张；
6. 重新执行 SHA-256、源定位符、pHash 与跨 split 重叠审计；
7. 只有全部检查通过后，重新生成 `COMPLETE`、`VALIDATION_V2_COMPLETE`、`TRAIN_V2_COMPLETE`、`TRAIN_V3_COMPLETE`、`EXTERNAL_UNSEEN_V3_COMPLETE` 标记。

相关构建入口：

- [`scripts/build_community_forensics.py`](../../scripts/build_community_forensics.py)
- [`src/repostguard/data/community_forensics.py`](../../src/repostguard/data/community_forensics.py)
- [`scripts/build_community_forensics_validation_v2.py`](../../scripts/build_community_forensics_validation_v2.py)
- [`src/repostguard/data/community_forensics_validation_v2.py`](../../src/repostguard/data/community_forensics_validation_v2.py)
- [`scripts/build_community_forensics_train_v2.py`](../../scripts/build_community_forensics_train_v2.py)
- [`scripts/build_community_forensics_train_v3.py`](../../scripts/build_community_forensics_train_v3.py)
- [`scripts/build_community_forensics_external_unseen_v3.py`](../../scripts/build_community_forensics_external_unseen_v3.py)

v3 的两个增量脚本依赖基础 `data/state/community_forensics.sqlite3`、前一版本清单及完成标记。`data/raw/` 和 `data/state/` 均被 Git 忽略，因此 GitHub 仓库只保存清单、审计记录、配置与构建代码，不保存原图和状态数据库。

如数据集需要认证，可在提交下载作业时通过安全环境变量提供 Hugging Face Access Token；不要把 token 写进 YAML、脚本、日志、Markdown 或 Git。Token 可以改善认证和限流体验，但不会自动改变网络带宽，也不能替代固定 revision 与完整性校验。

## 11. 能否仅凭当前清单重新下载

结论分为两层：

1. **数据可追溯性：可以。** 当前 CSV 与 audit JSON 已提供重定位原始样本所需的主要来源字段，并提供内容校验值。在上游固定 revision 仍可访问且字段结构未改变的条件下，可以恢复同一批图像。
2. **现有一键工具：尚不完全具备。** 当前构建脚本按 v1 → validation-v2 → train-v2 → train-v3 → external-v3 的状态机运行，不能直接把任意最终 CSV 当作唯一输入完成所有下载。

若希望未来只保留 GitHub 中的清单即可恢复数据，建议补充一个通用 `download_from_manifest.py`：按 `source_dataset + source_revision + source_file + source_row_group + source_row_index` 分组下载，支持断点续传、原子写入、文件大小/SHA-256 校验和最终恢复审计。该工具必须通过 SLURM 作业在 Compute Node 执行，不能在 Head Node 直接运行。

## 12. 复现检查清单

- [ ] 使用本文记录的固定 Hugging Face revision；
- [ ] 验证所有最终 CSV 的 SHA-256；
- [ ] 验证每个本地图像的 `byte_size` 与 `sha256`；
- [ ] 验证 `sample_id`、`path`、`sha256`、源定位符在要求的 split 间无精确重叠；
- [ ] 以 pHash 汉明距离 4 检查跨 split 近重复；
- [ ] 保持 label 语义：Real=`0`、AIGI=`1`；
- [ ] checkpoint 只能由内部验证集选择，external exact-seen、困难切片和 strict unseen 仅用于冻结后的诊断或终点评测；
- [ ] 不再使用 retired External seen-family 作为测试集；
- [ ] 记录重建时间、脚本版本、清单哈希、成功/失败行和上游异常；
- [ ] 遵守上游许可证，不在仓库中提交原始数据、模型权重、token 或状态数据库。
