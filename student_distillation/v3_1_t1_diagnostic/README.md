# V3.1 T=1 diagnostic：M3-only

这是只把 V3.1 baseline 的 KD temperature 从 3 改为 1 的诊断运行，用于检查“校准后再次施加温度”是否是性能下降来源。它保留旧采样与旧校准设置，不是最终 corrected 方法。

## 配置与结果边界

| 项目 | 值 |
|---|---|
| M2 / M3 teacher | 0% / 100% |
| Student 参数量 | 4,203,313 |
| KD temperature | 1.0 |
| Best checkpoint | epoch 15 / global step 7725 |
| Latest checkpoint | epoch 20 / global step 10300 |
| Best internal Clean AUROC | 0.783724 |
| Best internal balanced accuracy | 0.715569 |

T=1 单独改动没有超过同协议 T=3 baseline（0.791871 Clean AUROC），说明温度并非唯一问题；这一结论推动了 V3.2 同时修正采样、校准并加入 feature/forensic distillation。

本运行只完成训练诊断，没有同口径 18-condition robustness、逐样本预测或移动端导出。`best.pt`、`latest.pt`、`DONE` 和冻结配置均完整保留，但不能把缺失评测写成已完成结果。

## 核心文件 SHA-256

| 文件 | Bytes | SHA-256 |
|---|---:|---|
| [`best.pt`](best.pt) | 50,812,648 | `d295473b1e3739d71f7ff6d5437927b32e885b21fb100d68353f558e289400e5` |
| [`latest.pt`](latest.pt) | 50,812,648 | `ce84c89dbdce4e6474f69d7fb1c0331412bf203fbba55d626eebfb0dc43f9a97` |
| [`resolved_config.yaml`](resolved_config.yaml) | 3,311 | `8ce12c8192118853788c2bd99551741baf3a1d7dc49c09f6aafcd80fa7e9d61d` |

全部文件校验和见 [`SHA256SUMS.txt`](SHA256SUMS.txt)。
