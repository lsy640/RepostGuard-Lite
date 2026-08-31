# V3.1 T=3 baseline：M3-only

这是 V3.1 family-unseen 消融的 T=3 基线完整交付。Student 保持 V3.0 的 MobileNetV3-Large 架构，以 M3 为唯一教师；训练时暂时保留 19 个生成器 families 作为开发集，以观察未见 family 泛化。

## 配置与 checkpoint

| 项目 | 值 |
|---|---|
| M2 / M3 teacher | 0% / 100% |
| Student 参数量 | 4,203,313 |
| KD temperature | 3.0 |
| Best checkpoint | epoch 16 / global step 8240 |
| Latest checkpoint | epoch 20 / global step 10300 |
| Best internal Clean AUROC | 0.791871 |

实际配置以 [`resolved_config.yaml`](resolved_config.yaml) 为准。`best.pt` 用于下述评测，`latest.pt` 保留完整训练终点。

## Family-unseen 开发评测

| 指标 | 值 |
|---|---:|
| Clean AUROC | 0.791871 |
| Clean balanced accuracy | 0.721058 |
| Robust mean AUROC | 0.755244 |
| Robust mean balanced accuracy | 0.655072 |
| Robust worst AUROC | 0.689937 |
| Worst condition | `center_crop ratio=0.8 + jpeg quality=50` |

完整结果和逐样本预测位于 [`dev_family_unseen_robustness/`](dev_family_unseen_robustness/)。本运行没有生成 ONNX/TorchScript；目录中没有移动端导出属于真实产物边界。

## 核心文件 SHA-256

| 文件 | Bytes | SHA-256 |
|---|---:|---|
| [`best.pt`](best.pt) | 50,812,648 | `7b20def1750c862e43702dd43eb1519e4fcd953826cad4905a33fa554e442ec4` |
| [`latest.pt`](latest.pt) | 50,812,648 | `2344f8ed9efdbdce2074c0626cf30242274d0d5799295ab88628569d903043af` |
| [`resolved_config.yaml`](resolved_config.yaml) | 3,308 | `adb39990bc44a23d7af6dc72239d529eb9edacb058279a1172d818f11501934d` |

全部文件校验和见 [`SHA256SUMS.txt`](SHA256SUMS.txt)。
