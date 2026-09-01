# RepostGuard Lite V3.2.1 — Galaxy S23 Ultra 测试交付

- APK：`RepostGuardLite-V3.2.1-S23Ultra-debug.apk`
- 包名：`ai.repostguard.demo.debug`
- 版本：`3.2.1-demo-en-debug`（versionCode 3）
- 默认语言：English；简体中文作为 `zh-rCN` 可选本地化
- 大小：63,831,578 bytes（60.87 MiB）
- ABI：仅 `arm64-v8a`
- 最低 Android：API 28
- 目标 Android：API 35
- 模型：V3.2.1 epoch 3 / global step 1545
- 参数量：7,955,038
- 模型格式：ONNX FP32，未量化
- 运行时：ONNX Runtime Android CPU，4 intra-op threads
- 冻结判定阈值：0.060516357421875

## 已完成门禁

- Gradle 离线构建通过。
- Android lint 为 0 issues。
- APK 中实际嵌入的 ONNX SHA-256 与最终 V3.2.1 导出一致。
- APK 仅打包 `arm64-v8a` ONNX Runtime native libraries。
- 合并后的 APK manifest 没有 Internet 或存储权限。
- ONNX Runtime CPU parity 通过：最大 probability error 0.000427231，最大 gate error 0.000653714。

## 页面

1. 输入与结果：图像选择、AIGC 概率信号、冻结阈值标签、阈值距离不确定性 proxy。
2. 鲁棒性实验台：JPEG、blur、resize、noise、jitter、crop，并排图像与分数变化。
3. 证据与限制：语义/取证门控比例，SRM-like/NPR proxy 热图，以及误报和非溯源提示。

## 安装

手机开启开发者选项和 USB 调试后，在此目录执行：

```powershell
D:\projects\tiktok26\.android-sdk\platform-tools\adb.exe install -r .\RepostGuardLite-V3.2.1-S23Ultra-debug.apk
```

也可以把 APK 复制到手机后手动安装。该 APK 使用本机 debug key 签名，只用于测试，不用于商店发布。

真机尚未通过 ADB 连接，因此当前没有 Galaxy S23 Ultra 的实测推理延迟。App 会分别显示预处理、模型推理和总耗时；首次推理通常还包含 runtime warm-up，正式记录建议先运行 3 次，再统计后续 10 次。

配套的 30 张盲测图片包位于同一目录 `phone_test_images_v1.zip`。建议先只把 `blind/` 放进手机，测试完成后再查看 `answer_key.csv`。
