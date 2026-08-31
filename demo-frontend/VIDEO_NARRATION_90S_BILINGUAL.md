# AIGI Detect Demo：90 秒录屏口播稿（中英双语）

## 录制参数

- 目标时长：约 1 分 30 秒
- 主讲语言：英语
- 英文正文：约 203 词
- 建议语速：每分钟 135–140 词，清晰、稳定，功能名称处稍作停顿
- 建议页面语言：先在右上角切换为 `EN`

## 分镜与口播

### 00:00–00:13｜主界面

**画面操作**

从页面顶部开始录制。展示 `Tiktok TechJam 2026`、`AIGI Detect Demo`、语言切换、M2/M3 切换和本地设备状态。点击 `EN`，然后缓慢向下移动鼠标，让观众看到四个功能区域。

**English voice-over**

Welcome to AIGI Detect Demo, our fully local image-forensics interface for TikTok TechJam 2026. The dashboard brings single-image detection, robustness analysis, forensic evidence, and batch inference into one workflow.

**中文对照**

欢迎来到 AIGI Detect Demo，这是我们为 TikTok TechJam 2026 打造的完全本地图像取证界面。该仪表盘将单图检测、鲁棒性分析、取证证据和批量推理整合到一个工作流中。

### 00:13–00:31｜01 输入与结果：Real 与 AIGC 各一张

**画面操作**

1. 点击 `Select image`，选择提前准备好的真实照片 `01_real.jpg`，停留 3–4 秒，展示 `REAL` 标签、Raw AIGC Score、Research Calibrated、Score Uncertainty 和 Local Latency。
2. 将提前准备好的 AIGC 图片 `02_aigc.png` 拖入上传区域，停留 3–4 秒，展示标签切换为 `AIGC`。

**English voice-over**

In Input and Result, I first select a real photograph. The model returns a Real label, together with raw and research-calibrated AIGC scores, uncertainty, and local latency. Next, I drag in an AI-generated image, and the result changes to AIGC.

**中文对照**

在“输入与结果”区域，我先选择一张真实照片。模型返回 Real 标签，同时展示原始 AIGC 分数、研究校准分数、不确定性和本地推理耗时。接下来，我拖入一张 AI 生成图片，结果随即变为 AIGC。

### 00:31–00:43｜M2/M3 模型切换

**画面操作**

保持 AIGC 图片不变，先指向当前选中的 `M2`，再点击 `M3`。等待同一图片自动重新推理，并展示页面顶部的 `Active model M3` 与更新后的结果。不要在这一段重新上传图片。

**English voice-over**

M2 is selected by default. With one click, I can switch to M3; the same image is re-analysed automatically, making model comparison fast and consistent.

**中文对照**

系统默认选择 M2。只需点击一次即可切换到 M3；同一张图片会被自动重新分析，从而快速、一致地比较两个模型。

### 00:43–01:11｜02 鲁棒性实验台与 03 取证证据

**画面操作**

滚动到 `Robustness Lab`。依次短暂启用六项功能：`JPEG Compression`、`Gaussian Blur`、`Resize Roundtrip`、`Gaussian Noise`、`Color Jitter` 和 `Center Crop`；每项至少让开关或滑块在画面中清晰变化一次。随后展示 Clean 与 Perturbed 图片对比、Raw Score Δ、Label Flip、History Range 和响应曲线。最后滚动到 `Evidence & Limits`：先展示 M3 的语义/取证分支权重，再依次指向 SRM 与 NPR 热图，并将 `Color` 短暂切换为 `Raw`，让观众看到两种残差显示方式。

**English voice-over**

In the Robustness Lab, I can test JPEG compression, blur, resize, noise, color jitter, and cropping. Each change updates the score, label-flip indicator, and response history. The evidence panel then explains the forensic signal: SRM uses thirty high-pass filters to expose residual patterns, while NPR compares the image with a nearest-neighbor reconstruction to reveal resampling residuals. These heatmaps are diagnostic clues, not pixel-level proof of manipulation.

**中文对照**

在鲁棒性实验台中，我可以测试 JPEG 压缩、模糊、缩放、噪声、色彩扰动和裁剪。每次调整都会更新分数、标签翻转指示和响应历史。证据面板会进一步解释取证信号：SRM 使用 30 个高通滤波器提取残差模式；NPR 则将图像与最近邻重建结果进行比较，以显示重采样残差。这些热图提供的是诊断性线索，而不是像素级的图像篡改证明。

### 01:11–01:30｜04 批量文件推理与结尾

**画面操作**

滚动到 `Batch Import & Standard JSON`。点击 `Import files` 选择 3–5 张图片，或点击 `Import folder` 导入提前准备好的小文件夹。点击 `Start batch inference with M3`，展示进度、成功/失败数量和 JSON 预览；任务完成后，将鼠标移到 `Download JSON`，并让 `image_path` 与 `pred` 字段清晰可见。最后停留在完成状态 1–2 秒。

**English voice-over**

Finally, Batch Import supports multiple files or an entire folder, with up to one hundred images per job. After inference, the interface reports progress and exports standard JSON containing image_path and raw pred values. This supports both individual inspection and scalable local evaluation.

**中文对照**

最后，批量导入支持多个文件或整个文件夹，每个任务最多处理一百张图片。推理完成后，界面会显示处理进度，并导出包含 `image_path` 和原始 `pred` 值的标准 JSON。该功能同时支持单图检查和可扩展的本地评测。

## 英文连续口播版

Welcome to AIGI Detect Demo, our fully local image-forensics interface for TikTok TechJam 2026. The dashboard brings single-image detection, robustness analysis, forensic evidence, and batch inference into one workflow.

In Input and Result, I first select a real photograph. The model returns a Real label, together with raw and research-calibrated AIGC scores, uncertainty, and local latency. Next, I drag in an AI-generated image, and the result changes to AIGC.

M2 is selected by default. With one click, I can switch to M3; the same image is re-analysed automatically, making model comparison fast and consistent.

In the Robustness Lab, I can test JPEG compression, blur, resize, noise, color jitter, and cropping. Each change updates the score, label-flip indicator, and response history. The evidence panel then explains the forensic signal: SRM uses thirty high-pass filters to expose residual patterns, while NPR compares the image with a nearest-neighbor reconstruction to reveal resampling residuals. These heatmaps are diagnostic clues, not pixel-level proof of manipulation.

Finally, Batch Import supports multiple files or an entire folder, with up to one hundred images per job. After inference, the interface reports progress and exports standard JSON containing image_path and raw pred values. This supports both individual inspection and scalable local evaluation.

## 录制前检查清单

- 使用 `http://localhost:8000` 打开页面并确认标题为 `AIGI Detect Demo`。如果 `127.0.0.1:8000` 意外显示旧页面，请改用 `localhost` 或执行强制刷新后再录制。
- 提前准备并重命名两张单图：`01_real.jpg` 与 `02_aigc.png`。正式录制前分别在 M2 和 M3 下试跑，确认演示图片得到预期的 `REAL` 与 `AIGC` 标签。
- 不要使用 `reports/assets/error_analysis/` 中以 `fp_` 或 `fn_` 开头的图片作为正向演示样例；这些文件本身是已知误报或漏报案例。
- 录制前分别加载一次 M2 和 M3，让两个 checkpoint 预热完成，避免首次加载模型占用过多镜头时间。
- 将鲁棒性参数恢复到初始状态，再开始正式录屏；录制时按页面显示的执行顺序操作。
- 批量演示建议只放 3–5 张图片，以确保任务在 90 秒内完成，同时保留进度、结果预览和 `Download JSON` 按钮的展示时间。
- 英文读法建议：`M2` 读作 “M two”，`M3` 读作 “M three”，`AIGC` 逐字母读作 “A-I-G-C”，`SRM` 与 `NPR` 也逐字母读。
- 口播时不要朗读具体分数，因为不同样例和模型的结果会变化；让画面承担具体数值展示即可。
