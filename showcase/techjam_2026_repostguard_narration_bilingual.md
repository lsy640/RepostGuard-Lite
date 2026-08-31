# RepostGuard-Lite — TikTok TechJam 2026 两分钟双语口播稿

> 核心英文口播：**232 words**（只统计场景 1–5、7、9）
>
> 核心计时：**02:00**
>
> 场景 6 与场景 8 是计时外 MP4 插播，不计入 232 词或 02:00。最终成片总时长会长于两分钟。

## 录制节奏总览

| 核心时间 | 场景 | 核心时长 | 录制规则 |
|---|---|---:|---|
| `00:00–00:10` | 1. Opening | 10 s | 核心口播 |
| `00:10–00:30` | 2. The Real-World Gap | 20 s | 核心口播 |
| `00:30–00:55` | 3. Our Architecture | 25 s | 核心口播 |
| `00:55–01:15` | 4. Dataset Strategy | 20 s | 核心口播 |
| `01:15–01:42` | 5. Results | 27 s | 核心口播 |
| `Video A` | 6. M2/M3 Detection Demo | 不计时 | 停止核心口播，完整播放 `demo cv.mp4` |
| `01:42–01:55` | 7. Distillation | 13 s | 核心口播 |
| `Video B` | 8. Android Prototype | 不计时 | 当前仅占位；有视频后完整播放 |
| `01:55–02:00` | 9. Closing | 5 s | 核心口播并停顿收尾 |

---

<!-- CORE-NARRATION-START -->

## Scene 1 — Opening

**Core time:** `00:00–00:10`

**画面动作**

- 第一页出现后停半秒，再开始讲话。
- 说到 “after reposting” 时，让视线落到右侧 Real/AIGI 两张 ground-truth hard cases 和中间的 repost 标记。

**English narration**

> Hello everyone. We present RepostGuard-Lite for TikTok TechJam 2026: AI-generated image detection designed for the transformations of real social sharing.

**中文说明**

向评委问好并直接定位项目：这不是只对干净原图有效的检测器，而是面向真实社交传播变换的 AIGI 检测方案。右侧图片只表达 ground truth，不把已知困难样本描述为模型成功案例。

**发音提示**

- `RepostGuard-Lite`：**REE-post guard light**
- `AI-generated`：**A-I generated**

## Scene 2 — The Real-World Gap

**Core time:** `00:10–00:30`

**画面动作**

- 依次扫过 `Generate → Edit → Resize → JPEG → Repost`。
- 后半段把注意力移到 AIDE 原论文 Figure 2；不要逐模块朗读架构图。

**English narration**

> Detection is hardest after generation. Editing, resizing, JPEG compression, blur, noise, and repeated reposting can erase forensic traces. AIDE showed why hybrid evidence matters by combining semantic context with patch-level signals.

**中文说明**

核心困难发生在“生成之后”：压缩、缩放、模糊、噪声和编辑会破坏低层取证痕迹。AIDE 只作为“语义证据 + 局部取证证据”的研究启发，不把 RepostGuard-Lite 说成 AIDE 的直接复现。

**发音提示**

- `JPEG`：**JAY-peg**
- `AIDE`：读作英文单词 **aid**
- `forensic`：**fə-REN-sik**

## Scene 3 — Our Architecture

**Core time:** `00:30–00:55`

**画面动作**

- 先指向 OpenCLIP 语义分支，再指向 DCT/SRM/NPR 取证分支。
- 随后扫到 paired clean/degraded objective，最后落到 M2/M3 融合区和右侧参数卡片。

**English narration**

> Building on that idea, our two-stream detector pairs a frozen OpenCLIP semantic branch with DCT, SRM, and NPR-inspired forensic features. M2 fuses both streams directly. Clean and degraded views share one model and consistency losses. M3 adds a six-dimensional quality gate. The full model has 99.4 million parameters—below 0.1 billion.

**中文说明**

双流架构同时提取高层语义和低层取证特征。M2 做静态融合；训练时让 clean/degraded 配对视图共用同一模型，并通过预测与特征一致性约束提高传播鲁棒性。M3 增加六维质量门控，但当前不宣称它稳定优于 M2。完整模型小于 0.1B 参数，仅占题目 2B 上限约 4.97%。

**发音提示**

- `OpenCLIP`：**open clip**
- `DCT / SRM / NPR`：逐字母读 **D-C-T / S-R-M / N-P-R**
- `M2 / M3`：**M-two / M-three**
- `0.1B`：**zero point one billion**

## Scene 4 — Dataset Strategy

**Core time:** `00:55–01:15`

**画面动作**

- 从 CIFAKE、SID-Set 两个 pilot 向 Community Forensics 主实验过渡。
- 只强调 `24K`、`921`、`4K`、`12` 四个数字，不朗读 atlas 中的密集生成器名称。

**English narration**

> We moved beyond the CIFAKE and SID-Set pilots to Community Forensics train-v3: 24 thousand training images from 921 exact generators, plus a four-thousand-image strict-unseen test covering 12 generators absent from training.

**中文说明**

说明为什么没有把官方建议 pilot 当作最终证据：主实验参考 Community Forensics 构建更强调精确生成器多样性和严格未见生成器的紧凑协议。`strict-unseen` 指 12 个测试生成器的精确身份均未进入训练集，而不是宣称覆盖所有未来生成器。

**发音提示**

- `CIFAKE`：建议读 **sigh-fake**
- `strict-unseen`：**strict unseen**
- `921`：**nine hundred and twenty-one**

## Scene 5 — Results That Survive Reposting

**Core time:** `01:15–01:42`

**画面动作**

- 先看左侧 CIFAKE B0 的 clean/worst 落差。
- 再切到右侧 train-v3 strict-unseen 柱形图，只读 M2 的两个关键数值。
- 结尾停在 M2 的 `deploy first` 和 M3 的 `candidate` 标签。

**English narration**

> The pilots warned us that clean results can mislead: CIFAKE B0 scored 0.9945 clean AUROC but only 0.5442 in its worst condition. On strict-unseen data, M2 reaches 0.9308 clean and 0.8525 after six-stage degradation, making M2 our current deployment choice. M3 remains a data-dependent research candidate.

**中文说明**

左、右两块采用不同协议，不能跨面板直接排名。CIFAKE 只用于说明 clean 高分可能掩盖退化；当前主结论来自 4K strict-unseen train-v3：M2 是部署优先选择。M3 的门控策略在部分较小训练域有效，但在 train-v3 上没有形成稳定优势，因此只称候选研究方向。

**发音提示**

- `AUROC`：建议读 **A-U-rock**
- `0.9308`：**zero point nine three zero eight**
- `0.8525`：**zero point eight five two five**

## Scene 6 — M2/M3 Detection Demo

**Timing:** `Video interlude — outside the 02:00 core`

**画面动作**

- 核心计时应冻结在 `01:42`。
- 让 `demo cv.mp4` 完整播放；不要在这段上继续读核心口播。
- 视频自带声音时保留原声；若没有声音，则安静播放。结束后自动或手动进入 Scene 7。

**English cue — not counted**

> [Pause the core narration and play the prerecorded M2/M3 demo.]

**中文说明**

这是独立演示插播，不属于两分钟 pitch。不要为了控制核心时长而裁掉视频，也不要把视频时长加入口播时间码。

## Scene 7 — Distillation

**Core time:** `01:42–01:55`

**画面动作**

- 视频结束后恢复核心计时。
- 先看 `99.4M → 7.96M`，再看右侧 Student 和教师模型的 AUROC 差距。

**English narration**

> Distillation moves the detector toward the edge. Student V3.2 corrected uses 7.96 million parameters—about 92 percent fewer than the teachers—while retaining 0.9063 clean AUROC and 0.8711 robust mean AUROC. It is an efficiency trade-off, not teacher equivalence.

**中文说明**

蒸馏把参数量压缩到 7.96M，同时保留较强的 strict-unseen 排序能力；但数值仍低于教师模型，因此必须明确这是端侧规模和检测效果之间的折中。V3.2 corrected 的实验结果不等于它已经在 Android App 中完成可复现部署。

**发音提示**

- `distillation`：**dis-ti-LAY-shən**
- `7.96M`：**seven point nine six million**
- `teacher equivalence`：**TEE-chər i-KWIV-ə-ləns**

## Scene 8 — Android Prototype

**Timing:** `Video interlude — outside the 02:00 core`

**画面动作**

- 当前版本只显示 `Android demo video — reserved`，不需要为占位页增加核心口播。
- 正式 Android MP4 提供后，在这里冻结核心计时并完整播放；结束后进入 Closing。
- 如果本轮录制仍没有 Android 视频，可直接手动跳到 Scene 9。

**English cue — not counted**

> [Reserved for the prerecorded Android prototype demo.]

**中文说明**

该页只保留三项证据边界：纯本地推理、不上传待检测图片、原型仍在开发。不声称 production-ready、real-time、具体延迟、内存、功耗或 V3.2 手机端 parity。

## Scene 9 — Closing

**Core time:** `01:55–02:00`

**画面动作**

- 从 Android 插播页进入后恢复核心计时。
- 放慢最后一句，在 “edge” 后短暂停顿，再说 “Thank you”。

**English narration**

> Together, RepostGuard-Lite is robust to reposting and small enough for the edge. Thank you.

**中文说明**

用一句话收束两条主线：对传播退化保持鲁棒，同时通过蒸馏具备继续向端侧推进的模型规模。不要把 “small enough for the edge” 延伸成已经完成生产级移动部署。

<!-- CORE-NARRATION-END -->

---

## 连续英文核心口播版

> Hello everyone. We present RepostGuard-Lite for TikTok TechJam 2026: AI-generated image detection designed for the transformations of real social sharing.
>
> Detection is hardest after generation. Editing, resizing, JPEG compression, blur, noise, and repeated reposting can erase forensic traces. AIDE showed why hybrid evidence matters by combining semantic context with patch-level signals.
>
> Building on that idea, our two-stream detector pairs a frozen OpenCLIP semantic branch with DCT, SRM, and NPR-inspired forensic features. M2 fuses both streams directly. Clean and degraded views share one model and consistency losses. M3 adds a six-dimensional quality gate. The full model has 99.4 million parameters—below 0.1 billion.
>
> We moved beyond the CIFAKE and SID-Set pilots to Community Forensics train-v3: 24 thousand training images from 921 exact generators, plus a four-thousand-image strict-unseen test covering 12 generators absent from training.
>
> The pilots warned us that clean results can mislead: CIFAKE B0 scored 0.9945 clean AUROC but only 0.5442 in its worst condition. On strict-unseen data, M2 reaches 0.9308 clean and 0.8525 after six-stage degradation, making M2 our current deployment choice. M3 remains a data-dependent research candidate.
>
> **[Video interlude A — pause the core narration and play `demo cv.mp4`; core timer stays at 01:42.]**
>
> Distillation moves the detector toward the edge. Student V3.2 corrected uses 7.96 million parameters—about 92 percent fewer than the teachers—while retaining 0.9063 clean AUROC and 0.8711 robust mean AUROC. It is an efficiency trade-off, not teacher equivalence.
>
> **[Video interlude B — reserved for the Android MP4; do not advance the core timer.]**
>
> Together, RepostGuard-Lite is robust to reposting and small enough for the edge. Thank you.

## 建议读法速查

| 术语 | 建议读法 |
|---|---|
| AIGI | `A-I-G-I` |
| AIDE | `aid` |
| AUROC | `A-U-rock` |
| DCT | `D-C-T` |
| SRM | `S-R-M` |
| NPR | `N-P-R` |
| M2 / M3 | `M-two / M-three` |
| OpenCLIP | `open clip` |
| CIFAKE | `sigh-fake` |
| strict-unseen | `strict unseen` |

## 两分钟录制检查单

- [ ] 将 HTML 和 `demo cv.mp4` 保持在同一个 `showcase/` 文件夹中，不重命名视频。
- [ ] 以 1920×1080 或 16:9 窗口打开 HTML，按 `F` 进入全屏。
- [ ] 按 `R` 回到 Scene 1 并清零核心计时；按 `P` 开始核心演练。
- [ ] Scene 6 出现后，确认核心时间冻结在 `01:42`，并完整播放 `demo cv.mp4`。
- [ ] Scene 7 开始时确认核心计时从 `01:42` 恢复，而不是把视频时长累计进去。
- [ ] Scene 8 暂无 Android MP4 时手动跳过；不要把占位停留时间记入核心时长。
- [ ] Scene 9 的进度条最终到达 `02:00`，最后一句后保留约半秒停顿。
- [ ] 不把 Student V3.2 corrected 的实验数字说成已在 Android App 中完成部署。
- [ ] 不使用 `real-time`、`production-ready` 或未经实测的延迟、功耗、内存表述。
- [ ] 正式录屏前关闭通知，并测试 MP4 原声是否与麦克风口播音量匹配。

## 如果实际核心口播超过两分钟

优先删除下面两句；不要加速到影响清晰度：

1. Scene 2：`AIDE showed why hybrid evidence matters by combining semantic context with patch-level signals.`
2. Scene 5：`M3 remains a data-dependent research candidate.`

删除后仍需保留屏幕上的 AIDE 来源、M2 部署优先和 M3 候选定位，避免改变事实边界。
