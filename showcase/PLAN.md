# TikTok TechJam 2026 RepostGuard-Lite 两分钟 HTML 演示方案

## Summary

- 在 `/Users/liushiyuan/Downloads/AGI/TikTok_project_5/showcase/` 新建两个交付文件：
  - `techjam_2026_repostguard_showcase.html`：除本地 MP4 插播外资源自包含、可离线运行的英文演示平台。
  - `techjam_2026_repostguard_narration_bilingual.md`：约 230–250 个英文单词的两分钟口播，逐镜头附中文说明。
- HTML 采用 16:9 竞赛演示界面，而不是普通长网页：9 个全屏场景，其中 7 个是合计 120 秒的核心叙事场景，场景 6 和 8 是不计入核心时长的 MP4 视频插播；支持键盘/按钮导航、进度条、全屏模式和核心演练计时。
- 整体风格使用深海军蓝背景、取证青色、AIGC 洋红色和质量门控紫色，延续当前 M2/M3 架构图的语义色彩；画面只显示英文，中文仅进入 Markdown 口播说明。
- 除两段预录 MP4 外，所有 CSS、JavaScript、图表、论文裁图和项目架构图都嵌入 HTML，不加载 CDN、字体、远程媒体或分析脚本。视频使用 `showcase/` 内的本地相对路径，因此整个文件夹可在断网环境直接播放。
- 场景 6 直接插入现有的 `showcase/demo cv.mp4`；场景 8 预留同规格 Android MP4 视频位和替换说明，当前不伪造 App 截图、不等待 V3.2 实测信息，也不修改 README。
- 页面底部同时显示“核心 02:00 计时”和当前视频插播状态；进入场景 6 或 8 时核心计时与核心进度条冻结，视频结束或手动继续后才恢复。

## 两分钟叙事与画面

| 时间 | 场景 | 画面与核心信息 |
|---|---|---|
| `00:00–00:10` | 1. Opening | 标题 `Robust Detection of AI-Generated Images Under Real-World Transformations`，副标题 `RepostGuard-Lite · TikTok TechJam 2026`；用真实图与 AIGI 图的快速切换提出“Can we still detect it after reposting?”。 |
| `00:10–00:30` | 2. The Real-World Gap | 展示 `Generate → Edit → Resize → JPEG → Repost` 传播链，强调压缩、缩放、模糊、噪声和人工编辑会破坏低层取证痕迹；随后引出 AIDE 的 semantic + forensic hybrid 思想。 |
| `00:30–00:55` | 3. Our Architecture | 使用现有 RepostGuard-Lite 架构图的聚焦视图，依次高亮语义分支、DCT/SRM/NPR 取证分支、M2 静态融合、clean/degraded 一致性损失和 M3 六维质量门控；突出 `99.4M parameters · <0.1B · 4.97% of the 2B limit`。 |
| `00:55–01:15` | 4. Dataset Strategy | 从 CIFAKE/SID-Set pilot 过渡到 Community Forensics train-v3；突出 `24K training images`、`921 exact generators`、`4K strict-unseen test images`、`12 entirely unseen generators`，背景使用仓库中的生成器 atlas 局部拼图。 |
| `01:15–01:42` | 5. Results That Survive Reposting | 左侧用 CIFAKE B0 的 `0.9945 clean → 0.5442 worst` 说明封闭域高分可能误导；右侧绘制 train-v3 strict-unseen 的 B0/B1/B2/M2/M3 clean 与 six-stage AUROC 对比，重点标出 M2 `0.9308 → 0.8525`，并注明 M2 是当前部署优先模型、M3 是受数据规模影响的候选融合策略。 |
| `插播 A · 不计入 02:00` | 6. M2/M3 Detection Demo | 直接播放 `showcase/demo cv.mp4`。进入本场景时暂停核心计时与核心进度，视频使用自身完整时长；播放结束后自动进入场景 7，或允许手动继续。该段不再使用截图替代。 |
| `01:42–01:55` | 7. Distillation | 用参数量与 AUROC 双编码图比较 M2、M3 和 Student V3.2 corrected：`99.4M → 7.96M`、参数减少约 `92%`，Clean AUROC `0.9308/0.9305 → 0.9063`，Robust mean AUROC `0.9163/0.9154 → 0.8711`；明确这是性能与端侧规模之间的折中，而非教师模型完全等价。 |
| `插播 B · 不计入 02:00` | 8. Android Prototype | 预留 Android 预录 MP4 播放位，当前显示英文占位文案 `Android demo video — reserved`，并保留 `Local inference`、`No image upload`、`Prototype under development` 三项保守事实。未提供视频前不加载空 `src`、不伪造 App 截图，也不占用核心时间。 |
| `01:55–02:00` | 9. Closing | 收束为 `Robust to reposting. Small enough for the edge.`，补充 `Thank you` 和项目名，核心进度条在 120 秒完成。 |

## Content and Asset Implementation

### 论文与项目素材

- 从 [ICLR 2025 AIDE 正式论文](https://proceedings.iclr.cc/paper_files/paper/2025/file/b0303773962ea1b5394c3a83cc7dd066-Paper-Conference.pdf)第 6 页提取并裁切 Figure 2，只保留原始 AIDE 架构区域；在画面底部标注论文题名、`Figure 2`、会议和链接，不进行可能改变论文含义的重绘。
- Community Forensics 的动机使用 [CVPR 2025 正式论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Park_Community_Forensics_Using_Thousands_of_Generators_to_Train_Fake_Image_CVPR_2025_paper.pdf)作为外部来源；项目具体的 24K/921/4K/12 数字继续以当前主 README 为准。
- 复用现有 `repostguard_m2_m3_architecture.svg`，在 HTML 中嵌入高分辨率版本并通过遮罩/聚焦动画依次突出语义、取证、一致性训练和门控区域，不重新绘制一份可能与实现不一致的模型图。
- 复用 Community Forensics train/test atlas，但只裁取适合 16:9 画面的代表性区域；不得在两分钟画面中显示无法辨认的完整密集大图。
- M2/M3 Demo 不再制作或嵌入前端截图；场景 6 使用 `<video>` 直接加载 `showcase/demo cv.mp4`。文件名包含空格时使用浏览器可正确解析的相对 URL（例如 `demo%20cv.mp4`），同时在 `<video>` 内提供浏览器不支持 MP4 时的英文回退提示。
- 场景 8 使用与场景 6 相同的 16:9 视频容器和控制样式，但在 Android MP4 尚未提供时不设置空的 `src` 或 `<source>`，仅显示明确的预留占位。后续只需填入相对文件名即可替换，不改变前七个场景、核心计时或主口播结构。
- 不使用 ImageGen 或装饰性 AI 图来替代论文图、项目架构或真实 Demo，保证每个关键视觉都能追溯到论文或仓库产物。

### 图表与事实边界

- 实验页使用原生 HTML/CSS/内联 SVG 绘制，不引入图表库。
- 跨数据集结果分成独立面板并标注 `Different protocols — do not rank across panels`，防止把 CIFAKE、SID-Set 和 Community Forensics 的不同测试协议直接横向排名。
- 当前 train-v3 主结论固定为：
  - M2：Clean AUROC `0.9308`，six-stage AUROC `0.8525`，20 个 transformed 条件平均 AUROC `0.9163`。
  - M3：Clean AUROC `0.9305`，six-stage AUROC `0.8489`，20 个 transformed 条件平均 AUROC `0.9154`。
  - M2 是当前部署优先选择；不把 M3 动态门控描述为已经稳定优于 M2。
- Student 页明确分离模型效果与移动端状态：
  - V3.2 corrected 的 `7.96M / 0.9063 / 0.8711` 用于蒸馏效果比较。
  - Android 页只说明已有纯本地推理原型，不声称这组 V3.2 数字已经在手机 App 中完成可复现部署。
- 不宣称 Android App 是生产级产品，也不使用 `real-time`、`ultra-fast`、具体延迟或功耗数据。
- AIDE 页面强调其“高层语义 + 低层取证”的启发关系，不把 RepostGuard-Lite 描述成 AIDE 的直接复现，也不在缺少当前 README 证据时加入 AIDE 参数量对比。

## HTML Interaction Interface

- 默认进入第 1 页，支持：
  - `ArrowRight`、`Space`、`PageDown`：下一页。
  - `ArrowLeft`、`PageUp`：上一页。
  - `Home` / `End`：首尾页。
  - `F`：进入或退出全屏。
  - `P`：开始或暂停两分钟演练模式；在视频插播页控制当前视频。
  - `R`：从第 1 页重置核心计时。
- 页面底部显示细进度条、当前场景编号、核心演练时间和插播状态；正常手动录制时控制栏自动淡出，鼠标移动或键盘操作后短暂出现。
- 核心场景使用 `data-kind="core"` 和非零 `data-duration`；场景 6/8 使用 `data-kind="video"`、`data-duration="0"`。7 个核心场景的时长严格合计 120 秒，两个视频场景不参与求和。
- 自动演练进入场景 6 时冻结核心时间并播放 `demo cv.mp4`；收到视频 `ended` 事件后进入场景 7 并恢复计时。场景 8 在视频缺失时保持冻结并等待手动继续；后续填入 Android MP4 后采用相同的 `ended` 续播逻辑。
- 在视频插播页按 `P` 控制当前视频播放/暂停，但不改变已经累计的核心时间；离开视频页时主动暂停视频，避免隐藏场景继续出声。
- 所有动画只承担信息揭示：传播链逐步出现、架构分支高亮、数据数字计数和结果条形图生长；不使用持续旋转、粒子背景或影响阅读的炫技动效。
- 支持 `prefers-reduced-motion`，关闭非必要转场；隐藏场景中的动画和计时停止，避免后台继续播放。
- 使用固定 16:9 舞台并根据窗口等比缩放，确保 1920×1080 录制时无滚动条、无裁切，较小窗口仍可完整查看。
- HTML 中只允许外部论文/项目链接作为可点击引用；不得存在外部 CSS、JavaScript、字体、图片或分析脚本请求。唯一允许的本地媒体依赖是 `showcase/` 中明确列出的 MP4 插播文件。

## Markdown 口播稿

- 文稿按 9 个场景编排，每节包含：
  - 时间码。
  - 画面动作。
  - 英文口播。
  - 中文含义说明。
  - 必要的发音提示。
- 核心英文连续口播控制在约 230–250 词，目标语速约 120–130 WPM；词数和 120 秒均只统计场景 1–5、7、9，不包含场景 6/8 的视频时长、视频原声或转场提示。
- 英文表达以竞赛 pitch 为主，避免逐表朗读；数值只读最关键的 `<0.1B`、`921 generators`、`0.9308/0.8525` 和 `7.96M`。
- 中文部分用于解释英文句意和录制操作，不作为逐句直译字幕，也不进入英文展示页面。
- 场景 6/8 在 Markdown 中标记为 `Video interlude — outside the 02:00 core`，只写开始/结束操作提示；主讲人在视频播放期间暂停核心口播。若 MP4 自带声音则保留其原声，否则保持安静播放，不把额外解说计入核心稿。
- 文末附：
  - 无时间码的连续英文版。
  - 两分钟录制检查单。
  - `AIGI`、`AIDE`、`AUROC`、`DCT`、`SRM`、`NPR`、`M2/M3` 的建议读法。
  - “若实际录制超过 2 分钟”时可删除的两个低优先级句子，但主稿本身仍按 120 秒设计。

## Verification

- 内容验证：
  - 所有模型参数量、数据规模和实验数值逐项回查当前主 README。
  - AIDE 图必须确认来自正式论文 Figure 2，裁切后标题、箭头和模块名仍清晰。
  - Student V3.2 与 Android 原型不在口播中被错误绑定。
  - Markdown 的场景顺序与 HTML 完全一致；仅核心场景的时间段合计正好 120 秒，场景 6/8 明确标注为计时外插播。
- 离线资源验证：
  - 断网情况下从 `showcase/` 目录直接打开 HTML，内嵌图片、图表和动效完整显示，场景 6 能从相对路径播放 `demo cv.mp4`。
  - 扫描 `src`、`href`、CSS URL 与模块引用，确认除普通引用链接和已声明的本地 MP4 外不存在远程资源依赖。
  - 检查嵌入资源 MIME 类型、Base64 完整性和 HTML 最终文件大小；约 12 MB 的控制目标不包含独立 MP4 文件。
- 交互验证：
  - 验证键盘、按钮、全屏、进度、暂停、重置和 120 秒自动演练。
  - 验证场景 6 的 `demo cv.mp4` 可播放、有控制条、结束后可继续；进入和离开视频前后核心时间保持不变。
  - 验证场景 8 在缺少 Android MP4 时只显示占位、不发起无效媒体请求、不阻塞手动进入 Closing；以后接入视频时同样不计入核心时间。
  - 快速连续翻页、反向翻页和自动播放结束时，不得出现重复计时、越界或隐藏动画继续运行。
- 视觉验证：
  - 逐页检查 1920×1080 和 1366×768 两种视口。
  - 确认无文字溢出、图表裁切、论文图模糊、页脚遮挡或对比度不足。
  - 检查首屏、AIDE、方法、结果、Demo、蒸馏和结尾等关键帧，保证视频暂停在任意一页时仍能独立理解。
- 仓库安全：
  - 只新增 `showcase/` 中的 HTML 和 Markdown；把用户已经提供的 `showcase/demo cv.mp4` 作为只读媒体资产引用，不重编码、不重命名、不修改。不得改动现有 README、Student 文档、Demo 代码、模型代码或当前工作区中的其他未提交修改。
  - 不提交 checkpoint、用户图片、缓存、构建目录或临时 PDF 裁图；论文图和项目素材只以优化后的嵌入数据存在于最终 HTML。

## Assumptions

- 当前目标是本地录制用的单 HTML，不部署网站、不创建线上链接。
- 两段视频插播都不属于核心两分钟：最终成片的实际总时长等于 `120 秒 + demo cv.mp4 时长 + 后续 Android MP4 时长 + 少量转场时间`。
- App 不是本轮主要展示内容；Android MP4 和实测数据的缺失不会阻塞 HTML 与口播稿交付，场景 8 先保留视频位。
- 现有 `VIDEO_NARRATION_90S_BILINGUAL.md` 是推理 Demo 的独立口播稿，保持不变；本轮新建完整项目的两分钟口播。
- 画面默认英文，不加入中英切换或双语同屏。
- 主 README 是项目数字和结论的首要来源；外部论文只用于 AIDE 架构与 Community Forensics 研究动机。
- 若后续提供 Android MP4，只需把文件放入 `showcase/` 并更新场景 8 的本地相对路径；不重新嵌入 Base64，不改变核心 120 秒、前七个场景或主口播结构。
