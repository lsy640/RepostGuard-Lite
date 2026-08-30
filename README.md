# RepostGuard-Lite 本地图片推理

RepostGuard-Lite 用于判断图片由 AIGC 生成的可能性。当前本地推理程序接收一个图片目录，递归读取其中的图片，并为每张可读取图片输出一个 0 到 1 之间的置信度：

- image_path：相对于输入目录的图片路径；
- pred：图片为 AIGC 生成内容的概率，越接近 1 表示模型越倾向于判定为 AIGC。

输出是标准 JSON 文件：

~~~json
[
  {
    "image_path": "example.jpg",
    "pred": 0.9342
  },
  {
    "image_path": "subdir/image.png",
    "pred": 0.1276
  }
]
~~~

## 支持的模型

推理入口支持项目中的 B0、B1、B2、M2 和 M3 模型：

| 模型 | 主要结构 |
|---|---|
| B0 | EfficientNet-B0，Clean-only 训练 |
| B1 | EfficientNet-B0，加入鲁棒性数据增强 |
| B2 | 冻结 OpenCLIP 图像编码器和线性分类头 |
| M2 | OpenCLIP 语义分支与频域/残差取证分支 |
| M3 | M2 双分支与质量感知动态门控 |

推荐使用当前 train-v3 M3 模型进行本地推理。运行时必须同时提供：

~~~text
<MODEL_DIR>/
├── best.pt
└── resolved_config.yaml
~~~

resolved_config.yaml 必须是训练该 checkpoint 时保存的原始解析配置。程序会校验配置摘要；如果配置被修改或与 checkpoint 不匹配，推理会拒绝运行。

模型权重通常不会提交到代码仓库。请通过安全方式将 checkpoint 和对应配置复制到本机，不要把访问令牌或其他密钥写入配置文件。

## 系统要求

- Python 3.10 或更高版本，推荐 Python 3.11；
- 64 位 Windows、macOS 或 Linux；
- CPU 推理适用于所有三种系统；
- Windows/Linux 可使用 NVIDIA CUDA；
- macOS 当前命令行接口使用 CPU，尚未开放 MPS 设备选项；
- M2/M3 首次构建 OpenCLIP 主干时可能需要网络下载相应的预训练权重缓存。

下列命令均应在项目根目录执行。

## Windows 安装

以下命令适用于 PowerShell，并且不要求激活虚拟环境：

~~~powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e .
~~~

如需使用支持 CUDA 12.1 的 NVIDIA GPU，将 PyTorch 安装命令替换为：

~~~powershell
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
~~~

## macOS 安装

Intel Mac 和 Apple Silicon Mac 均可使用以下方式安装：

~~~bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
~~~

当前推理参数只支持 cpu 或 cuda，因此 macOS 请使用 --device cpu。

## Linux 安装

CPU 版本：

~~~bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
~~~

支持 CUDA 12.1 的 NVIDIA GPU 版本：

~~~bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e .
~~~

## 执行目录推理

程序会递归扫描输入目录，并按相对路径排序输出结果。

### Windows CPU

~~~powershell
.\.venv\Scripts\python.exe -m repostguard.infer --config "C:\path\to\model\resolved_config.yaml" --checkpoint "C:\path\to\model\best.pt" --input-dir "C:\path\to\images" --output "C:\path\to\results\predictions.json" --diagnostics "C:\path\to\results\diagnostics.json" --batch-size 16 --device cpu
~~~

### Windows NVIDIA GPU

~~~powershell
.\.venv\Scripts\python.exe -m repostguard.infer --config "C:\path\to\model\resolved_config.yaml" --checkpoint "C:\path\to\model\best.pt" --input-dir "C:\path\to\images" --output "C:\path\to\results\predictions.json" --diagnostics "C:\path\to\results\diagnostics.json" --batch-size 32 --device cuda
~~~

### macOS CPU

~~~bash
python -m repostguard.infer \
  --config "/path/to/model/resolved_config.yaml" \
  --checkpoint "/path/to/model/best.pt" \
  --input-dir "/path/to/images" \
  --output "/path/to/results/predictions.json" \
  --diagnostics "/path/to/results/diagnostics.json" \
  --batch-size 16 \
  --device cpu
~~~

### Linux CPU

~~~bash
python -m repostguard.infer \
  --config "/path/to/model/resolved_config.yaml" \
  --checkpoint "/path/to/model/best.pt" \
  --input-dir "/path/to/images" \
  --output "/path/to/results/predictions.json" \
  --diagnostics "/path/to/results/diagnostics.json" \
  --batch-size 16 \
  --device cpu
~~~

### Linux NVIDIA GPU

~~~bash
python -m repostguard.infer \
  --config "/path/to/model/resolved_config.yaml" \
  --checkpoint "/path/to/model/best.pt" \
  --input-dir "/path/to/images" \
  --output "/path/to/results/predictions.json" \
  --diagnostics "/path/to/results/diagnostics.json" \
  --batch-size 32 \
  --device cuda
~~~

如果显存或内存不足，请先将 --batch-size 降到 8、4 或 1。

## 命令行参数

| 参数 | 是否必需 | 说明 |
|---|---|---|
| --config | 是 | 与 checkpoint 匹配的 resolved_config.yaml |
| --checkpoint | 是 | 模型 checkpoint 文件 |
| --input-dir | 是 | 待检测图片目录，会递归扫描子目录 |
| --output | 是 | 预测 JSON 输出路径 |
| --diagnostics | 否 | 诊断 JSON 路径，默认 diagnostics.json |
| --batch-size | 否 | 推理 batch size，默认 32 |
| --device | 否 | cpu 或 cuda，默认 cuda |

安装完成后也可以使用等价的控制台入口：

~~~bash
repostguard-infer --config <CONFIG> --checkpoint <CHECKPOINT> --input-dir <IMAGE_DIR> --output predictions.json --diagnostics diagnostics.json --device cpu
~~~

## 输入图片

当前支持以下扩展名：

~~~text
.jpg .jpeg .png .webp .bmp .gif .tif .tiff
~~~

- 输入目录会递归扫描；
- GIF 只读取第一帧；
- 所有图片都会转换为 RGB；
- 如果训练配置启用了格式去偏，推理时会自动采用对应的统一尺寸、JPEG 质量和色度采样设置；
- 损坏或无法解码的图片不会终止整批推理，而会记录在 diagnostics.json 中；
- 如果目录中不存在任何受支持图片，程序会返回错误。

## 输出说明

predictions.json 是 JSON 数组，每条成功推理的图片只包含：

~~~json
{
  "image_path": "relative/path/image.jpg",
  "pred": 0.812345
}
~~~

pred 是模型输出 logit 经过 sigmoid 后的概率，不是已经应用阈值的二分类标签。本程序不会利用待推理图片重新选择阈值。

diagnostics.json 包含：

- checkpoint SHA256；
- 配置摘要；
- 推理设备；
- 模型参数量；
- 成功处理的图片数；
- 无法读取图片的路径和错误；
- 推理阶段实际使用的格式去偏设置。

两个 JSON 文件均采用临时文件写入后原子替换，避免留下半写入结果。

## 常见问题

### Checkpoint and inference config digests differ

checkpoint 与配置不匹配。请使用训练时保存在 checkpoint 同目录下的 resolved_config.yaml，不要改用另一个模型或实验版本的配置。

### CUDA requested but unavailable

当前环境没有可用 CUDA，或安装了 CPU 版本的 PyTorch。可以改用 --device cpu，或者在 Windows/Linux 上安装与本机驱动兼容的 CUDA PyTorch wheel。

### No supported images under

输入路径为空、路径填写错误，或者目录内没有受支持扩展名的图片。

### 推理速度慢或内存不足

M2/M3 的双分支结构明显重于 B0/B1。降低 --batch-size；没有 NVIDIA GPU 时使用 CPU 会更慢，但输出结构和概率语义保持不变。

## 代码位置

- 推理入口：src/repostguard/infer.py
- 模型构建：src/repostguard/models/detectors.py
- 图片预处理：src/repostguard/data/transforms.py
- checkpoint 完整性检查：src/repostguard/checkpoint.py
- 输出结构测试：tests/test_inference_schema.py

## 使用限制

- pred 是模型置信度，不代表经过现实部署校准后的绝对概率；
- 不同图片来源、生成器和后处理方式可能改变模型表现；
- 在真实业务中应使用独立校准集冻结阈值，不能使用待检测数据选择阈值；
- 批量测试结果不应在缺少人工复核和数据来源分析时作为唯一判定依据。
