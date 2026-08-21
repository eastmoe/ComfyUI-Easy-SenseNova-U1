# Comfy-Easy-SenseNova-U1

SenseNova-U1 的 ComfyUI 本地推理节点。节点位于右键菜单：

`eastmoe → Comfy-Easy-SenseNova-U1`

本项目直接使用 `origin/SenseNova-U1/src` 中随仓库保存的原项目推理实现，模型权重统一存放在：

`ComfyUI/models/SenseNova/<模型子文件夹>`

## 原生 ComfyUI 节点

- **SenseNova Loader**：从转换后的 checkpoint 输出标准 `MODEL` 和 HiDream-O1 同类的像素空间 `VAE`，同时保留插件内私有 `transformers_4571`、原始模型类与本地 tokenizer。`balanced/low` 使用 SenseNova 原生逐层卸载，`full` 要求整模显存；24GB 显卡推荐 `balanced`。
- **SenseNova Conditioning**：输出正面、仅图像、无条件三个 `CONDITIONING`。图片为可选输入，Think Mode 在采样首次前向时建立原生 DynamicCache；目标宽高和批量从实际 latent 推导。
- **SenseNova Sampling Patch**：设置原生 flow timestep shift、动态分辨率 noise scale、CFG 区间与 patch-space CFG 归一化。
- **SenseNova Scheduler**：输出与原项目完全相同的时间步；推荐连接 Euler。标准 KSampler 的 `simple` scheduler 在常用 50 步时也可使用。
- **SenseNova Dual Guider**：图像编辑时复现 `uncond + text_cfg*(positive-image) + image_cfg*(image-uncond)`，连接 `SamplerCustomAdvanced`。
- **SenseNova Think Text**：在采样完成后读取 Think Mode 文本。

空 latent 无需本插件重复实现：直接使用 ComfyUI 自带的 **空 HiDream-O1 潜空间图像**。解码使用 Loader 输出的像素空间 VAE。

原来的六个一体化节点已标为 `Legacy`，节点 ID 保持不变，旧工作流可继续加载：

- **SenseNova-U1 模型下载**：支持 Hugging Face、hf-mirror、并行文件下载、Xet 单文件连接数、自动断点续传、大小/SHA256 校验、指定 revision、访问令牌、关闭 TLS 校验和强制重新下载。每个仓库保存到 `models/SenseNova` 下独立的子文件夹；下载与校验进度同步到终端和 ComfyUI。
- **SenseNova-U1 模型加载**：支持浮点权重存储精度、MXFP8/MXFP4 边加载边动态量化、加载前清理显存、推理计算精度、注意力机制、单设备、多卡 `device_map` 和低显存层卸载。
- **SenseNova-U1 文生图**：默认开启 Think Mode，宽高可按 32 的倍数自由设置。
- **SenseNova-U1 图像编辑**：支持单图或 IMAGE 批次多图参考、自动输出分辨率，并默认开启 Think Mode。
- **SenseNova-U1 视觉问答**：支持图片描述、视觉问答、贪心或采样解码；输入批次时逐张回答。
- **SenseNova-U1 图文交错生成**：支持可选参考图批次、原生思考、多张图文交错输出及自由宽高。

节点名、参数名、接口名、工具提示和源码关键注释均提供简体中文。

## 转换 BF16 checkpoint

```bash
python tools/convert_hf_to_comfy_checkpoint.py \
  ComfyUI/models/SenseNova/<HF模型目录> \
  ComfyUI/models/checkpoints/SenseNova-U1.safetensors
```

转换器按 safetensors 字节区间流式合并 HF 分片，不把完整模型读入内存。输出包括单一权重文件及同名 `_assets` 目录；后者保存 tokenizer/config，并以链接引用权重文件。checkpoint 内加入 `vae.pixel_space_vae` 哨兵。移动或发布模型时必须同时保留 `_assets` 目录，因为 safetensors 不能替代 tokenizer、配置和插件运行代码。

转换器本身只做 BF16 无损重排。需要预量化 checkpoint 时，使用 `tools/quantize_checkpoint.py`，支持 `int8_convrot`、`mxfp8`、`w4a8_convrot` 和 `mxfp4`。脚本可读取 Hugging Face 仓库 ID、本地 HF 快照或已有的 safetensors checkpoint，只选择由私有模型结构确认的 `torch.nn.Linear` 权重；Embedding、Norm、卷积、bias 与其他张量保持原精度。输出仍是单一 checkpoint 加同名 `_assets`，Loader 会识别量化元数据并继续通过插件私有 Transformers 4.57.1 构造模型。

```bash
ComfyUI/venv/bin/python tools/quantize_checkpoint.py \
  sensenova/SenseNova-U1.5-8B-MoT \
  ComfyUI/models/checkpoints/SenseNova-U1.5-w4a8.safetensors \
  --method w4a8_convrot
```

建议先附加 `--dry-run` 检查模型来源、私有 Transformers 版本和 Linear 数量。前三种格式使用 ComfyUI/comfy-kitchen 的量化布局；MXFP4 需要与当前 PyTorch 匹配的 `torchao>=0.16`。量化逐个权重处理，不会把完整 checkpoint 放入内存，但仍需容纳最大单个 Linear 的设备内存和量化 payload 的临时磁盘空间。

原生逐层卸载模式不能应用 Comfy LoRA 权重补丁；需要 LoRA 时选择 `full`。这是因为原始 Transformers 线性层尚未改造成 `comfy.ops`，不能同时由 SenseNova 层卸载器和 Comfy 权重补丁器管理。

## 安装

将本目录放到 `ComfyUI/custom_nodes/`，然后在 ComfyUI 的 Python 环境安装依赖：

```bash
pip install -r requirements.txt
```

重启 ComfyUI。若使用 `flash` 注意力机制，还需单独安装与当前 Python、PyTorch、CUDA 匹配的 `flash-attn` wheel；否则使用 `auto` 或 `sdpa`。Transformers `4.57.1` 源码快照随插件放在 `transformer_patch/transformers_4571`。

MXFP8/MXFP4 动态加载还需要安装与当前 PyTorch 版本匹配的 `torchao>=0.16`。读取线性层权重时会直接压缩，避免先建立完整 BF16 模型；存储精度与计算精度彼此独立，前向时权重按需解量化到所选 BF16、FP16 或 FP32，`auto` 使用 BF16。预量化 checkpoint 的 `int8_convrot`、`mxfp8`、`w4a8_convrot` 使用 ComfyUI 原生量化算子；`balanced/low` 会以普通 qdata/scale 张量参与原生逐层卸载。预量化 checkpoint 暂不支持多卡 `device_map`。

## 兼容性

- Transformers：模型配置、注册、tokenizer、权重加载、生成停止条件及 SenseNova 后端全部固定走插件私有 `4.57.1`。全局 Transformers 可以是 ComfyUI 所需的任意版本。
- NumPy：支持 `1.24+` 的 1.x 与 2.x。

实际运行版本及来源会出现在“模型加载”节点的“模型信息”输出中。`PyTorch<2.6` 或 `NumPy<1.24/NumPy>=3` 会在加载前给出明确错误。

## 推荐工作流

1. 运行“模型下载”，选择官方模型与下载源。
2. 添加模型加载节点，按显存情况选择加载参数：
   - 常见 NVIDIA GPU：存储精度 `bfloat16`、计算精度 `auto`、注意力 `auto`、设备 `auto`。
   - 显存充足：显存模式 `full`。
   - 单卡低显存：显存模式 `balanced` 或 `low`，多卡映射保持 `none`。
   - 多卡：显存模式 `full`，多卡映射选择 `auto`，可填写 `0=20GiB,1=20GiB,cpu=64GiB`。
   - 需要压缩权重驻留空间时可选 `mxfp8` 或 `mxfp4`，并独立选择计算精度；首次切换模型时可开启“加载前清理显存”。
3. 把加载节点的模型输出连接到任一推理节点。

## 注意事项

- 官方模型体积较大，下载和首次加载需要较长时间。
- SenseNova-U1 推荐约 2K 输出，显存不仅由权重决定，生成分辨率、KV Cache、批量数量和交错图像数也会显著影响峰值显存。
- `vram_mode != full` 与多卡 `device_map != none` 互斥。
- `sdpa` 在 PyTorch 2.6+ 直接使用融合 GQA，NVIDIA 环境仍可使用 `flash` 获得外部 FlashAttention 内核的最佳速度。
- 未启用“强制重新下载”时，Hugging Face Hub 会复用完整文件并从未完成文件继续下载。下载结束后的“大小”校验速度较快；“大小和 SHA256”会重新读取大型权重文件，耗时取决于磁盘速度。
- 下载、等待下载锁、SHA256 校验、思考、逐 token 解码及每个扩散采样步均响应 ComfyUI“停止”按钮；停止后保留 Hugging Face 未完成文件，下次关闭“强制重新下载”运行时可继续续传。
- 图文交错文本中的 `<image>` 标签与“生成图像”批次按出现顺序对应；若本次没有生成图片，图像接口返回一个 1×1 黑色占位图，以保持 ComfyUI 接口稳定。

## 原项目

原项目源码、文档和许可证保留在 [`origin/SenseNova-U1`](origin/SenseNova-U1)。模型行为与参数含义以原项目文档为准。
