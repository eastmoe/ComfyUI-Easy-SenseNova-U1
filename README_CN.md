# Comfy-Easy-SenseNova-U1

SenseNova-U1 的 ComfyUI 本地推理节点。节点位于右键菜单：

`eastmoe → Comfy-Easy-SenseNova-U1`

本项目直接使用 `origin/SenseNova-U1/src` 中随仓库保存的原项目推理实现，模型权重统一存放在：

`ComfyUI/models/SenseNova/<模型子文件夹>`

## 节点

- **SenseNova-U1 模型下载**：支持 Hugging Face、hf-mirror、并行文件下载、Xet 单文件连接数、自动断点续传、大小/SHA256 校验、指定 revision、访问令牌、关闭 TLS 校验和强制重新下载。每个仓库保存到 `models/SenseNova` 下独立的子文件夹；下载与校验进度同步到终端和 ComfyUI。
- **SenseNova-U1 模型加载**：支持权重存储精度、推理计算精度、注意力机制、单设备、多卡 `device_map` 和低显存层卸载。
- **SenseNova-U1 文生图**：普通生成及 Think Mode，使用原项目推荐的约 2K 分辨率档位。
- **SenseNova-U1 图像编辑**：支持单图或 IMAGE 批次多图参考、自动输出分辨率和 Think Mode。
- **SenseNova-U1 视觉问答**：支持图片描述、视觉问答、贪心或采样解码；输入批次时逐张回答。
- **SenseNova-U1 图文交错生成**：支持可选参考图批次、原生思考和多张图文交错输出。

节点名、参数名、接口名、工具提示和源码关键注释均提供简体中文。

## 安装

将本目录放到 `ComfyUI/custom_nodes/`，然后在 ComfyUI 的 Python 环境安装依赖：

```bash
pip install -r requirements.txt
```

重启 ComfyUI。若使用 `flash` 注意力机制，还需单独安装与当前 Python、PyTorch、CUDA 匹配的 `flash-attn` wheel；否则使用 `auto` 或 `sdpa`。

依赖文件只会额外安装 ComfyUI 官方依赖未包含的 `accelerate`。`numpy`、`Pillow`、`tqdm` 以及它们提供的传递依赖均复用 ComfyUI 环境。Transformers `4.57.1` 源码快照随插件放在 `transformer_patch/transformers_4571`，不会安装或替换 ComfyUI 的全局 Transformers。

## 兼容性

- Transformers：模型配置、注册、tokenizer、权重加载、生成停止条件及 SenseNova 后端全部固定走插件私有 `4.57.1`。全局 Transformers 可以是 ComfyUI 所需的任意版本，插件不检查、不导入也不修改它；原先针对 4.57/5.x 分支的兼容层已移除。
- PyTorch：已检查 `2.6`～`2.13` 使用到的张量、SDPA、自动混合精度和权重加载接口，并实测 `2.6`、`2.12`、`2.13`。`2.6` 起 `torch.load` 默认使用 `weights_only=True`，原项目涉及的普通张量 state dict 可直接兼容；插件要求 `PyTorch>=2.6`。更高版本会显示“超出已检查范围”警告，但不会被强制阻止。
- NumPy：支持 `1.24+` 的 1.x 与 2.x。节点只使用两代均保留的数组、类型和张量桥接接口，不依赖 NumPy 2.0 移除的旧类型别名。

实际运行版本及来源会出现在“模型加载”节点的“模型信息”输出中。`PyTorch<2.6` 或 `NumPy<1.24/NumPy>=3` 会在加载前给出明确错误。

## 推荐工作流

1. 运行“模型下载”，选择官方模型与下载源。
2. 将“模型路径”连接到“模型加载”的可选“模型路径”输入。
3. 按显存情况选择加载参数：
   - 常见 NVIDIA GPU：存储精度 `bfloat16`、计算精度 `auto`、注意力 `auto`、设备 `auto`。
   - 显存充足：显存模式 `full`。
   - 单卡低显存：显存模式 `balanced` 或 `low`，多卡映射保持 `none`。
   - 多卡：显存模式 `full`，多卡映射选择 `auto`，可填写 `0=20GiB,1=20GiB,cpu=64GiB`。
4. 把加载节点的模型输出连接到任一推理节点。

## 精度说明

“存储精度”决定权重加载和驻留的数据类型；“计算精度”控制推理时的自动混合精度。`auto` 会跟随存储精度。float32 计算要求权重也以 float32 存储；CPU 不支持本节点的 float16 自动混合精度，请改用 bfloat16 或 float32。

## 注意事项

- 官方模型体积较大，下载和首次加载需要较长时间。
- SenseNova-U1 推荐约 2K 输出，显存不仅由权重决定，生成分辨率、KV Cache、批量数量和交错图像数也会显著影响峰值显存。
- `vram_mode != full` 与多卡 `device_map != none` 互斥。
- `sdpa` 在 PyTorch 2.6+ 直接使用融合 GQA，不再为 32 个查询头复制 8 组 K/V；NVIDIA 环境仍可使用 `flash` 获得外部 FlashAttention 内核的最佳速度。
- 关闭 TLS 证书校验会降低连接安全性，仅建议在可信网络中临时排障。
- 下载线程控制同时下载的文件数量；Xet 连接数控制每个大文件的并发范围请求。过高的组合会增加内存与磁盘压力，建议先使用默认值 `8` 和 `16`。
- 未启用“强制重新下载”时，Hugging Face Hub 会复用完整文件并从未完成文件继续下载。下载结束后的“大小”校验速度较快；“大小和 SHA256”会重新读取大型权重文件，耗时取决于磁盘速度。
- 所有推理节点均显示终端 `tqdm` 和 ComfyUI 前端进度：图像任务按真实扩散采样步计数，视觉问答按生成 token 计数。思考阶段也会持续检查中断状态。
- 下载、等待下载锁、SHA256 校验、思考、逐 token 解码及每个扩散采样步均响应 ComfyUI“停止”按钮；停止后保留 Hugging Face 未完成文件，下次关闭“强制重新下载”运行时可继续续传。
- 图文交错文本中的 `<image>` 标签与“生成图像”批次按出现顺序对应；若本次没有生成图片，图像接口返回一个 1×1 黑色占位图，以保持 ComfyUI 接口稳定。

## 原项目

原项目源码、文档和许可证保留在 [`origin/SenseNova-U1`](origin/SenseNova-U1)。模型行为与参数含义以原项目文档为准。
