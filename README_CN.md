# Comfy-Easy-SenseNova-U1

SenseNova-U1 的 ComfyUI 本地推理节点。节点位于右键菜单：

`eastmoe → Comfy-Easy-SenseNova-U1`

本项目直接使用 `origin/SenseNova-U1/src` 中随仓库保存的原项目推理实现，模型权重统一存放在：

`ComfyUI/models/SenseNova/<模型子文件夹>`

## 节点

- **SenseNova-U1 模型下载**：支持 Hugging Face、hf-mirror、指定 revision、访问令牌、关闭 TLS 校验、关闭 Xet 和强制重新下载。每个仓库保存到 `models/SenseNova` 下独立的子文件夹。
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

依赖文件只会额外安装 ComfyUI 官方依赖未包含的 `accelerate`。`transformers`、`numpy`、`Pillow` 以及它们提供的传递依赖均复用 ComfyUI 环境，原项目版本约束以注释形式保留在依赖文件中。

## 兼容性

- Transformers：支持 `4.57.1` 和 `5.x`。兼容层会在 5.x 下恢复 Qwen3 配置中移除的 `rope_theta` 属性，补齐新版权重初始化要求的 RoPE 方法，转换因果掩码接口，并按主版本选择 `torch_dtype`（4.x）或 `dtype`（5.x）加载参数。
- PyTorch：已检查 `2.6`～`2.12` 使用到的张量、SDPA、自动混合精度和权重加载接口，并实测 `2.6`、`2.12` 两个边界版本。`2.6` 起 `torch.load` 默认使用 `weights_only=True`，原项目涉及的普通张量 state dict 可直接兼容；插件要求 `PyTorch>=2.6`。更高版本会显示“超出已检查范围”警告，但不会被强制阻止。
- NumPy：支持 `1.24+` 的 1.x 与 2.x。节点只使用两代均保留的数组、类型和张量桥接接口，不依赖 NumPy 2.0 移除的旧类型别名。

实际运行版本及兼容状态会出现在“模型加载”节点的“模型信息”输出中。`transformers<4.57.1`、`transformers>=6`、`PyTorch<2.6` 或 `NumPy<1.24/NumPy>=3` 会在加载前给出明确错误。

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
- 关闭 TLS 证书校验会降低连接安全性，仅建议在可信网络中临时排障。
- 图文交错文本中的 `<image>` 标签与“生成图像”批次按出现顺序对应；若本次没有生成图片，图像接口返回一个 1×1 黑色占位图，以保持 ComfyUI 接口稳定。

## 原项目

原项目源码、文档和许可证保留在 [`origin/SenseNova-U1`](origin/SenseNova-U1)。模型行为与参数含义以原项目文档为准。
