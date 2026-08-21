# ComfyUI-Easy-SenseNova-U1

SenseNova-U1 的 ComfyUI 本地推理节点。节点在右键菜单 `eastmoe → Comfy-Easy-SenseNova-U1` 下，类原生节点在 `native` 子菜单。

## 作用

给 ComfyUI 加 SenseNova-U1 的模型支持，全程不动 ComfyUI 的 Python 环境：

- Transformers 4.57.1 源码快照随插件放在 `transformer_patch/`，全局 transformers 保持 ComfyUI 需要的版本
- 推理实现直接取自原项目，原样保留在 `origin/SenseNova-U1`
- 模型权重统一放 `ComfyUI/models/SenseNova/`，checkpoint 放 `ComfyUI/models/checkpoints/`

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/eastmoe/ComfyUI-Easy-SenseNova-U1
```

重启 ComfyUI 即可。

单卡使用连 `pip install -r requirements.txt` 都可以跳过，日常依赖 ComfyUI 官方环境已经齐了。requirements.txt 里只有 `accelerate` 一项是 ComfyUI 官方没有的，多卡 `device_map` 加载才用得上；`flash-attn` 和 `torchao` 属于可选优化，用到对应功能时再装。

## 节点

两组节点功能对应，一组拆成零件、一组整套封装。推荐用类原生节点自己组合工作流。

### 类原生节点

| 节点 | 作用 |
|------|------|
| SenseNova Loader | 从 checkpoint 输出标准 `MODEL` 和 HiDream-O1 同类的像素空间 `VAE`。显存模式分 `full` / `balanced` / `low`，`balanced` / `low` 走原生逐层卸载，24GB 显卡推荐 `balanced` |
| SenseNova Conditioning | 输出正面、仅图像、无条件三路 `CONDITIONING`。图片可选输入，Think Mode 在首次前向时建立原生 DynamicCache，宽高和批量从实际 latent 推导 |
| SenseNova Sampling Patch | 设置原生 flow timestep shift、动态分辨率 noise scale、CFG 区间与 patch-space CFG 归一化 |
| SenseNova Scheduler | 输出与原项目完全相同的时间步，推荐接 Euler |
| SenseNova Dual Guider | 图像编辑用，复现原项目的编辑 CFG 结构，接 `SamplerCustomAdvanced` |
| SenseNova Think Text | 采样完成后读取 Think Mode 的思考文本 |

空 latent 用 ComfyUI 自带的空 HiDream-O1 潜空间图像，解码用 Loader 输出的像素空间 VAE。

### 集成节点

| 节点 | 作用 |
|------|------|
| SenseNova-U1 模型下载 | 拉取模型仓库，支持 hf-mirror、并行下载、断点续传、大小/SHA256 校验 |
| SenseNova-U1 模型加载 | 加载 checkpoint，支持存储精度、边加载边量化（mxfp8 / mxfp4 / nvfp4）、显存模式和单/多卡 |
| SenseNova-U1 文生图 | 默认开启 Think Mode，宽高按 32 的倍数自由设置 |
| SenseNova-U1 图像编辑 | 单图或 IMAGE 批次多图参考，自动输出分辨率，默认开启 Think Mode |
| SenseNova-U1 视觉问答 | 图片描述、视觉问答，贪心或采样解码 |
| SenseNova-U1 图文交错生成 | 可选参考图批次、原生思考、多张图文交错输出 |

## 模型获取

两种方式任选：

**直接下载转换好的 checkpoint（省事）：**

```
https://huggingface.co/eastmoe/SenseNova-U1.5-8B-MoT
```

**下载官方模型自己转换：**

```
https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT
```

官方模型是 HF 格式，要先转换再量化，脚本在 `tools/` 下：

```bash
# 1. 转换：HF 模型目录 → 单文件 checkpoint
python tools/convert_hf_to_comfy_checkpoint.py \
  ComfyUI/models/SenseNova/SenseNova-U1.5-8B-MoT \
  ComfyUI/models/checkpoints/SenseNova-U1.5-8B-MoT.safetensors

# 2. 量化：普通 checkpoint → 预量化 checkpoint（可选）
python tools/quantize_checkpoint.py \
  ComfyUI/models/checkpoints/SenseNova-U1.5-8B-MoT.safetensors \
  ComfyUI/models/checkpoints/SenseNova-U1.5-8B-MoT-w4a8.safetensors \
  --method w4a8_convrot
```

转换器流式合并 HF 分片，不会把完整模型读进内存。量化方法支持 `bf16`、`int8_convrot`、`mxfp8`、`w4a8_convrot`、`mxfp4`、`nvfp4`。建议先加 `--dry-run` 检查模型来源和 Linear 数量。

## 跑起来

1. 模型下载节点拉模型（或按上文转换）
2. 模型加载节点读 checkpoint，显存模式按卡选：显存够用 `full`，24GB 左右 `balanced`，紧张用 `low`
3. 加载节点的 `MODEL` / `VAE` 输出接文生图节点，或接类原生节点自由组合

官方模型体积较大，下载和首次加载需要一些时间。SenseNova-U1 推荐约 2K 输出，峰值显存还受分辨率、KV Cache、批量数和交错图像数影响。

## 原项目

原项目源码、文档和许可证保留在 [`origin/SenseNova-U1`](origin/SenseNova-U1)。模型行为与参数含义以原项目文档为准。
