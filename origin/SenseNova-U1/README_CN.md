# SenseNova-U1

SenseNova-U1 是基于 NEO-unify 架构的原生统一多模态模型。本仓库仅保留推理所需源码。

[English](./README.md)

精简后的仓库包含模型实现、推理工具、纯文本示例脚本和部署文档。训练、评估、UI 集成及随仓库附带的示例多媒体均已移除。

## 模型

| 模型 | Hugging Face |
| --- | --- |
| SenseNova-U1-8B-MoT | [sensenova/SenseNova-U1-8B-MoT](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT) |
| SenseNova-U1-A3B-MoT | [sensenova/SenseNova-U1-A3B-MoT](https://huggingface.co/sensenova/SenseNova-U1-A3B-MoT) |

## 安装

先阅读[安装说明](./docs/installation_CN.md)，再使用 `uv` 安装：

```bash
uv sync
```

也可以使用 pip 安装核心推理依赖和本地包：

```bash
pip install -r requirements.txt
```

## 推理

根目录的 `inference.py` 是统一推理入口，支持视觉理解、文生图、图像编辑、交错生成和混合 JSONL 批处理。仓库不再附带输入图片；需要图像输入时，请传入本地媒体路径。完整参数可通过 `python inference.py --help` 查看。

视觉理解：

```bash
python inference.py --task vqa \
  --model_path sensenova/SenseNova-U1-8B-MoT \
  --image /path/to/input.jpg \
  --question "描述这张图片。" \
  --output outputs/answer.txt
```

文生图：

```bash
python inference.py --task t2i \
  --model_path sensenova/SenseNova-U1-8B-MoT \
  --prompt "一只雄孔雀正在吸引雌孔雀" \
  --output output.png
```

图像编辑：

```bash
python inference.py --task edit \
  --model_path sensenova/SenseNova-U1-8B-MoT \
  --image /path/to/input.webp \
  --prompt "把动物的毛色改深一些。" \
  --output output_edited.png
```

交错生成：

```bash
python inference.py --task interleave \
  --model_path sensenova/SenseNova-U1-8B-MoT \
  --prompt "制作一份适合新手的图文烹饪教程。" \
  --resolution "16:9" \
  --output_dir outputs/interleave/
```

完整推理参数和 JSONL 输入格式参见[示例说明](./examples/README_CN.md)。低显存推理与部署说明参见 [GPU 显存分析](./docs/gpu_mem_profiler_CN.md)和[部署文档](./docs/deployment_CN.md)。

## 许可证

本项目采用 [Apache 2.0 License](./LICENSE)。
