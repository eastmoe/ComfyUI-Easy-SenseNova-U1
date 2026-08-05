# SenseNova-U1

Inference-only source for SenseNova-U1, a native multimodal model based on the NEO-unify architecture.

[中文说明](./README_CN.md)

This pruned repository contains the model implementation, inference utilities, text-based example scripts, and deployment documentation. Training, evaluation, UI integrations, and bundled example media are intentionally excluded.

## Models

| Model | Hugging Face |
| --- | --- |
| SenseNova-U1-8B-MoT | [sensenova/SenseNova-U1-8B-MoT](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT) |
| SenseNova-U1-A3B-MoT | [sensenova/SenseNova-U1-A3B-MoT](https://huggingface.co/sensenova/SenseNova-U1-A3B-MoT) |

## Installation

See the [installation guide](./docs/installation.md), then install the project with `uv`:

```bash
uv sync
```

Alternatively, install the core inference dependencies and the local package with pip:

```bash
pip install -r requirements.txt
```

## Inference

The root `inference.py` is the unified entry point for visual understanding, text-to-image generation, image editing, interleaved generation, and mixed JSONL batches. Bundled input images have been removed; pass your own local media paths where an input image is required. Run `python inference.py --help` for the complete parameter list.

Visual understanding:

```bash
python inference.py --task vqa \
  --model_path sensenova/SenseNova-U1-8B-MoT \
  --image /path/to/input.jpg \
  --question "Describe this image." \
  --output outputs/answer.txt
```

Text-to-image:

```bash
python inference.py --task t2i \
  --model_path sensenova/SenseNova-U1-8B-MoT \
  --prompt "A male peacock trying to attract a female" \
  --output output.png
```

Image editing:

```bash
python inference.py --task edit \
  --model_path sensenova/SenseNova-U1-8B-MoT \
  --image /path/to/input.webp \
  --prompt "Change the animal's fur color to a darker shade." \
  --output output_edited.png
```

Interleaved generation:

```bash
python inference.py --task interleave \
  --model_path sensenova/SenseNova-U1-8B-MoT \
  --prompt "Create a beginner-friendly illustrated cooking tutorial." \
  --resolution "16:9" \
  --output_dir outputs/interleave/
```

See [examples/README.md](./examples/README.md) for the complete inference flags and JSONL input formats. Memory-efficient inference and deployment are documented in [GPU memory profiling](./docs/gpu_mem_profiler.md) and [deployment](./docs/deployment.md).

## License

Released under the [Apache 2.0 License](./LICENSE).
