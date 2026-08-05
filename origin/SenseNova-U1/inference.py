from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

import sensenova_u1
from sensenova_u1.models.neo_unify.utils import load_image_native, smart_resize
from sensenova_u1.utils import (
    DEFAULT_IMAGE_PATCH_SIZE,
    DEFAULT_VRAM_MODE,
    InferenceProfiler,
    add_offload_args,
    best_available_device,
    infer_input_device,
    load_and_merge_lora_weight_from_safetensors,
    load_model_and_tokenizer,
    make_offload_ctx,
    save_compare,
    seed_all_accelerators,
    vram_mode_to_prefetch_count,
)

DEFAULT_SEED = 42
DEFAULT_TARGET_PIXELS = 2048 * 2048
MIN_INPUT_MAX_PIXELS = 512 * 512
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)

T2I_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "1:1": (2048, 2048),
    "16:9": (2720, 1536),
    "9:16": (1536, 2720),
    "3:2": (2496, 1664),
    "2:3": (1664, 2496),
    "4:3": (2368, 1760),
    "3:4": (1760, 2368),
    "1:2": (1440, 2880),
    "2:1": (2880, 1440),
    "1:3": (1152, 3456),
    "3:1": (3456, 1152),
}

INTERLEAVE_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "1:1": (1536, 1536),
    "16:9": (2048, 1152),
    "9:16": (1152, 2048),
    "3:2": (1888, 1248),
    "2:3": (1248, 1888),
    "4:3": (1760, 1312),
    "3:4": (1312, 1760),
    "1:2": (1088, 2144),
    "2:1": (2144, 1088),
    "1:3": (864, 2592),
    "3:1": (2592, 864),
}

DEFAULT_SYSTEM_MESSAGE = """You are a multimodal assistant capable of reasoning with both text and images.
You support two modes:

Think Mode: When reasoning is needed, you MUST start with a <think></think> block and place all reasoning inside it.
You MUST interleave text with generated images using tags like <image1>, <image2>. Images can ONLY be generated
between <think> and </think>, and may be referenced in the final answer.

Non-Think Mode: When no reasoning is needed, directly provide the answer without reasoning. Do not use tags like
<image1>, <image2>; present any images naturally alongside the text.

After the think block, always provide a concise, user-facing final answer. The answer may include text, images,
or both. Match the user's language in both reasoning and the final answer."""


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    seed_all_accelerators(seed)


def _denorm(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(NORM_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(NORM_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def _tensor_batch_to_pil(batch: torch.Tensor) -> list[Image.Image]:
    array = _denorm(batch.float()).permute(0, 2, 3, 1).cpu().numpy()
    array = (array * 255.0).round().astype(np.uint8)
    return [Image.fromarray(item) for item in array]


def _tensor_image_to_pil(image: torch.Tensor) -> Image.Image:
    return _tensor_batch_to_pil(image)[0]


def _flatten_cli_images(groups: Sequence[Sequence[str]] | None) -> list[str]:
    return [path for group in groups or [] for path in group]


def _coerce_paths(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [str(value)]
    if isinstance(value, list):
        return [str(path) for path in value]
    raise ValueError(f"image must be a path or list of paths, got {type(value).__name__}")


def _resolve_paths(paths: Sequence[str], image_root: str | Path = "") -> list[Path]:
    root = Path(image_root) if image_root else None
    resolved: list[Path] = []
    for value in paths:
        path = Path(value)
        if root is not None and not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise FileNotFoundError(f"input image not found: {path}")
        resolved.append(path)
    return resolved


def _load_rgb(path: str | Path) -> Image.Image:
    image = Image.open(path)
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.getchannel("A"))
        return background
    return image.convert("RGB")


def _parse_resolution(value: str, task: str) -> tuple[int, int]:
    normalized = value.lower().replace("×", "x")
    if "x" in normalized:
        width, height = normalized.split("x", 1)
        return int(width), int(height)
    buckets = INTERLEAVE_RESOLUTIONS if task == "interleave" else T2I_RESOLUTIONS
    if normalized not in buckets:
        choices = ", ".join(buckets)
        raise ValueError(f"unsupported resolution {value!r}; use WIDTHxHEIGHT or one of: {choices}")
    return buckets[normalized]


def _explicit_size(options: dict[str, Any], task: str) -> tuple[int, int] | None:
    width = options.get("width")
    height = options.get("height")
    if (width is None) != (height is None):
        raise ValueError("width and height must be provided together")
    if width is not None:
        return int(width), int(height)
    resolution = options.get("resolution")
    return _parse_resolution(str(resolution), task) if resolution else None


def _validate_image_size(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("image width and height must be positive")
    if width % DEFAULT_IMAGE_PATCH_SIZE or height % DEFAULT_IMAGE_PATCH_SIZE:
        raise ValueError(f"image width and height must be divisible by {DEFAULT_IMAGE_PATCH_SIZE}")


def _editing_size(images: Sequence[Image.Image], options: dict[str, Any]) -> tuple[int, int]:
    explicit = _explicit_size(options, "edit")
    if explicit is not None:
        _validate_image_size(*explicit)
        return explicit
    width, height = images[0].size
    target_pixels = int(options["target_pixels"])
    resized_height, resized_width = smart_resize(
        height=height,
        width=width,
        factor=DEFAULT_IMAGE_PATCH_SIZE,
        min_pixels=target_pixels,
        max_pixels=target_pixels,
    )
    return resized_width, resized_height


def _interleave_size(images: Sequence[Image.Image], options: dict[str, Any]) -> tuple[int, int]:
    explicit = _explicit_size(options, "interleave")
    if images and explicit is None:
        width, height = images[0].size
        resized_height, resized_width = smart_resize(height=height, width=width, factor=32)
        return resized_width, resized_height
    size = explicit or INTERLEAVE_RESOLUTIONS["16:9"]
    _validate_image_size(*size)
    return size


def _resize_edit_inputs(
    images: Sequence[Image.Image],
    max_pixels: str | int | None,
    enabled: bool,
) -> list[Image.Image]:
    if not enabled or max_pixels is None:
        return list(images)
    if str(max_pixels).lower() == "auto":
        budget = max(MIN_INPUT_MAX_PIXELS, 2 * DEFAULT_TARGET_PIXELS // max(1, len(images)))
    else:
        budget = int(max_pixels)
    if budget < MIN_INPUT_MAX_PIXELS:
        raise ValueError(f"input_max_pixels must be at least {MIN_INPUT_MAX_PIXELS}")
    resized: list[Image.Image] = []
    for image in images:
        height, width = smart_resize(
            height=image.height,
            width=image.width,
            factor=DEFAULT_IMAGE_PATCH_SIZE,
            min_pixels=budget,
            max_pixels=budget,
        )
        resized.append(image if image.size == (width, height) else image.resize((width, height), Image.LANCZOS))
    return resized


def _save_images(images: Sequence[Image.Image], output: Path) -> list[str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for index, image in enumerate(images):
        path = output if len(images) == 1 else output.with_name(f"{output.stem}_{index}{output.suffix}")
        image.save(path)
        paths.append(str(path))
        print(f"[saved] {path}")
    return paths


def _sample_options(args: argparse.Namespace, sample: dict[str, Any] | None = None) -> dict[str, Any]:
    values = vars(args).copy()
    if sample:
        for key in (
            "task",
            "prompt",
            "question",
            "width",
            "height",
            "resolution",
            "target_pixels",
            "input_max_pixels",
            "do_resize",
            "seed",
            "cfg_scale",
            "img_cfg_scale",
            "cfg_norm",
            "timestep_shift",
            "enable_timestep_shift",
            "cfg_interval",
            "num_steps",
            "batch_size",
            "think",
            "t_eps",
            "max_new_tokens",
            "max_images",
            "do_sample",
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "system_message",
            "image_root",
            "compare",
            "vqa_min_pixels",
            "vqa_max_pixels",
            "vqa_upscale",
        ):
            if key in sample:
                values[key] = sample[key]
        if "prompt" not in sample:
            for turn in sample.get("conversations", []):
                if turn.get("from") in {"human", "user"}:
                    values["prompt"] = turn.get("value", turn.get("content"))
                    break
    return values


def _resolve_task(options: dict[str, Any], image_paths: Sequence[str]) -> str:
    task = options.get("task", "auto")
    if task != "auto":
        return task
    if options.get("question"):
        return "vqa"
    if image_paths:
        return "edit"
    if options.get("prompt"):
        return "t2i"
    raise ValueError("cannot infer task: provide --prompt, --question, --image, or an explicit --task")


def _preflight_sample(options: dict[str, Any], image_paths: Sequence[str]) -> str:
    task = _resolve_task(options, image_paths)
    prompt = options.get("prompt")
    question = options.get("question") or prompt
    if task == "vqa" and (not image_paths or not question):
        raise ValueError("vqa requires at least one image and --question or --prompt")
    if task == "edit" and (not image_paths or not prompt):
        raise ValueError("edit requires at least one image and --prompt")
    if task == "t2i" and (image_paths or not prompt):
        raise ValueError("t2i requires --prompt and does not accept input images")
    if task == "interleave" and not prompt:
        raise ValueError("interleave requires --prompt")
    _resolve_paths(image_paths, options.get("image_root", ""))
    explicit = _explicit_size(options, task)
    if explicit is not None:
        _validate_image_size(*explicit)
    cfg_interval = options["cfg_interval"]
    if len(cfg_interval) != 2 or not 0 <= float(cfg_interval[0]) <= float(cfg_interval[1]) <= 1:
        raise ValueError("cfg_interval must satisfy 0 <= LO <= HI <= 1")
    if int(options["num_steps"]) <= 0 or int(options["batch_size"]) <= 0:
        raise ValueError("num_steps and batch_size must be positive")
    if options["max_new_tokens"] is not None and int(options["max_new_tokens"]) <= 0:
        raise ValueError("max_new_tokens must be positive")
    if task != "t2i" and options["cfg_norm"] == "cfg_zero_star":
        raise ValueError("cfg_zero_star is supported by t2i only")
    return task


class UnifiedInference:
    def __init__(self, args: argparse.Namespace, profiler: InferenceProfiler) -> None:
        self.device = args.device
        self.prefetch_count = vram_mode_to_prefetch_count(args.vram_mode)
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[args.dtype]
        with profiler.time_load():
            self.model, self.tokenizer = load_model_and_tokenizer(
                args.model_path,
                dtype=dtype,
                device=args.device,
                gguf_checkpoint=args.gguf_checkpoint,
                device_map=args.device_map,
                max_memory=args.max_memory,
                for_offload=self.prefetch_count > 0,
            )
        if args.device_map:
            self.device = str(infer_input_device(self.model, fallback=args.device))
        if args.lora_path:
            print(f"[lora] loading {args.lora_path}")
            self.model = load_and_merge_lora_weight_from_safetensors(self.model, args.lora_path)

    def _offload(self):
        return make_offload_ctx(self.model, self.prefetch_count, self.device)

    def t2i(self, prompt: str, size: tuple[int, int], options: dict[str, Any]) -> tuple[list[Image.Image], str]:
        with self._offload() as model:
            output = model.t2i_generate(
                self.tokenizer,
                prompt,
                image_size=size,
                cfg_scale=float(options["cfg_scale"]),
                cfg_norm=options["cfg_norm"],
                timestep_shift=float(options["timestep_shift"]),
                enable_timestep_shift=bool(options["enable_timestep_shift"]),
                cfg_interval=tuple(float(value) for value in options["cfg_interval"]),
                num_steps=int(options["num_steps"]),
                batch_size=int(options["batch_size"]),
                t_eps=float(options["t_eps"]),
                seed=int(options["seed"]),
                think_mode=bool(options["think"]),
            )
        if options["think"]:
            tensor, think_text = output
            return _tensor_batch_to_pil(tensor), think_text
        return _tensor_batch_to_pil(output), ""

    def edit(
        self,
        prompt: str,
        images: Sequence[Image.Image],
        size: tuple[int, int],
        options: dict[str, Any],
    ) -> tuple[list[Image.Image], str]:
        if options["cfg_norm"] == "cfg_zero_star":
            raise ValueError("cfg_zero_star is supported by t2i only; use none, global, or channel for editing")
        with self._offload() as model:
            output = model.it2i_generate(
                self.tokenizer,
                prompt,
                list(images),
                image_size=size,
                cfg_scale=float(options["cfg_scale"]),
                img_cfg_scale=float(options["img_cfg_scale"]),
                cfg_norm=options["cfg_norm"],
                timestep_shift=float(options["timestep_shift"]),
                enable_timestep_shift=bool(options["enable_timestep_shift"]),
                cfg_interval=tuple(float(value) for value in options["cfg_interval"]),
                num_steps=int(options["num_steps"]),
                batch_size=int(options["batch_size"]),
                t_eps=float(options["t_eps"]),
                seed=int(options["seed"]),
                think_mode=bool(options["think"]),
            )
        if options["think"]:
            tensor, think_text = output
            return _tensor_batch_to_pil(tensor), think_text
        return _tensor_batch_to_pil(output), ""

    def vqa(self, question: str, paths: Sequence[Path], options: dict[str, Any]) -> str:
        pixel_values: list[torch.Tensor] = []
        grids: list[torch.Tensor] = []
        for path in paths:
            pixels, grid = load_image_native(
                path,
                min_pixels=int(options["vqa_min_pixels"]),
                max_pixels=int(options["vqa_max_pixels"]),
                upscale=bool(options["vqa_upscale"]),
            )
            pixel_values.append(pixels)
            grids.append(grid)
        pixels = torch.cat(pixel_values).to(self.device, dtype=self.model.dtype)
        grid_hw = torch.cat(grids).to(self.device)
        placeholder_count = question.count("<image>")
        if placeholder_count > len(paths):
            raise ValueError(
                f"question contains {placeholder_count} <image> placeholders but only {len(paths)} images were provided"
            )
        missing = len(paths) - placeholder_count
        if missing > 0:
            if len(paths) == 1:
                question = "<image>\n" + question
            else:
                prefix = "".join(f"Image-{index + 1}: <image>\n" for index in range(missing))
                question = prefix + question
        generation_config: dict[str, Any] = {
            "max_new_tokens": int(options["max_new_tokens"] or 1024),
            "do_sample": bool(options["do_sample"]),
        }
        if options["do_sample"]:
            generation_config.update(
                temperature=float(options["temperature"]),
                top_p=float(options["top_p"]),
            )
            if options["top_k"] is not None:
                generation_config["top_k"] = int(options["top_k"])
        if options["repetition_penalty"] is not None:
            generation_config["repetition_penalty"] = float(options["repetition_penalty"])
        with self._offload() as model:
            return model.chat(
                self.tokenizer,
                pixels,
                question,
                generation_config,
                history=None,
                return_history=False,
                grid_hw=grid_hw,
            )

    def interleave(
        self,
        prompt: str,
        images: Sequence[Image.Image],
        size: tuple[int, int],
        options: dict[str, Any],
    ) -> tuple[str, list[Image.Image]]:
        if options["cfg_norm"] == "cfg_zero_star":
            raise ValueError("cfg_zero_star is supported by t2i only; use none, global, or channel for interleave")
        generation_config = SimpleNamespace(max_new_tokens=int(options["max_new_tokens"] or 8192))
        with self._offload() as model:
            text, tensors = model.interleave_gen(
                self.tokenizer,
                prompt,
                images=list(images),
                image_size=size,
                generation_config=generation_config,
                cfg_scale=float(options["cfg_scale"]),
                img_cfg_scale=float(options["img_cfg_scale"]),
                cfg_norm=options["cfg_norm"],
                max_images=int(options["max_images"]),
                timestep_shift=float(options["timestep_shift"]),
                enable_timestep_shift=bool(options["enable_timestep_shift"]),
                cfg_interval=tuple(float(value) for value in options["cfg_interval"]),
                num_steps=int(options["num_steps"]),
                t_eps=float(options["t_eps"]),
                think_mode=bool(options["think"]),
                system_message=options["system_message"],
                seed=int(options["seed"]),
                verbose=bool(options["verbose"]),
            )
        return text, [_tensor_image_to_pil(tensor) for tensor in tensors]


def _run_sample(
    engine: UnifiedInference,
    profiler: InferenceProfiler,
    options: dict[str, Any],
    image_paths: Sequence[str],
    output: Path | None,
    output_dir: Path,
    stem: str,
) -> dict[str, Any]:
    task = _resolve_task(options, image_paths)
    options = options.copy()
    if options["think"] is None:
        options["think"] = task == "interleave"
    resolved_paths = _resolve_paths(image_paths, options.get("image_root", ""))
    _seed_everything(int(options["seed"]))

    if task == "vqa":
        if not resolved_paths:
            raise ValueError("vqa requires at least one image")
        question = options.get("question") or options.get("prompt")
        if not question:
            raise ValueError("vqa requires --question or --prompt")
        with profiler.time_generate(width=1, height=1, batch=1):
            text = engine.vqa(str(question), resolved_paths, options)
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text, encoding="utf-8")
            print(f"[saved] {output}")
        else:
            print(text)
        return {"task": task, "text": text, "images": []}

    prompt = options.get("prompt")
    if not prompt:
        raise ValueError(f"{task} requires --prompt")

    if task == "t2i":
        if resolved_paths:
            raise ValueError("t2i does not accept input images; use --task edit or interleave")
        size = _explicit_size(options, task) or T2I_RESOLUTIONS["1:1"]
        _validate_image_size(*size)
        with profiler.time_generate(*size, int(options["batch_size"])):
            images, think_text = engine.t2i(str(prompt), size, options)
        image_output = output or Path("output.png")
        paths = _save_images(images, image_output)
        if think_text:
            think_path = image_output.with_suffix(".think.txt")
            think_path.write_text(think_text, encoding="utf-8")
            print(f"[saved] {think_path}")
        return {
            "task": task,
            "prompt": prompt,
            "images": paths,
            "think": think_text,
            "width": size[0],
            "height": size[1],
        }

    input_images = [_load_rgb(path) for path in resolved_paths]
    if task == "edit":
        if not input_images:
            raise ValueError("edit requires at least one image")
        input_images = _resize_edit_inputs(input_images, options["input_max_pixels"], bool(options["do_resize"]))
        size = _editing_size(input_images, options)
        with profiler.time_generate(*size, int(options["batch_size"])):
            images, think_text = engine.edit(str(prompt), input_images, size, options)
        image_output = output or Path("output.png")
        paths = _save_images(images, image_output)
        if think_text:
            think_path = image_output.with_suffix(".think.txt")
            think_path.write_text(think_text, encoding="utf-8")
            print(f"[saved] {think_path}")
        if options["compare"]:
            save_compare(image_output, input_images, images[0], str(prompt))
        return {
            "task": task,
            "prompt": prompt,
            "images": paths,
            "think": think_text,
            "width": size[0],
            "height": size[1],
        }

    if task == "interleave":
        size = _interleave_size(input_images, options)
        with profiler.time_generate(*size, 1):
            text, images = engine.interleave(str(prompt), input_images, size, options)
        output_dir.mkdir(parents=True, exist_ok=True)
        text_path = output_dir / f"{stem}.txt"
        text_path.write_text(f"# PROMPT\n{prompt}\n\n# OUTPUT\n{text}\n", encoding="utf-8")
        print(f"[saved] {text_path}")
        paths = _save_images(images, output_dir / f"{stem}_image.png")
        profiler.update_last_batch(len(images))
        return {"task": task, "prompt": prompt, "text": text, "images": paths, "width": size[0], "height": size[1]}

    raise ValueError(f"unsupported task: {task}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified SenseNova-U1 inference: T2I, editing, VQA, and interleaved text/image generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", required=True, help="Hugging Face model id or local checkpoint directory.")
    parser.add_argument("--lora_path", default=None, help="Optional LoRA model id or local safetensors path.")
    parser.add_argument("--gguf_checkpoint", default=None, help="Optional GGUF checkpoint override.")
    parser.add_argument("--task", choices=["auto", "t2i", "edit", "vqa", "interleave"], default="auto")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--prompt", help="Prompt, edit instruction, or VQA question.")
    source.add_argument(
        "--jsonl",
        help="Batch JSONL; each row may override task, inputs, resolution, seed, and sampling.",
    )
    parser.add_argument("--question", help="VQA question. Makes --task auto select VQA.")
    parser.add_argument(
        "--image",
        nargs="+",
        action="append",
        metavar="PATH",
        help="Optional input image(s). May be repeated. Auto mode treats prompt + image as editing.",
    )
    parser.add_argument("--image_root", default="", help="Base directory for relative image paths.")

    parser.add_argument(
        "--output",
        default=None,
        help="Single output path. Images default to output.png; VQA defaults to stdout.",
    )
    parser.add_argument("--output_dir", default="outputs", help="Batch or interleave output directory.")
    parser.add_argument("--stem", default="sample", help="Single interleave output filename stem.")

    parser.add_argument("--resolution", default=None, help="Aspect bucket such as 1:1/16:9, or explicit WIDTHxHEIGHT.")
    parser.add_argument("--width", type=int, default=None, help="Explicit output width; requires --height.")
    parser.add_argument("--height", type=int, default=None, help="Explicit output height; requires --width.")
    parser.add_argument(
        "--target_pixels",
        type=int,
        default=DEFAULT_TARGET_PIXELS,
        help="Auto-sized editing output area.",
    )
    parser.add_argument("--input_max_pixels", default="auto", help="Edit input budget: integer, auto, or none.")
    parser.add_argument("--do_resize", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="Text classifier-free guidance scale.")
    parser.add_argument("--img_cfg_scale", type=float, default=1.0, help="Input-image guidance scale.")
    parser.add_argument("--cfg_norm", choices=["none", "global", "channel", "cfg_zero_star"], default="none")
    parser.add_argument("--timestep_shift", type=float, default=3.0)
    parser.add_argument("--enable_timestep_shift", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cfg_interval", type=float, nargs=2, default=[0.0, 1.0], metavar=("LO", "HI"))
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--t_eps", type=float, default=0.02, help="Flow endpoint epsilon.")
    parser.add_argument(
        "--think",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Reason before generation. Defaults off for T2I/edit and on for interleave.",
    )
    parser.add_argument("--max_images", type=int, default=10, help="Maximum generated images for interleave.")
    parser.add_argument("--system_message", default=DEFAULT_SYSTEM_MESSAGE, help="Interleave system prompt.")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="VQA/interleave text limit. Defaults to 1024 for VQA and 8192 for interleave.",
    )
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=False, help="VQA text sampling.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--vqa_min_pixels", type=int, default=65536)
    parser.add_argument("--vqa_max_pixels", type=int, default=4194304)
    parser.add_argument("--vqa_upscale", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--device", default=str(best_available_device()))
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    add_offload_args(parser)
    parser.add_argument("--attn_backend", choices=["auto", "flash", "sdpa"], default="auto")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--compare", action="store_true", help="Save an edit input/output comparison montage.")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.prompt is None and args.question is None and args.jsonl is None:
        parser.error("one of --prompt, --question, or --jsonl is required")
    if (args.width is None) != (args.height is None):
        parser.error("--width and --height must be provided together")
    if args.question and args.jsonl:
        parser.error("--question is only valid for single-sample inference")
    if args.image and args.jsonl:
        parser.error("--image is only valid with --prompt; put image paths in each JSONL row")
    if args.input_max_pixels is not None and str(args.input_max_pixels).lower() == "none":
        args.input_max_pixels = None
    if args.num_steps <= 0 or args.batch_size <= 0 or (args.max_new_tokens is not None and args.max_new_tokens <= 0):
        parser.error("--num_steps, --batch_size, and --max_new_tokens must be positive")
    if not 0 <= args.cfg_interval[0] <= args.cfg_interval[1] <= 1:
        parser.error("--cfg_interval must satisfy 0 <= LO <= HI <= 1")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)

    single_options: dict[str, Any] | None = None
    single_images: list[str] = []
    samples: list[dict[str, Any]] = []
    if args.jsonl is None:
        single_options = _sample_options(args)
        single_images = _flatten_cli_images(args.image)
        try:
            _preflight_sample(single_options, single_images)
        except (FileNotFoundError, TypeError, ValueError) as exc:
            parser.error(str(exc))
    else:
        jsonl_path = Path(args.jsonl)
        try:
            samples = [
                json.loads(line)
                for line in jsonl_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            for index, sample in enumerate(samples, 1):
                options = _sample_options(args, sample)
                image_paths = _coerce_paths(sample.get("image", sample.get("images")))
                _preflight_sample(options, image_paths)
        except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError) as exc:
            parser.error(f"invalid JSONL input {args.jsonl!r}: {exc}")

    sensenova_u1.set_attn_backend(args.attn_backend)
    print(f"[attn] requested={args.attn_backend!r}, effective={sensenova_u1.effective_attn_backend()!r}")
    profiler = InferenceProfiler(
        enabled=args.profile,
        device=args.device,
        config={
            "vram_mode": args.vram_mode,
            "attn_backend": sensenova_u1.effective_attn_backend(),
            "dtype": args.dtype,
            "gguf": args.gguf_checkpoint,
        },
    )
    engine = UnifiedInference(args, profiler)

    if args.jsonl is None:
        assert single_options is not None
        result = _run_sample(
            engine,
            profiler,
            single_options,
            single_images,
            Path(args.output) if args.output else None,
            Path(args.output_dir),
            args.stem,
        )
        if args.task == "interleave" or result["task"] != "vqa":
            print(json.dumps(result, ensure_ascii=False, indent=2))
        profiler.report()
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as results_file:
        for index, sample in enumerate(samples, 1):
            options = _sample_options(args, sample)
            image_paths = _coerce_paths(sample.get("image", sample.get("images")))
            task = _resolve_task(options, image_paths)
            suffix = ".txt" if task == "vqa" else ".png"
            stem = str(sample.get("id") or sample.get("type") or f"{index:04d}")
            result = _run_sample(
                engine,
                profiler,
                options,
                image_paths,
                output_dir / f"{stem}{suffix}",
                output_dir,
                stem,
            )
            result["index"] = index - 1
            result["id"] = sample.get("id")
            results_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            results_file.flush()
    print(f"[saved] {results_path}")
    profiler.report()


if __name__ == "__main__":
    main()
