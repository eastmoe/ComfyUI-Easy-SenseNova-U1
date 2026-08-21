from __future__ import annotations

import gc
import json
import math
import threading
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

from .backend import (
    import_sensenova_backend,
    load_model_and_tokenizer,
    runtime_report,
)
from .quantized_checkpoint import SUPPORTED_METHODS, checkpoint_quantization


MODEL_TYPE = "EASY_SENSENOVA_U1_MODEL"
DEFAULT_SEED = 42
GRID_SIZE = 32
QUANTIZED_STORAGE_PRECISIONS = ("mxfp8", "mxfp4", "nvfp4")
PREQUANT_STORAGE_PRECISIONS = SUPPORTED_METHODS
STORAGE_PRECISIONS = (
    "bfloat16",
    "float16",
    "float32",
    *QUANTIZED_STORAGE_PRECISIONS,
)
COMPUTE_PRECISIONS = ("auto", "bfloat16", "float16", "float32")
ATTENTION_BACKENDS = ("auto", "flash", "sdpa")
VRAM_MODES = ("full", "balanced", "low")
DEVICE_MAPS = ("none", "auto", "balanced", "balanced_low_0", "sequential")
CFG_NORMS = ("none", "global", "channel", "cfg_zero_star")

DEFAULT_SYSTEM_MESSAGE = """You are a multimodal assistant capable of reasoning with text and images.
In Think Mode, place reasoning in <think></think> and interleave generated images with <image> tags.
After reasoning, provide a concise user-facing answer. Match the user's language."""

_CACHE_LOCK = threading.RLock()
_ATTENTION_LOCK = threading.RLock()
_MODEL_CACHE: dict[tuple[Any, ...], "SenseNovaHandle"] = {}


def dtype_from_name(name: str) -> torch.dtype:
    try:
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[name]
    except KeyError as exc:
        raise ValueError(f"不支持的精度: {name}") from exc


def available_devices() -> list[str]:
    devices = ["auto"]
    if torch.cuda.is_available():
        devices.extend(f"cuda:{index}" for index in range(torch.cuda.device_count()))
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        devices.extend(f"xpu:{index}" for index in range(torch.xpu.device_count()))
    devices.append("cpu")
    return list(dict.fromkeys(devices))


def resolve_device(value: str) -> str:
    if value != "auto":
        return value
    try:
        import comfy.model_management as mm

        return str(mm.get_torch_device())
    except Exception:
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu:0"
        return "cpu"


def comfy_to_pil_batch(image: torch.Tensor) -> list[Image.Image]:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    array = image.detach().cpu().float().clamp(0, 1).numpy()
    return [Image.fromarray((item * 255.0).round().astype(np.uint8), mode="RGB") for item in array]


def generated_to_comfy(batch: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor((0.5, 0.5, 0.5), device=batch.device, dtype=batch.dtype).view(1, 3, 1, 1)
    std = torch.tensor((0.5, 0.5, 0.5), device=batch.device, dtype=batch.dtype).view(1, 3, 1, 1)
    return (batch * std + mean).clamp(0, 1).permute(0, 2, 3, 1).float().cpu()


def _clear_memory() -> None:
    """卸载 ComfyUI 已托管模型并释放设备缓存。"""
    gc.collect()
    try:
        import comfy.model_management as mm

        mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@dataclass
class SenseNovaHandle:
    model: Any
    tokenizer: Any
    model_path: str
    device: str
    input_device: str
    storage_precision: str
    compute_precision: str
    attention_backend: str
    effective_attention_backend: str
    vram_mode: str
    prefetch_count: int
    device_map: str
    compatibility: dict[str, str]

    @property
    def info(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "device": self.device,
            "input_device": self.input_device,
            "storage_precision": self.storage_precision,
            "compute_precision": self.compute_precision,
            "attention_backend": self.attention_backend,
            "effective_attention_backend": self.effective_attention_backend,
            "vram_mode": self.vram_mode,
            "device_map": self.device_map,
            "compatibility": self.compatibility,
        }

    def offload_context(self):
        import_sensenova_backend()
        from sensenova_u1.utils import make_offload_ctx

        return make_offload_ctx(self.model, self.prefetch_count, self.device)

    def compute_context(self):
        if self.storage_precision in PREQUANT_STORAGE_PRECISIONS:
            # 预量化 Linear 自己执行量化计算或按 compute_precision 解量化。
            return nullcontext()
        precision = self.storage_precision if self.compute_precision == "auto" else self.compute_precision
        if precision == self.storage_precision:
            return nullcontext()
        dtype = dtype_from_name(precision)
        device_type = torch.device(self.input_device).type
        if dtype == torch.float32:
            raise RuntimeError("低精度存储权重不能以 float32 自动混合精度计算；请将存储精度也设为 float32。")
        if device_type == "cpu" and dtype == torch.float16:
            raise RuntimeError("CPU 不支持本节点的 float16 自动混合精度，请使用 bfloat16 或 float32。")
        try:
            return torch.autocast(device_type=device_type, dtype=dtype)
        except RuntimeError as exc:
            raise RuntimeError(f"设备 {self.input_device} 不支持 {precision} 自动混合精度。") from exc

    def generation_context(self):
        return _GenerationContext(self)


class _GenerationContext:
    def __init__(self, handle: SenseNovaHandle):
        self.handle = handle
        self.stack = ExitStack()

    def __enter__(self):
        try:
            self.stack.enter_context(_ATTENTION_LOCK)
            sensenova_u1 = import_sensenova_backend()

            sensenova_u1.set_attn_backend(self.handle.attention_backend)
            self.stack.enter_context(self.handle.compute_context())
            return self.stack.enter_context(self.handle.offload_context())
        except Exception:
            self.stack.close()
            raise

    def __exit__(self, exc_type, exc, tb):
        return self.stack.__exit__(exc_type, exc, tb)


def load_handle(
    model_path: str,
    device: str,
    storage_precision: str,
    compute_precision: str,
    attention_backend: str,
    vram_mode: str,
    device_map: str,
    max_memory: str,
    reload_model: bool,
    clear_memory_before_load: bool = False,
) -> SenseNovaHandle:
    sensenova_u1 = import_sensenova_backend()
    from sensenova_u1.utils import infer_input_device, vram_mode_to_prefetch_count

    resolved_device = resolve_device(device)
    prequantized = checkpoint_quantization(model_path)
    if prequantized:
        storage_precision = prequantized
    prefetch_count = vram_mode_to_prefetch_count(vram_mode)
    normalized_map = None if device_map == "none" else device_map
    if prefetch_count and normalized_map:
        raise RuntimeError("低显存层卸载与多卡 device_map 不能同时启用。")
    key = (
        model_path,
        resolved_device,
        storage_precision,
        compute_precision,
        attention_backend,
        vram_mode,
        device_map,
        max_memory.strip(),
    )
    with _CACHE_LOCK:
        if reload_model or key not in _MODEL_CACHE:
            _MODEL_CACHE.clear()
            gc.collect()
            if clear_memory_before_load:
                _clear_memory()
            sensenova_u1.set_attn_backend(attention_backend)
            dynamic_quant_precision = (
                storage_precision
                if not prequantized
                and storage_precision in QUANTIZED_STORAGE_PRECISIONS
                else None
            )
            quant_compute_precision = (
                "bfloat16" if compute_precision == "auto" else compute_precision
            )
            model, tokenizer = load_model_and_tokenizer(
                model_path,
                dtype=(
                    dtype_from_name(quant_compute_precision)
                    if prequantized
                    else torch.bfloat16
                    if dynamic_quant_precision
                    else dtype_from_name(storage_precision)
                ),
                device=resolved_device,
                device_map=normalized_map,
                max_memory=max_memory.strip() or None,
                for_offload=prefetch_count > 0,
                dynamic_quant_precision=dynamic_quant_precision,
                quant_compute_dtype=(
                    dtype_from_name(quant_compute_precision)
                    if dynamic_quant_precision
                    else None
                ),
            )
            input_device = str(infer_input_device(model, fallback=resolved_device))
            _MODEL_CACHE[key] = SenseNovaHandle(
                model=model,
                tokenizer=tokenizer,
                model_path=model_path,
                device=resolved_device,
                input_device=input_device,
                storage_precision=storage_precision,
                compute_precision=compute_precision,
                attention_backend=attention_backend,
                effective_attention_backend=sensenova_u1.effective_attn_backend(),
                vram_mode=vram_mode,
                prefetch_count=prefetch_count,
                device_map=device_map,
                compatibility=runtime_report(),
            )
        return _MODEL_CACHE[key]


def validate_size(width: int, height: int) -> None:
    if width <= 0 or height <= 0 or width % GRID_SIZE or height % GRID_SIZE:
        raise ValueError(f"宽和高必须为正数且能被 {GRID_SIZE} 整除，当前为 {width}x{height}。")


def target_size_for_edit(image: Image.Image, megapixels: float) -> tuple[int, int]:
    target = max(GRID_SIZE * GRID_SIZE, int(megapixels * 1_000_000))
    scale = math.sqrt(target / max(1, image.width * image.height))
    width = max(GRID_SIZE, round(image.width * scale / GRID_SIZE) * GRID_SIZE)
    height = max(GRID_SIZE, round(image.height * scale / GRID_SIZE) * GRID_SIZE)
    return width, height


def metadata(handle: SenseNovaHandle, task: str, **values: Any) -> str:
    return json.dumps({**handle.info, "task": task, **values}, ensure_ascii=False, indent=2)
