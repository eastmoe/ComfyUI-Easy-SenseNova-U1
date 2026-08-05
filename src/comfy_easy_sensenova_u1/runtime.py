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

from .paths import ensure_origin_source


MODEL_TYPE = "EASY_SENSENOVA_U1_MODEL"
DEFAULT_SEED = 42
GRID_SIZE = 32
STORAGE_PRECISIONS = ("bfloat16", "float16", "float32")
COMPUTE_PRECISIONS = ("auto", "bfloat16", "float16", "float32")
ATTENTION_BACKENDS = ("auto", "flash", "sdpa")
VRAM_MODES = ("full", "balanced", "low")
DEVICE_MAPS = ("none", "auto", "balanced", "balanced_low_0", "sequential")
CFG_NORMS = ("none", "global", "channel", "cfg_zero_star")

T2I_RESOLUTIONS = {
    "2048x2048 (1:1)": (2048, 2048),
    "2720x1536 (16:9)": (2720, 1536),
    "1536x2720 (9:16)": (1536, 2720),
    "2496x1664 (3:2)": (2496, 1664),
    "1664x2496 (2:3)": (1664, 2496),
    "2368x1760 (4:3)": (2368, 1760),
    "1760x2368 (3:4)": (1760, 2368),
    "2880x1440 (2:1)": (2880, 1440),
    "1440x2880 (1:2)": (1440, 2880),
    "3456x1152 (3:1)": (3456, 1152),
    "1152x3456 (1:3)": (1152, 3456),
}
INTERLEAVE_RESOLUTIONS = {
    "1536x1536 (1:1)": (1536, 1536),
    "2048x1152 (16:9)": (2048, 1152),
    "1152x2048 (9:16)": (1152, 2048),
    "1888x1248 (3:2)": (1888, 1248),
    "1248x1888 (2:3)": (1248, 1888),
    "1760x1312 (4:3)": (1760, 1312),
    "1312x1760 (3:4)": (1312, 1760),
}
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
    gc.collect()
    try:
        import comfy.model_management as mm

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
        }

    def offload_context(self):
        ensure_origin_source()
        from sensenova_u1.utils import make_offload_ctx

        return make_offload_ctx(self.model, self.prefetch_count, self.device)

    def compute_context(self):
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
            ensure_origin_source()
            import sensenova_u1

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
) -> SenseNovaHandle:
    ensure_origin_source()
    import sensenova_u1
    from sensenova_u1.utils import infer_input_device, load_model_and_tokenizer, vram_mode_to_prefetch_count

    resolved_device = resolve_device(device)
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
            _clear_memory()
            sensenova_u1.set_attn_backend(attention_backend)
            model, tokenizer = load_model_and_tokenizer(
                model_path,
                dtype=dtype_from_name(storage_precision),
                device=resolved_device,
                device_map=normalized_map,
                max_memory=max_memory.strip() or None,
                for_offload=prefetch_count > 0,
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
