"""加载固定版本的私有 Transformers 与 SenseNova-U1 后端。"""

from __future__ import annotations

import importlib
import logging
from typing import Any

import numpy as np
import torch
from packaging.version import InvalidVersion, Version

from .paths import ensure_origin_source
from .transformer_patch import PINNED_VERSION, load_transformers


LOGGER = logging.getLogger(__name__)
MIN_TORCH = Version("2.6.0")
MAX_CHECKED_TORCH = Version("2.14.0")
MIN_NUMPY = Version("1.24.0")
MAX_NUMPY = Version("3.0.0")
_WARNED_UNCHECKED_TORCH = False


def _version(value: str) -> Version:
    try:
        return Version(value)
    except InvalidVersion as exc:
        raise RuntimeError(f"无法解析依赖版本: {value}") from exc


def runtime_report() -> dict[str, str]:
    """校验数值运行时，并报告插件实际使用的私有版本。"""
    global _WARNED_UNCHECKED_TORCH

    private_transformers = load_transformers()
    torch_version = _version(torch.__version__)
    numpy_version = _version(np.__version__)

    if torch_version < MIN_TORCH:
        raise RuntimeError(
            f"SenseNova-U1 要求 PyTorch>=2.6；当前为 {torch.__version__}。"
        )
    if not MIN_NUMPY <= numpy_version < MAX_NUMPY:
        raise RuntimeError(
            f"SenseNova-U1 需要 NumPy>=1.24,<3；当前为 {np.__version__}。"
        )

    torch_status = "支持（已实测 2.6/2.12/2.13）"
    if torch_version >= MAX_CHECKED_TORCH:
        torch_status = "超出已检查范围"
        if not _WARNED_UNCHECKED_TORCH:
            LOGGER.warning(
                "PyTorch %s 超出本插件已检查的 2.6–2.13 范围。",
                torch.__version__,
            )
            _WARNED_UNCHECKED_TORCH = True

    return {
        "transformers": private_transformers.__version__,
        "transformers_source": "插件私有补丁",
        "transformers_pin": PINNED_VERSION,
        "pytorch": torch.__version__,
        "pytorch_status": torch_status,
        "numpy": np.__version__,
        "numpy_api": f"{numpy_version.major}.x",
    }


def import_sensenova_backend():
    """先建立私有 API 命名空间，再注册随插件分发的后端。"""
    runtime_report()
    ensure_origin_source()
    return importlib.import_module("sensenova_u1")


def load_model_and_tokenizer(
    model_path: str,
    *,
    dtype: torch.dtype,
    device: str | torch.device | None = None,
    device_map: str | None = None,
    max_memory: str | dict[int | str, str] | None = None,
    for_offload: bool = False,
) -> tuple[torch.nn.Module, Any]:
    """始终用插件私有的 Transformers 4.57.1 加载本地模型。"""
    private_transformers = load_transformers()
    backend = import_sensenova_backend()
    from sensenova_u1.utils import parse_max_memory

    if for_offload and device_map:
        LOGGER.warning("层卸载模式会忽略 device_map=%r。", device_map)
        device_map = None

    config = private_transformers.AutoConfig.from_pretrained(model_path)
    backend.check_checkpoint_compatibility(config)
    tokenizer = private_transformers.AutoTokenizer.from_pretrained(model_path)

    model_kwargs: dict[str, Any] = {
        "config": config,
        "dtype": dtype,
    }
    if device_map:
        model_kwargs["device_map"] = device_map
        parsed_max_memory = parse_max_memory(max_memory)
        if parsed_max_memory:
            model_kwargs["max_memory"] = parsed_max_memory

    model = private_transformers.AutoModel.from_pretrained(
        model_path, **model_kwargs
    ).eval()
    if not device_map and device is not None and not for_offload:
        model = model.to(device)
    return model, tokenizer
