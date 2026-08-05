"""ComfyUI-side compatibility helpers for the bundled SenseNova-U1 backend.

The upstream source is kept unchanged under ``origin/``.  Version-sensitive
Transformers behavior is normalized here before a checkpoint is instantiated.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from typing import Any

import numpy as np
import torch
from packaging.version import InvalidVersion, Version

from .paths import ensure_origin_source


LOGGER = logging.getLogger(__name__)
MIN_TRANSFORMERS = Version("4.57.1")
MAX_TRANSFORMERS = Version("6.0.0")
MIN_TORCH = Version("2.6.0")
MAX_CHECKED_TORCH = Version("2.13.0")
MIN_NUMPY = Version("1.24.0")
MAX_NUMPY = Version("3.0.0")
_WARNED_UNCHECKED_TORCH = False
_BACKEND_PATCHED = False


def _version(value: str) -> Version:
    """Parse release versions while tolerating vendor/local suffixes."""
    try:
        return Version(value)
    except InvalidVersion as exc:
        raise RuntimeError(f"无法解析依赖版本: {value}") from exc


def compatibility_report() -> dict[str, str]:
    """Validate the supported dependency window and return UI diagnostics."""
    global _WARNED_UNCHECKED_TORCH

    transformers = importlib.import_module("transformers")
    transformers_version = _version(transformers.__version__)
    torch_version = _version(torch.__version__)
    numpy_version = _version(np.__version__)

    if not MIN_TRANSFORMERS <= transformers_version < MAX_TRANSFORMERS:
        raise RuntimeError(
            "SenseNova-U1 需要 transformers>=4.57.1,<6；"
            f"当前为 {transformers.__version__}。"
        )
    if torch_version < MIN_TORCH:
        raise RuntimeError(
            f"SenseNova-U1 兼容层要求 PyTorch>=2.6；当前为 {torch.__version__}。"
        )
    if not MIN_NUMPY <= numpy_version < MAX_NUMPY:
        raise RuntimeError(
            f"SenseNova-U1 需要 NumPy>=1.24,<3；当前为 {np.__version__}。"
        )

    torch_status = "支持（已实测 2.6/2.12 边界）"
    if torch_version >= MAX_CHECKED_TORCH:
        torch_status = "超出已检查范围（兼容模式）"
        if not _WARNED_UNCHECKED_TORCH:
            LOGGER.warning(
                "PyTorch %s 超出本插件已检查的 2.6–2.12 范围，将继续使用兼容模式。",
                torch.__version__,
            )
            _WARNED_UNCHECKED_TORCH = True

    return {
        "transformers": transformers.__version__,
        "transformers_api": "4.57" if transformers_version.major == 4 else "5.x",
        "pytorch": torch.__version__,
        "pytorch_status": torch_status,
        "numpy": np.__version__,
        "numpy_api": f"{numpy_version.major}.x",
    }


def import_sensenova_backend():
    """Register and import the bundled backend after dependency validation."""
    compatibility_report()
    ensure_origin_source()
    backend = importlib.import_module("sensenova_u1")
    _patch_transformers_5_rope_initialization()
    return backend


def _patch_transformers_5_rope_initialization() -> None:
    """Adapt RoPE initialization and causal-mask keywords for Transformers 5."""
    global _BACKEND_PATCHED
    if _BACKEND_PATCHED:
        return
    transformers = importlib.import_module("transformers")
    if _version(transformers.__version__).major < 5:
        _BACKEND_PATCHED = True
        return

    modeling = importlib.import_module(
        "sensenova_u1.models.neo_unify.modeling_qwen3"
    )
    rotary_class = modeling.Qwen3RotaryEmbedding
    if not hasattr(rotary_class, "compute_default_rope_parameters"):

        def compute_default_rope_parameters(self, config=None, device=None, **kwargs):
            target_config = self.config if config is None else config
            return modeling._compute_default_rope_parameters(
                target_config, device=device, **kwargs
            )

        rotary_class.compute_default_rope_parameters = compute_default_rope_parameters

    masking = importlib.import_module("transformers.masking_utils")
    create_causal_mask = masking.create_causal_mask
    mask_parameters = inspect.signature(create_causal_mask).parameters

    def create_causal_mask_compat(**kwargs):
        if "inputs_embeds" in mask_parameters and "input_embeds" in kwargs:
            kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
        if "cache_position" not in mask_parameters:
            kwargs.pop("cache_position", None)
        return create_causal_mask(**kwargs)

    modeling.create_causal_mask = create_causal_mask_compat
    moe_modeling = importlib.import_module(
        "sensenova_u1.models.neo_unify.modeling_qwen3_moe"
    )
    moe_modeling.create_causal_mask = create_causal_mask_compat
    _BACKEND_PATCHED = True


def _rope_parameters(config: Any) -> dict[str, Any] | None:
    for name in ("rope_parameters", "rope_scaling"):
        value = getattr(config, name, None)
        if isinstance(value, dict):
            return value
    return None


def normalize_sensenova_config(config: Any) -> Any:
    """Restore the Qwen3 RoPE aliases removed by Transformers 5.

    Transformers 4.57 exposes ``rope_theta`` directly.  Transformers 5 stores
    it in ``rope_parameters``; the bundled SenseNova implementation still
    needs the scalar attribute for its temporal and spatial rotary embeddings.
    """
    llm_config = getattr(config, "llm_config", config)
    if not hasattr(llm_config, "rope_theta"):
        rope_parameters = _rope_parameters(llm_config)
        rope_theta = rope_parameters.get("rope_theta") if rope_parameters else None
        if rope_theta is None:
            raise RuntimeError(
                "模型 LLM 配置缺少 rope_theta/rope_parameters.rope_theta，"
                "无法安全构造 SenseNova-U1 RoPE。"
            )
        llm_config.rope_theta = float(rope_theta)

    # Some 5.x code paths expose only rope_parameters while the bundled model
    # checks the historical rope_scaling alias to select the RoPE function.
    if not hasattr(llm_config, "rope_scaling"):
        rope_parameters = getattr(llm_config, "rope_parameters", None)
        if isinstance(rope_parameters, dict):
            llm_config.rope_scaling = dict(rope_parameters)
    return config


def load_model_and_tokenizer_compat(
    model_path: str,
    *,
    dtype: torch.dtype,
    device: str | torch.device | None = None,
    device_map: str | None = None,
    max_memory: str | dict[int | str, str] | None = None,
    for_offload: bool = False,
) -> tuple[torch.nn.Module, Any]:
    """Load a local checkpoint across Transformers 4.57 and 5.x."""
    backend = import_sensenova_backend()
    from sensenova_u1.utils import parse_max_memory
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    if for_offload and device_map:
        LOGGER.warning("层卸载模式会忽略 device_map=%r。", device_map)
        device_map = None

    config = normalize_sensenova_config(AutoConfig.from_pretrained(model_path))
    backend.check_checkpoint_compatibility(config)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    transformers_version = _version(
        importlib.import_module("transformers").__version__
    )
    model_kwargs: dict[str, Any] = {"config": config}
    if transformers_version.major >= 5:
        model_kwargs["dtype"] = dtype
    else:
        model_kwargs["torch_dtype"] = dtype

    if device_map:
        model_kwargs["device_map"] = device_map
        parsed_max_memory = parse_max_memory(max_memory)
        if parsed_max_memory:
            model_kwargs["max_memory"] = parsed_max_memory

    model = AutoModel.from_pretrained(model_path, **model_kwargs).eval()
    if not device_map and device is not None and not for_offload:
        model = model.to(device)
    return model, tokenizer
