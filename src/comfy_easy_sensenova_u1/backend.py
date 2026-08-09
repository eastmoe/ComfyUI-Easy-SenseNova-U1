"""加载固定版本的私有 Transformers 与 SenseNova-U1 后端。"""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
from dataclasses import dataclass
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
_MX_CONFIG_CLASSES: dict[str, type] = {}


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
    dynamic_mx_precision: str | None = None,
    mx_compute_dtype: torch.dtype | None = None,
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
    if dynamic_mx_precision:
        model_kwargs["quantization_config"] = _mx_quantization_config(
            private_transformers,
            dynamic_mx_precision,
            mx_compute_dtype or torch.bfloat16,
        )
    if device_map:
        model_kwargs["device_map"] = device_map
        parsed_max_memory = parse_max_memory(max_memory)
        if parsed_max_memory:
            model_kwargs["max_memory"] = parsed_max_memory

    model = private_transformers.AutoModel.from_pretrained(
        model_path, **model_kwargs
    ).eval()
    if dynamic_mx_precision:
        _cast_non_mx_tensors(model, mx_compute_dtype or torch.bfloat16)
    if not device_map and device is not None and not for_offload:
        model = model.to(device)
    return model, tokenizer


def _mx_quantization_config(
    private_transformers, precision: str, compute_dtype: torch.dtype = torch.bfloat16
):
    """建立只量化权重的 TorchAO MX 加载配置。"""
    try:
        import torchao  # noqa: F401
        from torchao.core.config import AOBaseConfig
        from torchao.prototype.mx_formats.mx_tensor import MXTensor, ScaleCalculationMode
        from torchao.quantization.quantize_.common import KernelPreference
        from torchao.quantization.transform_module import register_quantize_module_handler
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "MXFP8/MXFP4 动态加载需要 torchao>=0.16；请在 ComfyUI 的 Python 环境安装兼容版本。"
        ) from exc

    ao_version = _version(importlib.metadata.version("torchao"))
    if ao_version < Version("0.16.0"):
        raise RuntimeError(f"MXFP8/MXFP4 需要 torchao>=0.16；当前为 {ao_version}。")

    dtype_names = {
        "mxfp8": "float8_e4m3fn",
        "mxfp4": "float4_e2m1fn_x2",
    }
    try:
        dtype_name = dtype_names[precision]
    except KeyError as exc:
        raise ValueError(f"不支持的 MX 精度: {precision}") from exc
    if not hasattr(torch, dtype_name):
        raise RuntimeError(
            f"当前 PyTorch {torch.__version__} 不提供 {dtype_name}，无法使用 {precision.upper()}。"
        )

    config_class = _MX_CONFIG_CLASSES.get(precision)
    if config_class is None:
        @dataclass
        class MXWeightOnlyConfig(AOBaseConfig):
            elem_dtype: torch.dtype = getattr(torch, dtype_name)
            block_size: int = 32
            compute_dtype: torch.dtype = torch.bfloat16

        # 让 Transformers 在读取每个 Linear.weight 时立即压缩，而不是先建立完整 BF16 模型。
        @register_quantize_module_handler(MXWeightOnlyConfig)
        def _mx_weight_only_transform(module, config):
            quantized = MXTensor.to_mx(
                module.weight,
                elem_dtype=config.elem_dtype,
                block_size=config.block_size,
                scaling_mode=ScaleCalculationMode.RCEIL,
                kernel_preference=KernelPreference.EMULATED,
                act_quant_kwargs=None,
                is_swizzled_scales=False,
            )
            _store_mx_weight(module, quantized, config.compute_dtype)
            return module

        MXWeightOnlyConfig.__name__ = f"{precision.upper()}WeightOnlyConfig"
        config_class = MXWeightOnlyConfig
        _MX_CONFIG_CLASSES[precision] = config_class

    ao_config = config_class(compute_dtype=compute_dtype)
    return private_transformers.TorchAoConfig(quant_type=ao_config)


class _MXStorageLinear(torch.nn.Linear):
    """以 MX 缓冲区驻留权重，并按选定 dtype 解量化后执行 Linear。"""

    @property
    def weight(self):
        # 后端仅用 weight 的 dtype/device/shape 查询；expand 不分配完整权重。
        return torch.empty(
            (), dtype=self._mx_compute_dtype, device=self._mx_qdata.device
        ).expand(self.out_features, self.in_features)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        from torchao.prototype.mx_formats.mx_tensor import to_dtype

        elem_dtype = getattr(torch, self._mx_elem_dtype_name)
        weight = to_dtype(
            self._mx_qdata,
            self._mx_scale,
            elem_dtype,
            self._mx_block_size,
            self._mx_compute_dtype,
        )
        bias = self.bias
        if bias is not None and bias.dtype != self._mx_compute_dtype:
            bias = bias.to(self._mx_compute_dtype)
        return torch.nn.functional.linear(
            input_tensor.to(self._mx_compute_dtype), weight, bias
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"storage={self._mx_precision}, compute={self._mx_compute_dtype}"
        )


def _store_mx_weight(module: torch.nn.Linear, weight, compute_dtype: torch.dtype) -> None:
    """立即把 MXTensor 拆成普通缓冲区，兼容 dispatch 与层卸载。"""
    if not all(hasattr(weight, attr) for attr in ("qdata", "scale", "elem_dtype", "block_size")):
        raise TypeError("TorchAO 未返回有效的 MXTensor 权重。")
    with torch.no_grad():
        qdata = weight.qdata
        scale = weight.scale
        elem_dtype = weight.elem_dtype
        module._parameters.pop("weight")
        module.register_buffer("_mx_qdata", qdata, persistent=False)
        module.register_buffer("_mx_scale", scale, persistent=False)
        module._mx_elem_dtype_name = str(elem_dtype).removeprefix("torch.")
        module._mx_block_size = int(weight.block_size)
        module._mx_compute_dtype = compute_dtype
        module._mx_precision = "mxfp4" if elem_dtype == torch.float4_e2m1fn_x2 else "mxfp8"
        module.__dict__.pop("extra_repr", None)
        module.__class__ = _MXStorageLinear


def _cast_non_mx_tensors(model: torch.nn.Module, compute_dtype: torch.dtype) -> None:
    """让未压缩参数跟随计算精度，同时保持 MX 缓冲区不变。"""
    # 非 MX 参数也跟随计算精度；MX qdata/scale 始终保留原始压缩格式。
    for module in model.modules():
        for parameter in module.parameters(recurse=False):
            if parameter.device.type != "meta" and parameter.is_floating_point():
                parameter.data = parameter.data.to(compute_dtype)
        for name, buffer in module.named_buffers(recurse=False):
            if name.startswith("_mx_") or buffer.device.type == "meta":
                continue
            if buffer.is_floating_point():
                buffer.data = buffer.data.to(compute_dtype)
