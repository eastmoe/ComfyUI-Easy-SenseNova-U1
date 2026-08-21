"""Loading support for checkpoints produced by ``tools/quantize_checkpoint.py``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

import comfy.ops
from comfy.quant_ops import QuantizedTensor, get_layout_class


SUPPORTED_METHODS = (
    "int8_convrot",
    "mxfp8",
    "w4a8_convrot",
    "mxfp4",
    "nvfp4",
)


def checkpoint_quantization(model_path: str | Path) -> str | None:
    checkpoint = Path(model_path) / "model.safetensors"
    if not checkpoint.is_file():
        return None
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        method = (handle.metadata() or {}).get("sensenova_quantization")
    if method is None:
        return None
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported SenseNova quantization method: {method}")
    return method


class MXFP4Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool, compute_dtype: torch.dtype):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        self.register_buffer("qdata", None)
        self.register_buffer("scale", None)
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, dtype=compute_dtype))
        else:
            self.register_parameter("bias", None)

    @property
    def weight(self):
        return torch.empty((), device=self.qdata.device, dtype=self.compute_dtype).expand(
            self.out_features, self.in_features
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        try:
            from torchao.prototype.mx_formats.mx_tensor import to_dtype
        except (ImportError, AttributeError) as exc:
            raise RuntimeError("MXFP4 inference requires a compatible torchao>=0.16") from exc

        weight = to_dtype(
            self.qdata,
            self.scale,
            torch.float4_e2m1fn_x2,
            32,
            self.compute_dtype,
        )
        return torch.nn.functional.linear(input_tensor.to(self.compute_dtype), weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"storage=mxfp4, compute={self.compute_dtype}"
        )


class BufferedQuantLinear(torch.nn.Module):
    """Quantized Linear whose offloadable state consists only of plain tensors."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        compute_dtype: torch.dtype,
        config: dict[str, Any],
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        self.config = config
        self.register_buffer("qdata", None)
        self.register_buffer("scale", None)
        self.register_buffer("scale_2", None)
        self.register_buffer("s_channel", None)
        self.register_buffer("codebook", None)
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, dtype=compute_dtype))
        else:
            self.register_parameter("bias", None)

    @property
    def weight(self):
        return torch.empty((), device=self.qdata.device, dtype=self.compute_dtype).expand(
            self.out_features, self.in_features
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        quant_format = self.config["format"]
        if quant_format == "int8_tensorwise":
            layout_name = "TensorWiseINT8Layout"
            values = {
                "scale": self.scale,
                "orig_dtype": self.compute_dtype,
                "orig_shape": (self.out_features, self.in_features),
                "is_weight": True,
                "convrot": True,
                "convrot_groupsize": int(self.config["convrot_groupsize"]),
                "transposed": False,
            }
        elif quant_format == "mxfp8":
            layout_name = "TensorCoreMXFP8Layout"
            values = {
                "scale": self.scale,
                "orig_dtype": self.compute_dtype,
                "orig_shape": (self.out_features, self.in_features),
                "transposed": False,
            }
        elif quant_format == "nvfp4":
            layout_name = "TensorCoreNVFP4Layout"
            values = {
                "scale": self.scale_2,
                "block_scale": self.scale,
                "orig_dtype": self.compute_dtype,
                "orig_shape": (self.out_features, self.in_features),
                "transposed": False,
            }
        elif quant_format == "asym_w4a8_int8":
            layout_name = "AsymW4A8Int8Layout"
            values = {
                "scale": self.scale,
                "orig_dtype": self.compute_dtype,
                "orig_shape": (self.out_features, self.in_features),
                "s_channel": self.s_channel,
                "correction": None,
                "codebook": self.codebook,
                "group_size": int(self.config["group_size"]),
                "convrot_groupsize": int(self.config["convrot_groupsize"]),
                "transposed": False,
            }
        else:
            raise ValueError(f"Unsupported buffered quantization format: {quant_format}")
        params = get_layout_class(layout_name).Params(**values)
        weight = QuantizedTensor(self.qdata, layout_name, params)
        if quant_format not in {"mxfp8", "nvfp4"}:
            return torch.nn.functional.linear(input_tensor, weight, self.bias)

        input_shape = input_tensor.shape
        matrix = input_tensor.to(self.compute_dtype)
        matrix = matrix.reshape(-1, input_shape[-1]) if matrix.ndim >= 3 else matrix
        if matrix.ndim != 2:
            return torch.nn.functional.linear(input_tensor, weight, self.bias)
        use_quantized_input = quant_format == "mxfp8"
        if quant_format == "nvfp4":
            try:
                import comfy.model_management as model_management

                use_quantized_input = model_management.supports_nvfp4_compute(
                    matrix.device
                )
            except Exception:
                pass
        if use_quantized_input:
            matrix = QuantizedTensor.from_float(matrix, layout_name)
        output = torch.nn.functional.linear(matrix, weight, self.bias)
        if input_tensor.ndim >= 3:
            output = output.reshape(*input_shape[:-1], self.out_features)
        return output

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"storage={self.config['format']}, compute={self.compute_dtype}"
        )


def _layer_config(handle, layer: str) -> dict[str, Any]:
    value = handle.get_tensor(f"{layer}.comfy_quant")
    return json.loads(value.numpy().tobytes())


def _comfy_quant_linear(
    old: torch.nn.Linear,
    handle,
    layer: str,
    compute_dtype: torch.dtype,
    buffered: bool,
):
    config = _layer_config(handle, layer)
    # NVFP4 needs an explicit capability gate: Comfy's generic Linear always
    # quantizes activations, but swizzled NVFP4 activation tensors are invalid
    # on CPU and unsupported GPUs. The buffered implementation keeps native
    # NVFP4 matrix multiplication on Blackwell and cleanly dequantizes elsewhere.
    if buffered or config["format"] == "nvfp4":
        module = BufferedQuantLinear(
            old.in_features,
            old.out_features,
            old.bias is not None,
            compute_dtype,
            config,
        )
        module.qdata = handle.get_tensor(f"{layer}.weight")
        scale_name = "weight_s_rel" if config["format"] == "asym_w4a8_int8" else "weight_scale"
        module.scale = handle.get_tensor(f"{layer}.{scale_name}")
        consumed = {
            f"{layer}.weight",
            f"{layer}.{scale_name}",
            f"{layer}.comfy_quant",
        }
        if config["format"] == "nvfp4":
            module.scale_2 = handle.get_tensor(f"{layer}.weight_scale_2")
            consumed.add(f"{layer}.weight_scale_2")
        if config["format"] == "asym_w4a8_int8":
            module.s_channel = handle.get_tensor(f"{layer}.weight_s_channel")
            module.codebook = handle.get_tensor(f"{layer}.weight_codebook")
            consumed.update(
                {f"{layer}.weight_s_channel", f"{layer}.weight_codebook"}
            )
        if old.bias is not None:
            bias_key = f"{layer}.bias"
            module.bias = torch.nn.Parameter(
                handle.get_tensor(bias_key).to(compute_dtype), requires_grad=False
            )
            consumed.add(bias_key)
        return module, consumed, config

    operations = comfy.ops.mixed_precision_ops({}, compute_dtype=compute_dtype)
    module = operations.Linear(
        old.in_features,
        old.out_features,
        bias=old.bias is not None,
        device="cpu",
        dtype=compute_dtype,
    )
    keys = {
        "weight",
        "weight_scale",
        "weight_scale_2",
        "weight_s_rel",
        "weight_s_channel",
        "weight_codebook",
        "comfy_quant",
    }
    if old.bias is not None:
        keys.add("bias")
    state = {
        suffix: handle.get_tensor(f"{layer}.{suffix}")
        for suffix in keys
        if f"{layer}.{suffix}" in handle.keys()
    }
    missing, unexpected = module.load_state_dict(state, strict=False)
    missing = [name for name in missing if name != "weight"]
    if missing or unexpected:
        raise ValueError(f"Invalid quantized Linear {layer}: missing={missing}, unexpected={unexpected}")
    return module, {f"{layer}.{name}" for name in state}, config


def _mxfp4_linear(old: torch.nn.Linear, handle, layer: str, compute_dtype: torch.dtype):
    config = _layer_config(handle, layer)
    if config != {"format": "sensenova_mxfp4", "block_size": 32}:
        raise ValueError(f"Invalid MXFP4 config for {layer}: {config}")
    module = MXFP4Linear(old.in_features, old.out_features, old.bias is not None, compute_dtype)
    module.qdata = handle.get_tensor(f"{layer}.weight")
    module.scale = handle.get_tensor(f"{layer}.weight_scale")
    loaded = {
        f"{layer}.weight",
        f"{layer}.weight_scale",
        f"{layer}.comfy_quant",
    }
    if old.bias is not None:
        bias_key = f"{layer}.bias"
        module.bias = torch.nn.Parameter(handle.get_tensor(bias_key).to(compute_dtype), requires_grad=False)
        loaded.add(bias_key)
    return module, loaded, config


def load_prequantized_model(
    private_transformers,
    config,
    model_path: str | Path,
    *,
    dtype: torch.dtype,
    device: str | torch.device | None,
    device_map: str | None,
    for_offload: bool,
) -> tuple[torch.nn.Module, str, int]:
    if device_map:
        raise RuntimeError("Pre-quantized SenseNova checkpoints do not yet support device_map; use full/balanced/low on one device")
    try:
        from accelerate import init_empty_weights
        from accelerate.utils import set_module_tensor_to_device
    except ImportError as exc:
        raise RuntimeError("Pre-quantized checkpoint loading requires accelerate>=1") from exc

    checkpoint = Path(model_path) / "model.safetensors"
    with init_empty_weights():
        model = private_transformers.AutoModel.from_config(config)

    loaded_keys: set[str] = set()
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        method = metadata.get("sensenova_quantization")
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"Unsupported SenseNova quantization method: {method}")
        layer_names = sorted(key[:-len(".comfy_quant")] for key in handle.keys() if key.endswith(".comfy_quant"))
        if not layer_names:
            raise ValueError("Quantized checkpoint contains no Linear descriptors")

        modules = dict(model.named_modules())
        for layer in layer_names:
            old = modules.get(layer)
            if not isinstance(old, torch.nn.Linear):
                raise ValueError(f"Quantized checkpoint key is not a model Linear: {layer}")
            if method == "mxfp4":
                replacement, consumed, _ = _mxfp4_linear(old, handle, layer, dtype)
            else:
                replacement, consumed, _ = _comfy_quant_linear(
                    old, handle, layer, dtype, buffered=for_offload
                )
            model.set_submodule(layer, replacement)
            loaded_keys.update(consumed)

        checkpoint_keys = set(handle.keys())
        tensors = list(model.named_parameters(remove_duplicate=False)) + list(
            model.named_buffers(remove_duplicate=False)
        )
        for name, _ in tensors:
            if name in loaded_keys or name not in checkpoint_keys:
                continue
            value = handle.get_tensor(name)
            if value.is_floating_point():
                value = value.to(dtype)
            set_module_tensor_to_device(model, name, "cpu", value=value)
            loaded_keys.add(name)

    model.tie_weights()
    meta = [name for name, value in model.named_parameters() if value.device.type == "meta"]
    if meta:
        raise ValueError(f"Checkpoint is missing model parameters: {', '.join(meta[:8])}")
    model.eval()
    if not for_offload and device is not None:
        model = model.to(device)
    return model, method, len(layer_names)


__all__ = ["SUPPORTED_METHODS", "checkpoint_quantization", "load_prequantized_model"]
