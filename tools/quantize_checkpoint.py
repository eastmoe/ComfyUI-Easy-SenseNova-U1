#!/usr/bin/env python3
"""Convert or quantize SenseNova Linear weights into a loadable ComfyUI checkpoint.

The model architecture is always created through this plugin's private patched
Transformers 4.57.1.  Inputs may be a Hugging Face repo id, a local HF snapshot,
or a converted single-file ``.safetensors`` checkpoint with embedded assets.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import struct
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import torch
from safetensors import safe_open

from checkpoint_assets import (
    ASSETS_FORMAT,
    ASSETS_FORMAT_KEY,
    build_assets_metadata,
    materialize_checkpoint_assets,
)


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parent.parent
PLUGIN_SRC = PLUGIN_ROOT / "src"
PINNED_TRANSFORMERS = "4.57.1+SenseNova-patch"
PIXEL_VAE_KEY = "vae.pixel_space_vae"
QUANT_METHODS = (
    "bf16",
    "int8_convrot",
    "mxfp8",
    "w4a8_convrot",
    "mxfp4",
    "nvfp4",
)


@dataclass(frozen=True)
class SourceTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    path: Path
    offset: int
    size: int


@dataclass(frozen=True)
class OutputTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    path: Path
    offset: int
    size: int


@dataclass(frozen=True)
class ResolvedSource:
    files: tuple[Path, ...]
    assets: Path
    label: str
    metadata: dict[str, str]


def read_safetensors(path: Path) -> tuple[dict, int]:
    with path.open("rb") as handle:
        raw_size = handle.read(8)
        if len(raw_size) != 8:
            raise ValueError(f"Not a safetensors file: {path}")
        header_size = struct.unpack("<Q", raw_size)[0]
        header = json.loads(handle.read(header_size).rstrip(b" \t\r\n\0"))
    return header, 8 + header_size


def source_files(directory: Path) -> tuple[Path, ...]:
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
        files = tuple(directory / name for name in sorted(set(weight_map.values())))
    elif (directory / "model.safetensors").is_file():
        files = (directory / "model.safetensors",)
    else:
        files = tuple(sorted(directory.glob("*.safetensors")))
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint shards: {', '.join(missing[:8])}")
    if not files:
        raise FileNotFoundError(f"No safetensors weights found in {directory}")
    return files


def resolve_source(value: str, revision: str, token: str) -> ResolvedSource:
    candidate = Path(value).expanduser()
    if candidate.exists():
        candidate = candidate.resolve()
        if candidate.is_dir():
            return ResolvedSource(source_files(candidate), candidate, str(candidate), {})
        if candidate.suffix.lower() not in {".safetensors", ".sft"}:
            raise ValueError(f"Only safetensors checkpoints are supported: {candidate}")
        header, _ = read_safetensors(candidate)
        metadata = dict(header.get("__metadata__", {}))
        return ResolvedSource(
            (candidate,),
            materialize_checkpoint_assets(candidate, metadata, COMFY_ROOT / "temp"),
            str(candidate),
            metadata,
        )

    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=value,
            revision=revision or None,
            token=token or None,
        )
    ).resolve()
    return ResolvedSource(source_files(snapshot), snapshot, value, {})


def collect_source_tensors(files: tuple[Path, ...]) -> tuple[list[SourceTensor], dict[str, str]]:
    tensors: list[SourceTensor] = []
    metadata: dict[str, str] = {}
    seen: set[str] = set()
    for path in files:
        header, data_start = read_safetensors(path)
        if not metadata:
            metadata = dict(header.get("__metadata__", {}))
        for name, info in header.items():
            if name == "__metadata__":
                continue
            if name in seen:
                raise ValueError(f"Duplicate tensor {name!r} in {path}")
            seen.add(name)
            start, end = info["data_offsets"]
            tensors.append(
                SourceTensor(
                    name=name,
                    dtype=info["dtype"],
                    shape=tuple(info["shape"]),
                    path=path,
                    offset=data_start + int(start),
                    size=int(end) - int(start),
                )
            )
    if (
        any(name.endswith(".comfy_quant") for name in seen)
        or metadata.get("sensenova_quantization")
        or metadata.get("_quantization_metadata")
    ):
        raise ValueError("The source is already quantized; start from an FP16/BF16/FP32 checkpoint")
    return tensors, metadata


def private_model_layout(assets: Path) -> tuple[set[str], set[str], str]:
    sys.path.insert(0, str(COMFY_ROOT))
    sys.path.insert(0, str(PLUGIN_SRC))

    from accelerate import init_empty_weights
    from comfy_easy_sensenova_u1.backend import import_sensenova_backend
    from comfy_easy_sensenova_u1.transformer_patch import load_transformers

    transformers = load_transformers()
    import_sensenova_backend()
    config = transformers.AutoConfig.from_pretrained(str(assets))
    with init_empty_weights():
        model = transformers.AutoModel.from_config(config)
    names = {name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
    state_keys = set(model.state_dict())
    del model
    return names, state_keys, transformers.__version__


def select_linear_weights(
    tensors: list[SourceTensor],
    linear_names: set[str],
    include: re.Pattern[str] | None,
    excludes: tuple[re.Pattern[str], ...],
) -> dict[str, str]:
    source_names = {tensor.name for tensor in tensors}
    source_tensors = {tensor.name: tensor for tensor in tensors}
    selected: dict[str, str] = {}
    for name in sorted(linear_names):
        weight = f"{name}.weight"
        if weight not in source_names:
            continue
        if include is not None and include.search(name) is None:
            continue
        if any(pattern.search(name) for pattern in excludes):
            continue
        if source_tensors[weight].dtype not in {"BF16", "F16", "F32"}:
            raise ValueError(
                f"Linear source weight must be BF16/F16/F32, got {source_tensors[weight].dtype}: {weight}"
            )
        selected[weight] = name
    if not selected:
        raise ValueError("No Linear weights matched the checkpoint and filters")
    return selected


def quantize_weight(weight: torch.Tensor, method: str) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    if method == "bf16":
        return {"weight": weight.to(torch.bfloat16)}, {"format": "bf16"}

    # comfy-kitchen's CUDA NVFP4 quantizer accepts FP16/BF16 only. Some
    # SenseNova checkpoints store a substantial subset of Linear weights as
    # FP32, so narrow those inputs before invoking the kernel. NVFP4's 4-bit
    # payload cannot retain the additional FP32 mantissa precision anyway.
    if method == "nvfp4" and weight.dtype == torch.float32:
        weight = weight.to(torch.bfloat16)

    if method != "mxfp4":
        from comfy.quant_ops import QuantizedTensor

        if method == "int8_convrot":
            quantized = QuantizedTensor.from_float(
                weight,
                "TensorWiseINT8Layout",
                is_weight=True,
                per_channel=True,
                convrot=True,
                convrot_groupsize=256,
            )
            return (
                {"weight": quantized._qdata, "weight_scale": quantized._params.scale},
                {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
            )
        if method == "mxfp8":
            quantized = QuantizedTensor.from_float(weight, "TensorCoreMXFP8Layout")
            return (
                {"weight": quantized._qdata, "weight_scale": quantized._params.scale},
                {"format": "mxfp8"},
            )
        if method == "nvfp4":
            quantized = QuantizedTensor.from_float(weight, "TensorCoreNVFP4Layout")
            return (
                {
                    "weight": quantized._qdata,
                    "weight_scale": quantized._params.block_scale,
                    "weight_scale_2": quantized._params.scale,
                },
                {"format": "nvfp4"},
            )
        if method == "w4a8_convrot":
            quantized = QuantizedTensor.from_float(
                weight,
                "AsymW4A8Int8Layout",
                group_size=16,
                convrot_groupsize=256,
            )
            return (
                {
                    "weight": quantized._qdata,
                    "weight_s_rel": quantized._params.scale,
                    "weight_s_channel": quantized._params.s_channel,
                    "weight_codebook": quantized._params.codebook,
                },
                {"format": "asym_w4a8_int8", "group_size": 16, "convrot_groupsize": 256},
            )
        raise ValueError(f"Unsupported quantization method: {method}")

    try:
        from torchao.prototype.mx_formats.mx_tensor import MXTensor, ScaleCalculationMode
        from torchao.quantization.quantize_.common import KernelPreference
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("MXFP4 quantization requires a compatible torchao>=0.16") from exc

    quantized = MXTensor.to_mx(
        weight,
        elem_dtype=torch.float4_e2m1fn_x2,
        block_size=32,
        scaling_mode=ScaleCalculationMode.RCEIL,
        kernel_preference=KernelPreference.EMULATED,
        act_quant_kwargs=None,
        is_swizzled_scales=False,
    )
    return (
        {"weight": quantized.qdata, "weight_scale": quantized.scale},
        {"format": "sensenova_mxfp4", "block_size": 32},
    )


def tensor_dtype_name(tensor: torch.Tensor) -> str:
    names = {
        torch.float64: "F64",
        torch.float32: "F32",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.int64: "I64",
        torch.int32: "I32",
        torch.int16: "I16",
        torch.int8: "I8",
        torch.uint64: "U64",
        torch.uint32: "U32",
        torch.uint16: "U16",
        torch.uint8: "U8",
        torch.bool: "BOOL",
    }
    optional_names = {
        "float8_e4m3fn": "F8_E4M3",
        "float8_e5m2": "F8_E5M2",
        "float8_e8m0fnu": "F8_E8M0",
        "float4_e2m1fn_x2": "F4",
    }
    names.update(
        (getattr(torch, name), safetensors_name)
        for name, safetensors_name in optional_names.items()
        if hasattr(torch, name)
    )
    try:
        return names[tensor.dtype]
    except KeyError as exc:
        raise TypeError(f"Cannot serialize dtype {tensor.dtype}") from exc


def spool_tensor(spool: BinaryIO, path: Path, name: str, tensor: torch.Tensor) -> OutputTensor:
    value = tensor.detach().contiguous().cpu()
    shape = tuple(value.shape)
    offset = spool.tell()
    # Viewing a 0-D scalar as a dtype with a different element size is rejected
    # by PyTorch. NVFP4's per-tensor weight_scale_2 is such an FP32 scalar, so
    # flatten only the temporary byte view while retaining the original shape
    # in the safetensors header.
    raw = value.reshape(-1).view(torch.uint8).numpy()
    spool.write(memoryview(raw).cast("B"))
    size = spool.tell() - offset
    output = OutputTensor(name, tensor_dtype_name(value), shape, path, offset, size)
    del value, raw
    return output


def quantize_to_spool(
    tensors: list[SourceTensor],
    selected: dict[str, str],
    method: str,
    device: torch.device,
    spool_path: Path,
) -> tuple[list[OutputTensor], dict[str, dict[str, object]]]:
    outputs: list[OutputTensor] = []
    layer_configs: dict[str, dict[str, object]] = {}
    by_path: dict[Path, list[SourceTensor]] = {}
    for tensor in tensors:
        by_path.setdefault(tensor.path, []).append(tensor)

    completed = 0
    started = time.time()
    with spool_path.open("w+b") as spool:
        for path, entries in by_path.items():
            with safe_open(path, framework="pt", device="cpu") as handle:
                for source in entries:
                    layer = selected.get(source.name)
                    if layer is None:
                        outputs.append(
                            OutputTensor(
                                source.name,
                                source.dtype,
                                source.shape,
                                source.path,
                                source.offset,
                                source.size,
                            )
                        )
                        continue
                    weight = handle.get_tensor(source.name)
                    # BF16 conversion is a CPU-only byte transformation. NVFP4
                    # also narrows FP32 on CPU before the accelerator transfer,
                    # avoiding an unnecessary full-size FP32 GPU allocation.
                    if method in {"bf16", "nvfp4"} and weight.dtype == torch.float32:
                        weight = weight.to(torch.bfloat16)
                    if method != "bf16":
                        weight = weight.to(device)
                    quantized, config = quantize_weight(weight, method)
                    prefix = f"{layer}."
                    for suffix, value in quantized.items():
                        outputs.append(spool_tensor(spool, spool_path, prefix + suffix, value))
                    if method != "bf16":
                        config_bytes = json.dumps(config, separators=(",", ":")).encode("utf-8")
                        outputs.append(
                            spool_tensor(
                                spool,
                                spool_path,
                                prefix + "comfy_quant",
                                torch.tensor(list(config_bytes), dtype=torch.uint8),
                            )
                        )
                    layer_configs[layer] = config
                    completed += 1
                    del weight, quantized, value
                    if completed % 32 == 0:
                        gc.collect()
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                    elapsed = time.time() - started
                    print(f"[{completed}/{len(selected)}] {layer} ({elapsed:.1f}s)", flush=True)
    return outputs, layer_configs


def copy_range(source: Path, source_offset: int, length: int, output: BinaryIO) -> None:
    remaining = length
    with source.open("rb") as handle:
        handle.seek(source_offset)
        while remaining:
            chunk = handle.read(min(16 * 1024 * 1024, remaining))
            if not chunk:
                raise EOFError(f"Unexpected EOF in {source}")
            output.write(chunk)
            remaining -= len(chunk)


def write_checkpoint(output: Path, tensors: list[OutputTensor], metadata: dict[str, str]) -> None:
    offset = 0
    header: dict[str, object] = {"__metadata__": metadata}
    for tensor in tensors:
        if tensor.name in header:
            raise ValueError(f"Duplicate output tensor: {tensor.name}")
        header[tensor.name] = {
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + tensor.size],
        }
        offset += tensor.size
    raw_header = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    raw_header += b" " * ((8 - len(raw_header) % 8) % 8)

    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".partial", dir=output.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(struct.pack("<Q", len(raw_header)))
            handle.write(raw_header)
            for index, tensor in enumerate(tensors, 1):
                copy_range(tensor.path, tensor.offset, tensor.size, handle)
                if index % 128 == 0 or index == len(tensors):
                    print(f"wrote {index}/{len(tensors)} tensors", flush=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def validate_checkpoint(path: Path, expected_method: str, expected_layers: int) -> None:
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        keys = set(handle.keys())
    if expected_method == "bf16":
        if metadata.get("sensenova_quantization") is not None:
            raise ValueError("BF16 conversion output must not be marked as quantized")
        if metadata.get("sensenova_linear_conversion") != "fp32_to_bf16":
            raise ValueError("Output BF16 conversion metadata is invalid")
        if metadata.get("sensenova_converted_linear_count") != str(expected_layers):
            raise ValueError("Output BF16 converted Linear count is invalid")
    elif metadata.get("sensenova_quantization") != expected_method:
        raise ValueError("Output quantization metadata is invalid")
    if metadata.get(ASSETS_FORMAT_KEY) != ASSETS_FORMAT:
        raise ValueError("Output does not contain valid embedded checkpoint assets metadata")
    found = sum(key.endswith(".comfy_quant") for key in keys)
    expected_descriptors = 0 if expected_method == "bf16" else expected_layers
    if found != expected_descriptors:
        raise ValueError(
            f"Expected {expected_descriptors} quantized layers, found {found}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="HF repo id, local HF snapshot, or source safetensors checkpoint")
    parser.add_argument("output", type=Path, help="Output .safetensors checkpoint")
    parser.add_argument("--method", choices=QUANT_METHODS, required=True)
    parser.add_argument("--revision", default="", help="Hugging Face revision")
    parser.add_argument("--token", default="", help="Hugging Face access token")
    parser.add_argument(
        "--device",
        default="cuda",
        help="Quantization device, normally cuda or cuda:N; bf16 conversion always runs on CPU",
    )
    parser.add_argument("--include", help="Only quantize Linear module names matching this regex")
    parser.add_argument("--exclude", action="append", default=[], help="Keep matching Linear modules in source precision; repeatable")
    parser.add_argument("--dry-run", action="store_true", help="Resolve source and list the quantization plan without reading weights")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.suffix.lower() not in {".safetensors", ".sft"}:
        raise ValueError("Output filename must end in .safetensors or .sft")
    if output.exists() and not args.overwrite:
        raise FileExistsError("Output exists; pass --overwrite to replace it")

    source = resolve_source(args.source, args.revision, args.token)
    if output in source.files:
        raise ValueError("In-place quantization is not supported; choose a different output name")
    tensors, source_metadata = collect_source_tensors(source.files)
    linear_names, model_keys, transformers_version = private_model_layout(source.assets)
    source_keys = {tensor.name for tensor in tensors}
    missing = sorted(model_keys - source_keys)
    if missing:
        raise ValueError(
            f"Checkpoint does not match the patched Transformer model; missing: {', '.join(missing[:8])}"
        )
    selected = select_linear_weights(
        tensors,
        linear_names,
        re.compile(args.include) if args.include else None,
        tuple(re.compile(value) for value in args.exclude),
    )
    if args.method == "bf16":
        source_by_name = {tensor.name: tensor for tensor in tensors}
        selected = {
            name: layer
            for name, layer in selected.items()
            if source_by_name[name].dtype == "F32"
        }
        if not selected:
            raise ValueError("No FP32 Linear weights matched the checkpoint and filters")
    source_bytes = sum(tensor.size for tensor in tensors)
    print(f"source: {source.label}")
    print(f"private transformers: {transformers_version} ({PINNED_TRANSFORMERS})")
    print(f"checkpoint tensors: {len(tensors)}, size: {source_bytes / 2**30:.2f} GiB")
    print(f"Linear modules selected: {len(selected)}/{len(linear_names)}")
    if args.method == "bf16":
        print(
            f"BF16 conversion: {len(selected)} FP32 Linear weights will be converted; "
            "existing BF16 and non-Linear tensors will be copied unchanged"
        )
    if args.method == "nvfp4":
        source_by_name = {tensor.name: tensor for tensor in tensors}
        fp32_count = sum(
            source_by_name[name].dtype == "F32" for name in selected
        )
        if fp32_count:
            print(
                f"NVFP4 input conversion: {fp32_count} FP32 Linear weights "
                "will be converted to BF16 before quantization"
            )
    if args.dry_run:
        for name in selected.values():
            print(name)
        return 0

    device = torch.device(args.device)
    if args.method != "bf16" and device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; pass --device cpu only for formats supported on CPU")
    output.parent.mkdir(parents=True, exist_ok=True)
    spool_fd, spool_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".quant-spool", dir=output.parent)
    os.close(spool_fd)
    spool_path = Path(spool_name)
    try:
        outputs, layer_configs = quantize_to_spool(tensors, selected, args.method, device, spool_path)
        if PIXEL_VAE_KEY not in {tensor.name for tensor in outputs}:
            with spool_path.open("ab") as spool:
                outputs.append(spool_tensor(spool, spool_path, PIXEL_VAE_KEY, torch.tensor(1.0)))
        processing_metadata = (
            {
                "sensenova_linear_conversion": "fp32_to_bf16",
                "sensenova_converted_linear_count": str(len(layer_configs)),
            }
            if args.method == "bf16"
            else {
                "sensenova_quantization": args.method,
                "sensenova_quantized_linear_count": str(len(layer_configs)),
                "_quantization_metadata": json.dumps(
                    {"format_version": "1.0", "layers": layer_configs},
                    separators=(",", ":"),
                ),
            }
        )
        metadata = {
            **source_metadata,
            "format": "pt",
            "comfyui_model_family": "sensenova_u1",
            "sensenova_checkpoint_version": "3",
            "sensenova_transformers": PINNED_TRANSFORMERS,
            "sensenova_source": source.label,
            "sensenova_pixel_space_vae": "true",
            **processing_metadata,
            **build_assets_metadata(source.assets),
        }
        write_checkpoint(output, outputs, metadata)
        validate_checkpoint(output, args.method, len(layer_configs))
    finally:
        spool_path.unlink(missing_ok=True)

    print(f"checkpoint: {output}")
    print("assets:     embedded in checkpoint")
    action = "converted" if args.method == "bf16" else "quantized"
    print(f"{action} Linear layers: {len(layer_configs)} ({args.method})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
