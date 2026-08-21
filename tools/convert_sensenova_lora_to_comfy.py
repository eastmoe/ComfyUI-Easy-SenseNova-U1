#!/usr/bin/env python3
"""Convert a SenseNova Hugging Face LoRA to ComfyUI's generic LoRA keys.

SenseNova LoRAs name targets like ``language_model.model.layers...`` while
ComfyUI exposes the same weights below ``diffusion_model``.  This converter
adds that prefix without decoding or changing any tensor data.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import tempfile
from pathlib import Path
from typing import BinaryIO


COMFY_PREFIX = "diffusion_model."
LORA_SUFFIXES = (".lora_down.weight", ".lora_up.weight", ".alpha")


def read_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"Not a safetensors file (short header): {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        raw_header = handle.read(header_length)
        if len(raw_header) != header_length:
            raise ValueError(f"Not a safetensors file (short JSON header): {path}")
    try:
        return json.loads(raw_header.rstrip(b" \t\r\n\0")), 8 + header_length
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid safetensors header: {path}") from exc


def converted_header(header: dict) -> tuple[bytes, int]:
    metadata = dict(header.get("__metadata__", {}))
    metadata.update(
        {
            "format": "pt",
            "comfyui_lora_format": "generic",
            "comfyui_model_family": "sensenova_u1",
        }
    )
    converted: dict[str, object] = {"__metadata__": metadata}
    tensor_count = 0
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if not name.endswith(LORA_SUFFIXES):
            raise ValueError(f"Unsupported tensor in SenseNova LoRA: {name}")
        target = name if name.startswith(COMFY_PREFIX) else COMFY_PREFIX + name
        if target in converted:
            raise ValueError(f"Duplicate converted tensor name: {target}")
        converted[target] = info
        tensor_count += 1
    if not tensor_count:
        raise ValueError("The input contains no LoRA tensors")
    raw = json.dumps(converted, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return raw + b" " * ((8 - len(raw) % 8) % 8), tensor_count


def copy_payload(source: BinaryIO, output: BinaryIO) -> None:
    while chunk := source.read(16 * 1024 * 1024):
        output.write(chunk)


def convert(source: Path, output: Path) -> int:
    header, payload_offset = read_header(source)
    new_header, tensor_count = converted_header(header)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    try:
        with source.open("rb") as input_handle, os.fdopen(fd, "wb") as output_handle:
            input_handle.seek(payload_offset)
            output_handle.write(struct.pack("<Q", len(new_header)))
            output_handle.write(new_header)
            copy_payload(input_handle, output_handle)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return tensor_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Input SenseNova LoRA .safetensors")
    parser.add_argument("output", type=Path, help="Output ComfyUI LoRA .safetensors")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if source.suffix.lower() != ".safetensors" or output.suffix.lower() != ".safetensors":
        raise ValueError("Input and output filenames must end in .safetensors")
    if source == output:
        raise ValueError("Input and output must be different files")
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists() and not args.overwrite:
        raise FileExistsError("Output already exists; pass --overwrite to replace it")
    tensor_count = convert(source, output)
    print(f"Converted {tensor_count} tensors")
    print(f"LoRA: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
